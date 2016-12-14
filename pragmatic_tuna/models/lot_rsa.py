from argparse import ArgumentParser
from collections import Counter, defaultdict, namedtuple
from itertools import permutations
import json
import re

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import trange, tqdm
import nltk

from pragmatic_tuna.environments import TUNAWithLoTEnv
from pragmatic_tuna.environments.spatial import *
from pragmatic_tuna.reinforce import reinforce_episodic_gradients
from pragmatic_tuna.util import colors


EMBEDDING_INTITIALIZER = tf.truncated_normal_initializer()


class NaiveGenerativeModel(object):

    """
    A very stupid generative utterance model $p(u | z)$ which is intended to
    map from bag-of-features $z$ representations to bag-of-words $u$
    representations. Optionally performs add-1 smoothing.
    """

    smooth_val = 0.1

    def __init__(self, vocab_size, max_length, smooth=True):
        self.smooth = smooth
        self.counter = defaultdict(lambda: Counter())
        self.vocab_size = vocab_size
        self.max_length = max_length

    def observe(self, obs, gold_lf):
        u = obs[1]
        z = gold_lf

        z, u = tuple(z), tuple(u)
        self.counter[z][u] += 1

    def score(self, z, u, u_seq):
        """Retrieve unnormalized p(u|z)"""
        # TODO: weight on Z?
        z, u = tuple(z), tuple(u)
        score = self.counter[z][u]
        if self.smooth:
            # Add-1 smoothing.
            score += self.smooth_val
        return np.exp(score)

    def sample(self, z):
        """Sample from the distribution p(u|z)"""
        g_dict = self.counter[tuple(z)]
        keys = list(g_dict.keys())
        values = np.array(list(g_dict.values()))

        if self.smooth:
            # Allow that we might sample one of the unseen u's.
            mass_seen = np.exp(values + self.smooth_val).sum()
            n_unseen = 2 ** self.max_length - len(g_dict)
            mass_unseen = np.exp(self.smooth_val) * (n_unseen)
            p_unseen = mass_unseen / (mass_unseen + mass_seen)

            if np.random.random() < p_unseen:
                print("Rejection!")
                # Rejection-sample a random unseen utterance.
                done = False
                while not done:
                    length = np.random.randint(1, self.max_length + 1)
                    idxs = np.random.randint(self.vocab_size, size=length)
                    utt = np.zeros(self.vocab_size)
                    utt[idxs] = 1
                    utt = tuple(utt)
                    done = utt not in keys
                return utt

        distr = np.exp(values - values.max())
        distr /= distr.sum()
        return keys[np.random.choice(len(keys), p=distr)]


class DiscreteGenerativeModel(object):

    """
    A generative model that maps atoms and functions of a logical form
    z to and words of an utterance u and scores the fluency of the utterance
    using a bigram language model.
    """

    smooth_val = 0.1
    unk_prob = 0.01
    #if set to 0, use +1 smoothing of bigrams instead of backoff LM
    backoff_factor = 0

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, env, max_timesteps=4, smooth=True):
        self.smooth = smooth
        self.vocab_size = env.vocab_size
        self.max_conjuncts = max_timesteps / 2
        self.env = env

        self.counter = defaultdict(lambda: Counter())
        self.bigramcounter = defaultdict(lambda: Counter())
        self.unigramcounter = Counter()

    def observe(self, obs, gold_lf):
        u = obs[2]
        z = gold_lf

        for lf_token in z:
            self.counter[lf_token].update(u)

        words = []
        words.extend(u)
        #words.append(self.END_TOKEN)

        prev_word = self.START_TOKEN
        for word in words:
            self.bigramcounter[prev_word][word] +=1
            self.unigramcounter[word] +=1
            prev_word = word

    def _score_word_atom(self, word, atom):
        score = self.counter[atom][word]
        denom = sum(self.counter[atom].values())
        if self.smooth:
          score += 1
          denom += len(self.env.vocab)
        return float(score) / denom

    def _score_bigram(self, w1, w2):
        score = self.bigramcounter[w1][w2]
        denom = sum(self.bigramcounter[w1].values())
        if self.smooth and self.backoff_factor == 0:
          score += 1
          denom += len(self.env.vocab)
        if score < 1 :
            return 0
        return np.log(float(score) / denom)


    def _score_unigram(self, w):
        score = self.unigramcounter[w]
        if score < 1:
          prob = self.unk_prob
        else:
          prob = float(score) / sum(self.unigramcounter.values())

        return np.log(prob)


    def _score_sequence(self, u):
        prev_word = self.START_TOKEN

        words = []
        words.extend(u)
        #words.append(self.END_TOKEN)
        prob = 0
        for word in u:
            p_bigram = self._score_bigram(prev_word, word)
            p_bigram = ((p_bigram + np.log(1-self.backoff_factor))
                            if p_bigram < 0
                            else self._score_unigram(word) + np.log(self.backoff_factor))
            prob += p_bigram
            prev_word = word

        return prob

    def score(self, z, u_bag, u):
        # Limit utterance lengths to LF length.
        if len(u) != len(z):
            return -np.Inf
        #compute translation probability p(u|z)
        words = u
        alignments = permutations(range(len(z)))
        p_trans = 0
        for a in alignments:
            p = 1
            for i, w in enumerate(words):
                p *= self._score_word_atom(w, z[a[i]])
            p_trans += p

        p_trans = np.log(p_trans)

        #compute fluency probability, i.e., lm probability
        p_seq  = self._score_sequence(u)

        return 0.1 * p_seq + 0.9 * p_trans


    def sample_with_alignment(self, z, alignment):
        unigram_denom = max(sum(self.unigramcounter.values()), 1.0)
        unigram_probs = np.array(list(self.unigramcounter.values())) / unigram_denom
        keys = list(self.unigramcounter.keys())

        prev_word = self.START_TOKEN

        u = []
        ps = []
        i = 0
        for i in range(len(z)):

            bigram_counts = np.array([self.bigramcounter[prev_word][w]
                                        for w in keys])
            bigram_denom = max(sum(bigram_counts), 1.0)
            bigram_probs = bigram_counts / bigram_denom

            cond_probs = np.array([self._score_word_atom(w, z[alignment[i]]) for w in keys])

            interpolated = bigram_probs * (1 - self.backoff_factor) + unigram_probs * self.backoff_factor
            distr = interpolated * cond_probs
            distr = distr / np.sum(distr)

            idx = np.random.choice(len(keys), p=distr)
            word = keys[idx]
            u.append(word)
            prev_word = word
            ps.append(distr[idx])

        p = np.exp(np.sum(np.log(ps)))
        return " ".join(u), p

    def sample(self, z):
        alignments = permutations(range(len(z)))
        utterances = []
        distr = []
        for a in alignments:
            u, p = self.sample_with_alignment(z, a)
            utterances.append(u)
            distr.append(p)

        distr = np.array(distr) / np.sum(distr)
        idx = np.random.choice(len(utterances), p=distr)
        return utterances[idx]


class WindowedSequenceSpeakerModel(object):

    """
    Windowed sequence speaker/decoder model that mirrors
    `WindowedSequenceListenerModel`.
    """
    # could probably be unified with WindowedSequenceListenerModel if there is
    # sufficient motivation.

    def __init__(self, env, scope="speaker", max_timesteps=4,
                 word_embeddings=None, lf_embeddings=None, embedding_dim=10):
        self.env = env
        self._scope_name = scope
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

        self.train_op = None

        self._build_embeddings(word_embeddings, lf_embeddings)
        self._build_graph()

    def _build_embeddings(self, word_embeddings, lf_embeddings):
        with tf.variable_scope(self._scope_name):
            emb_shape = (self.env.vocab_size, self.embedding_dim)
            if word_embeddings is None:
                word_embeddings = tf.get_variable("word_embeddings", emb_shape)
            assert tuple(word_embeddings.get_shape().as_list()) == emb_shape

            lf_emb_shape = (len(self.env.lf_vocab), self.embedding_dim)
            if lf_embeddings is None:
                lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape)
            assert tuple(lf_embeddings.get_shape().as_list()) == lf_emb_shape

            self.word_embeddings = word_embeddings
            self.lf_embeddings = lf_embeddings

    def _build_graph(self):
        with tf.variable_scope(self._scope_name):
            self.lf_toks = tf.placeholder(tf.int32, shape=(self.max_timesteps,),
                                          name="lf_toks")

            lf_window = tf.nn.embedding_lookup(self.lf_embeddings, self.lf_toks)
            lf_window = tf.reshape(lf_window, (-1,))

            null_embedding = tf.gather(self.word_embeddings, self.env.word_unk_id)

            # Now run a teeny utterance decoder.
            outputs, probs, samples = [], [], []
            output_dim = self.env.vocab_size
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(0, [prev_sample, lf_window])
                    output_t = layers.fully_connected(tf.expand_dims(input_t, 0),
                                                      output_dim, tf.identity)
                    probs_t = tf.squeeze(tf.nn.softmax(output_t))

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1))
                    prev_sample = tf.nn.embedding_lookup(self.word_embeddings, sample_t)

                    # TODO support stop token here?

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.probs = probs
            self.samples = samples

    def build_xent_gradients(self):
        gold_words = [tf.zeros((), dtype=tf.int32, name="gold_word_%i" % t)
                      for t in range(self.max_timesteps)]
        gold_length = tf.placeholder(tf.int32, shape=(), name="gold_length")
        losses = [tf.to_float(t < gold_length) *
                  tf.nn.sparse_softmax_cross_entropy_with_logits(
                          tf.squeeze(output_t), gold_word_t)
                  for t, (output_t, gold_word_t)
                  in enumerate(zip(self.outputs, gold_words))]
        loss = tf.add_n(losses) / tf.to_float(gold_length)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_words = gold_words
        self.xent_gold_length = gold_length
        self.xent_gradients = zip(gradients, params)

        return ((self.xent_gold_words, self.xent_gold_length),
                (self.xent_gradients,))

    def _pad_lf_idxs(self, z):
        assert len(z) <= self.max_timesteps
        z_new = z + [self.env.lf_unk_id] * (self.max_timesteps - len(z))
        return z_new

    def sample(self, z):
        sess = tf.get_default_session()
        feed = {self.lf_toks: self._pad_lf_idxs(z)}

        sample = sess.run(self.samples, feed)
        try:
            stop_idx = sample.index(self.env.word_eos_id)
            sample = sample[:stop_idx]
        except ValueError:
            # No stop token. No trimming necessary. Pass.
            pass

        return " ".join(self.env.vocab[idx] for idx in sample)

    def score(self, z, u_bag, u):
        sess = tf.get_default_session()

        z = self._pad_lf_idxs(z)
        words = [self.env.word2idx[word] for word in u]

        feed = {self.lf_toks: z}
        feed.update({self.samples[t]: word for t, word in enumerate(words)})

        probs = sess.run(self.probs[:len(words)], feed)
        probs = [probs_t[word_t] for probs_t, word_t in zip(probs, words)]
        return np.prod(probs)

    def observe(self, obs, gold_lf):
        z = self._pad_lf_idxs(gold_lf)

        words = [self.env.word2idx[word] for word in obs[2]]
        real_length = min(len(words) + 1, self.max_timesteps) # train to output a single EOS token
        # Add a EOS token to words
        if len(words) < self.max_timesteps:
            words.append(self.env.word_eos_id)

        sess = tf.get_default_session()
        feed = {self.lf_toks: z, self.xent_gold_length: real_length}
        feed.update({self.xent_gold_words[t]: word_t
                     for t, word_t in enumerate(words)})
        sess.run(self.train_op, feed)


class ListenerModel(object):

    """
    Parametric listener model $q_\\theta(z|u)$ which maps utterances to LF
    representations.
    """

    def __init__(self, env, scope="listener"):
        assert not env.bag
        self.env = env
        self._scope = tf.variable_scope(scope)
        self.feeds = []
        self.train_op = None

        self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def build_rl_gradients(self):
        raise NotImplementedError

    def build_xent_gradients(self):
        raise NotImplementedError

    def sample(self, utterance_bag, words):
        raise NotImplementedError

    def observe(self, obs, lf_pred, reward, gold_lf):
        raise NotImplementedError

    def reset(self):
        pass

class SimpleListenerModel(ListenerModel):

    def _build_graph(self):
        """
        Build the core model graph.
        """
        n_outputs = len(self.env.lf_functions) * len(self.env.lf_atoms)

        with self._scope:
            self.items = tf.placeholder(tf.float32, shape=(None, self.env.attr_dim))
            self.utterance = tf.placeholder(tf.float32, shape=(self.env.vocab_size,))

            self.scores = layers.fully_connected(tf.expand_dims(self.utterance, 0),
                                                 n_outputs, tf.identity)
            self.probs = tf.squeeze(tf.nn.softmax(self.scores))

        self.feeds.extend([self.items, self.utterance])

    def build_rl_gradients(self):
        if hasattr(self, "rl_action"):
            return (self.rl_action, self.rl_reward), (self.rl_gradients,)

        action = tf.placeholder(tf.int32, shape=(), name="action")
        reward = tf.placeholder(tf.float32, shape=(), name="reward")

        scores = [self.scores]
        actions = [action]
        rewards = reward
        gradients = reinforce_episodic_gradients(scores, actions, reward)

        self.rl_action = action
        self.rl_reward = reward
        self.rl_gradients = gradients

        self.feeds.extend([self.rl_action, self.rl_reward])

        return (action, reward), (gradients,)

    def build_xent_gradients(self):
        """
        Assuming the client can determine some gold-standard LF for a given
        trial, we can simply train by cross-entropy (maximize log prob of the
        gold output).
        """
        gold_lf = tf.placeholder(tf.int32, shape=(), name="gold_lf")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                tf.squeeze(self.scores), gold_lf)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_lf = gold_lf
        self.xent_gradients = zip(gradients, params)

        self.feeds.extend([self.xent_gold_lf])

        return (gold_lf,), (gradients,)

    def _list_to_id(self, id_list):
        """
        Convert an LF token ID list to an action ID.
        """
        fn_tok_id, atom_tok_id = id_list

        # Convert ID sequence back to our hacky space.
        fn_name = self.env.lf_vocab[fn_tok_id]
        fn_id = self.env.lf_functions.index(fn_name)
        atom = self.env.lf_vocab[atom_tok_id]
        atom_id = self.env.lf_atoms.index(atom)

        action_id = fn_id * len(self.env.lf_atoms) + atom_id
        return action_id

    def _id_to_list(self, idx):
        """
        Convert an action ID to an LF token ID list.
        """
        fn_id = idx // len(self.env.lf_atoms)
        atom_id = idx % len(self.env.lf_atoms)
        fn_name = self.env.lf_functions[fn_id]
        atom_name = self.env.lf_atoms[atom_id]
        token_ids = [self.env.lf_token_to_id[fn_name],
                     self.env.lf_token_to_id[atom_name]]
        return token_ids

    def sample(self, utterance_bag, words):
        sess = tf.get_default_session()
        probs = sess.run(self.probs, {self.utterance: utterance_bag})
        lf = np.random.choice(len(probs), p=probs)

        # Jump through some hoops to make sure we sample a valid fn(atom) LF
        return self._id_to_list(lf)

    def observe(self, obs, lf_pred, reward, gold_lf):
        lf_pred = self._list_to_id(lf_pred)
        if gold_lf is not None:
            gold_lf = self._list_to_id(gold_lf)

        if hasattr(self, "rl_action"):
            train_feeds = {self.utterance: obs[1],
                           self.rl_action: lf_pred,
                           self.rl_reward: reward}
        elif hasattr(self, "xent_gold_lf"):
            if gold_lf is None:
                # TODO log?
                return
            train_feeds = {self.utterance: obs[1],
                           self.xent_gold_lf: gold_lf}
        else:
            raise RuntimeError("no gradients defined")

        sess = tf.get_default_session()
        sess.run(self.train_op, train_feeds)


class WindowedSequenceListenerModel(ListenerModel):

    """
    Parametric listener model $q_\\theta(z|u) which maps utterances (sequences)
    to LF representations (factored as sequences).

    This model takes a window of embedding inputs and outputs a sequence of LF
    tokens.
    """

    def __init__(self, env, scope="listener", max_timesteps=4, embedding_dim=10):
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim
        super(WindowedSequenceListenerModel, self).__init__(env, scope=scope)

    def _build_graph(self):
        with self._scope:
            self.temperature = tf.constant(1.0, name="sampling_temperature")

            # TODO: padding representation?
            self.words = tf.placeholder(tf.int32, shape=(self.max_timesteps,),
                                        name="words")

            emb_shape = (self.env.vocab_size, self.embedding_dim)
            word_embeddings = tf.get_variable(
                    "word_embeddings", shape=emb_shape, initializer=EMBEDDING_INTITIALIZER)

            word_window = tf.nn.embedding_lookup(word_embeddings, self.words)
            word_window = tf.reshape(word_window, (-1,))

            # Create embeddings for LF tokens
            lf_emb_shape = (len(self.env.lf_vocab), self.embedding_dim)
            lf_embeddings = tf.get_variable(
                    "lf_embeddings", shape=lf_emb_shape, initializer=EMBEDDING_INTITIALIZER)
            null_embedding = tf.gather(lf_embeddings, self.env.lf_unk_id)

            # Now run a teeny LF decoder.
            outputs, samples = [], []
            output_dim = lf_emb_shape[0]
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(0, [prev_sample, word_window])
                    output_t = layers.fully_connected(tf.expand_dims(input_t, 0),
                                                      output_dim, tf.identity)

                    output_t /= self.temperature

                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1))
                    # Hack shape.
                    sample_t.set_shape(())
                    prev_sample = tf.nn.embedding_lookup(lf_embeddings, sample_t)

                    # TODO support stop token here?

                    outputs.append(output_t)
                    samples.append(sample_t)

            self.word_embeddings = word_embeddings
            self.lf_embeddings = lf_embeddings

            self.outputs = outputs
            self.samples = samples

    def build_xent_gradients(self):
        gold_lf_tokens = [tf.zeros((), dtype=tf.int32, name="gold_lf_%i" % t)
                          for t in range(self.max_timesteps)]
        gold_lf_length = tf.placeholder(tf.int32, shape=(), name="gold_lf_length")
        losses = [tf.to_float(t < gold_lf_length) *
                  tf.nn.sparse_softmax_cross_entropy_with_logits(
                          tf.squeeze(output_t), gold_lf_t)
                  for t, (output_t, gold_lf_t)
                  in enumerate(zip(self.outputs, gold_lf_tokens))]
        loss = tf.add_n(losses) / tf.to_float(gold_lf_length)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_lf_tokens = gold_lf_tokens
        self.xent_gold_lf_length = gold_lf_length
        self.xent_gradients = zip(gradients, params)

        self.feeds.extend([self.xent_gold_lf_tokens, self.xent_gold_lf_length])

        return ((self.xent_gold_lf_tokens, self.xent_gold_lf_length),
                (self.xent_gradients,))

    def _get_word_idxs(self, words):
        # Look up word indices. TODO: padding with something other than UNK..?
        word_idxs = [self.env.word2idx[word] for word in words]
        assert len(word_idxs) <= self.max_timesteps
        word_idxs += [self.env.word_unk_id] * (self.max_timesteps - len(word_idxs))
        return word_idxs

    def sample(self, utterance_bag, words, temperature=1.0):
        sess = tf.get_default_session()
        feed = {self.words: self._get_word_idxs(words),
                self.temperature: temperature}
        # Rejection-sample a valid LF (alternating fn-atom-fn-atom..)
        attempts = 0
        while True:
            sample = sess.run(self.samples, feed)
            ret_sample = []
            valid = True
            for i, sample_i in enumerate(sample):
                stop = sample_i == self.env.lf_eos_id
                if stop and i > 0 and i % 2 == 0:
                    # Valid stopping point. Truncate and end.
                    ret_sample = ret_sample[:i]
                    break

                ret_sample.append(sample_i)
                sample_i_str = self.env.lf_vocab[sample_i]
                if i % 2 == 0 and sample_i_str not in self.env.lf_function_map:
                    valid = False
                elif i % 2 == 1 and sample_i_str not in self.env.lf_atoms:
                    valid = False

            if len(ret_sample) > 3 and ret_sample[0:2] == ret_sample[2:4]:
                ret_sample = ret_sample[:2]

            if valid and len(ret_sample) == len(words):
                return ret_sample

            attempts += 1
            if attempts > 10000:
                print("%sFailed to sample LF for %r after 10000 attempts. Dying.%s"
                      % (colors.FAIL, " ".join(words), colors.ENDC))
                return []

    def observe(self, obs, lf_pred, reward, gold_lf):
        if gold_lf is None:
            return

        # Pad with stop tokens
        real_length = min(self.max_timesteps, len(gold_lf) + 1) # train to output a single stop token
        if len(gold_lf) < self.max_timesteps:
            gold_lf.extend([self.env.lf_eos_id] * (self.max_timesteps - len(gold_lf)))

        feed = {self.words: self._get_word_idxs(obs[2]),
                self.xent_gold_lf_length: real_length}
        feed.update({self.xent_gold_lf_tokens[t]: gold_lf[t]
                     for t in range(self.max_timesteps)})

        sess = tf.get_default_session()
        sess.run(self.train_op, feed)

class SkipGramListenerModel(ListenerModel):
    """
        MaxEnt model p(z|u) with skip-gram features of utterance
        and logical form.
    """


    def __init__(self, env, scope="listener"):
        self.vocab_size = len(env.word2idx.keys())
        self.lf_vocab_size = len(env.lf_vocab)
        self.word_feat_count = self.vocab_size * (self.vocab_size + 1)
        self.lf_feat_count = self.lf_vocab_size * (self.lf_vocab_size + 1)

        self.feat_count = self.word_feat_count * self.lf_feat_count

        self.l1_reg = 0.5

        self.reset()
        super(SkipGramListenerModel, self).__init__(env, scope=scope)


    def _build_graph(self):
        with self._scope:
            self.weights = tf.get_variable("weights", shape=(self.feat_count, 1),
                                             initializer=tf.constant_initializer(0))
            self.feats = tf.placeholder(tf.float32, shape=(None, self.feat_count))


            self.scores = tf.squeeze(tf.matmul(self.feats, self.weights), [1])
            self.probs = tf.nn.softmax(self.scores)



    """
        Convert internal LF of form left(dog,cat) to LoT expression
        id(cat) AND left(dog)

        Assumes at most 2 conjuncts.
    """
    def to_lot_lf(self, lf):
        #case 1: id(x)
        if len(lf) == 1:
            return [self.env.lf_token_to_id["id"], lf[0]]
        #case 2: fn(x)
        elif len(lf) == 2:
            return [lf[0], lf[1]]
        #case 3:
        elif len(lf) == 3:
            return [lf[0],
                    lf[1],
                    self.env.lf_token_to_id["id"],
                    lf[2]]

    """
        Convert LoT expression of the form id(cat) AND left(dog) to
        internal LF of form left(dog,cat)

        Assumes at most 2 conjuncts and that one of them has the form id(x).
    """

    def from_lot_lf(self, lot):

        id_idx = self.env.lf_token_to_id["id"]

        if len(lot) == 2:
            if lot[0] == id_idx:
                return lot[1:]
            else:
                return lot
        elif len(lot) == 4:
            if id_idx not in lot:
                print("%sfrom_lot_lf: Invalid LF.%s" % (colors.FAIL, colors.ENDC))
                return []
            else:
                if lot[0] == id_idx:
                    #id(x) and fn(y)
                    return [lot[2], lot[3], lot[1]]
                else:
                    #fn(x) and id(y)
                    return [lot[0], lot[1], lot[3]]

        else:
            print("%sfrom_lot_lf: Invalid LF.%s" % (colors.FAIL, colors.ENDC))
            return []

    def build_xent_gradients(self):
        """
        Assuming the client can determine some gold-standard LF for a given
        trial, we can simply train by cross-entropy (maximize log prob of the
        gold output).
        """

        self.gold_lfs = tf.placeholder(tf.float32, shape=(None))

        loss = tf.nn.softmax_cross_entropy_with_logits(tf.squeeze(self.scores), tf.squeeze(self.gold_lfs))
        loss += self.l1_reg * tf.reduce_sum(tf.abs(self.weights))

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gradients = zip(gradients, params)


        return (self.gold_lfs, None), (gradients,)


    def featurize_words(self, words):

        word_idxs = [self.env.word2idx[word] for word in words]


        #unigrams
        skip_gram_idxs = word_idxs
        #bigrams and bi-skip-grams
        for i in range(len(words)-1):
            #bigram (add-1, 0=no word, i.e., unigram)
            idx = word_idxs[i] + self.vocab_size * (word_idxs[i+1]+1)
            skip_gram_idxs.append(idx)

        #trigrams and tri-skip-grams:
        #for i in range(len(words)-2):
            #trigram (add-1, 0=no word
            #idx = word_idxs[i] + self.vocab_size * (word_idxs[i+1]+1) + self.vocab_size * (self.vocab_size + 1) * (word_idxs[i+2]+1)
            #skip_gram_idxs.append(idx)
            #skip-gram (word * word)
            #idx = word_idxs[i] + self.vocab_size * (self.vocab_size + 1) * (word_idxs[i+2]+1)
            #skip_gram_idxs.append(idx)




        #turn into one-hot vectors
        word_feats = np.zeros((self.word_feat_count,))
        word_feats[skip_gram_idxs] = 1
        word_feats = word_feats.reshape((1, self.word_feat_count))

        return word_feats

    def featurize_lf(self, lf):
        lf_idxs = []
        lf_idxs.extend(lf)

        if len(lf) > 1:
            idx = lf_idxs[0] + self.lf_vocab_size * (lf_idxs[1]+1)
            lf_idxs.append(idx)

        #if len(lf) > 2:
            #trigram (add-1, 0=no token)
            #idx = lf_idxs[0] + self.lf_vocab_size * (lf_idxs[1]+1) + self.lf_vocab_size * (self.lf_vocab_size + 1) * (lf_idxs[2]+1)
            #lf_idxs.append(idx)

        #turn into one-hot vectors
        lf_feats = np.zeros(self.lf_feat_count)
        lf_feats[lf_idxs] = 1
        return lf_feats


    def _populate_cache(self, words, test=False):
        id_idx = self.env.lf_token_to_id["id"]

        # HACK: pre-allocate lf_feats cache using known size
        num_lfs = (len(self.env.lf_atoms) * len(self.env.lf_functions)) ** 2
        all_lf_feats = np.empty((num_lfs, self.lf_feat_count))

        #TODO: tune this number?
        i = 0
        for lf_pref in self.env.enumerate_lfs(includeOnlyPossibleReferents=not test):
            for lf in self.env.enumerate_lfs(includeOnlyPossibleReferents=not test, lf_prefix=lf_pref):
                valid = len(lf) < 3 or id_idx in lf
                if not valid:
                    continue

                lf =  self.from_lot_lf(lf)
                self.lf_cache.append(lf)
                all_lf_feats[i] = self.featurize_lf(lf)
                i += 1

        self.lf_feats = all_lf_feats[:i]

        word_feats = self.featurize_words(words)
        #take cross product
        self.feat_matrix = np.zeros((len(self.lf_cache), self.feat_count))
        for i in range(len(self.lf_cache)):
            lf_feats = self.lf_feats[i].reshape(self.lf_feat_count, 1)

            self.feat_matrix[i] = np.dot(lf_feats, word_feats).reshape((self.feat_count,))

            #print(feats)

        sess = tf.get_default_session()
        feed = {self.feats: self.feat_matrix}

        self.probs_cache = sess.run(self.probs, feed)

    #TODO: iteratively sample, i.e., start with single predicate and then only consider LFs with that predicate
    def sample(self, utterance_bag, words, temperature=None, test=False):
        if len(self.lf_cache) < 1:
            self._populate_cache(words, test)

        if not test:
            idx = np.random.choice(len(self.probs_cache), p=self.probs_cache)

        else:
            idx = np.argmax(self.probs_cache)

        sampled_lf = self.to_lot_lf(self.lf_cache[idx])

        #print("####")
        #print(self.lf_cache[idx])
        #print(sampled_lf)
        #self.reset()
        return sampled_lf, self.probs_cache[idx]

    def reset(self):
        self.lf_cache = []
        self.lf_feats = None
        self.probs_cache = None


    def observe(self, obs, lf_pred, reward, gold_lf):
        if gold_lf is None:
            return

        #print(gold_lf)

        referent = self.env.resolve_lf(gold_lf)[0]

        #word_feats = self.featurize_words(obs[2])
        #take cross product


        self._populate_cache(obs[2], test=True)


        #lf =  self.from_lot_lf(gold_lf)
        #lf_feats = self.featurize_lf(lf).reshape((self.lf_feat_count, 1))
        #feats = np.dot(lf_feats, word_feats).reshape((1,self.feat_count))

        #go through all LFs, check if they



        gold_lfs = np.zeros((len(self.lf_cache),1))
        for i, lf in enumerate(self.lf_cache):
            lf = self.to_lot_lf(lf)
            if lf == gold_lf:
                gold_lfs[i] = 1.0

            #matches = self.env.resolve_lf(lf)
            #if matches and len(matches) == 1 and matches[0] == referent:
            #    gold_lfs[i] = 1.0

        gold_lfs /= np.sum(gold_lfs)


        train_feeds = {self.feats: self.feat_matrix,
                       self.gold_lfs: gold_lfs}

        sess = tf.get_default_session()
        sess.run(self.train_op, train_feeds)




def infer_trial(env, obs, listener_model, speaker_model, args):
    """
    Run RSA inference for a given trial.

    Args:
        env:
        obs: Observation from environment
        listener_model:
        speaker_model:
        args:

    Returns:
        lfs: LF IDs sampled from `listener_model`
        weights: Associated weight `p(z, u)` for each LF
        rejs_per_sample:
    """

    items, utterance_bag, words = obs

    # Sample N LFs from the predicted distribution, accepting only when they
    # resolve to a referent in the scene.
    lfs, g_lfs, weights = [], [], []
    num_rejections = 0
    while len(weights) < args.num_listener_samples:
        lf, p = listener_model.sample(utterance_bag, words)

        # Resolve referent.
        referent = env.resolve_lf(lf)
        if not referent:
            # Dereference failed. No object matched.
            num_rejections += 1
            continue
        referent = env._domain.index(referent[0])

        # Sample an LF z' ~ p(z|r).
        g_lf = env.sample_lf(referent=referent, n_parts=len(words) // 2)
        #why are we doing this?

        # Record unnormalized score p(u, z)
        weight = speaker_model.score(lf, utterance_bag, words)

        lfs.append(lf)
        g_lfs.append(g_lf)
        weights.append((weight, p))

    # Debug logging.
    data = sorted(zip(lfs, g_lfs, weights), key=lambda xs: xs[2], reverse=True)
    for lf, g_lf, weight in data:
        print("LF %30s  =>  Referent %10s  =>  Gen LF %30s  =>  %s" %
              (env.describe_lf(lf),
               env.resolve_lf(lf)[0]["attributes"][args.atom_attribute],
               env.describe_lf(g_lf),
               weight))

    rejs_per_sample = num_rejections / args.num_listener_samples
    print("%sRejections per sample: %.2f%s" % (colors.BOLD + colors.WARNING,
                                             rejs_per_sample, colors.ENDC))

    listener_model.reset()
    return lfs, weights, rejs_per_sample


def run_listener_trial(listener_model, speaker_model,
                       env, sess, args):
    """
    Run single recognition trial.

    1. Predict a referent give an utterance.
    2. Update listener model weights.
    3. Update speaker model weights.
    """
    env.configure(dreaming=False)
    obs = env.reset()

    rejs_per_sample = np.inf
    # TODO: magic number
    while rejs_per_sample > args.max_rejections_after_trial:
        lfs, lf_weights, rejs_per_sample = \
                infer_trial(env, obs, listener_model, speaker_model, args)
        lfs = sorted(zip(lfs, lf_weights), key=lambda el: el[1], reverse=True)

        # Now select action based on maximum generative score.
        lf_pred = lfs[0][0]
        _, reward, _, _ = env.step(lf_pred)

        success = reward > 0
        color = colors.OKGREEN if success else colors.FAIL
        print("%s%s => %s%s" % (colors.BOLD + color, env._trial["string_description"],
                                env.describe_lf(lf_pred), colors.ENDC))

        # Find the highest-scoring LF that dereferences to the correct referent.
        gold_lf, gold_lf_pos = None, -1
        for i, (lf_i, weight_i) in enumerate(lfs):
            resolved = env.resolve_lf(lf_i)
            if resolved and resolved[0]["target"]:
                gold_lf = lf_i
                gold_lf_pos = i
                break
        if gold_lf is not None:
            print("gold", env.describe_lf(gold_lf), gold_lf_pos)

        # Update listener parameters.
        listener_model.observe(obs, lf_pred, reward, gold_lf)

        # Update speaker parameters.
        if gold_lf is not None:
            speaker_model.observe(obs, gold_lf)
        listener_model.reset()

def run_dream_trial(listener_model, generative_model, env, sess, args):
    """
    Run a single dream trial.
    """
    env.configure(dreaming=True)
    items, _, _ = env.reset()

    referent_idx = [i for i, referent in enumerate(env._domain)
                    if referent["target"]][0]
    referent = env._domain[referent_idx]

    for run_i in range(2):
        success = False
        i = 0
        while not success:
            print("Dream trial %i" % i)

            # Sample an LF from p(z|r).
            g_lf = env.sample_lf(referent=referent_idx)

            # Sample utterances from p(u|z).
            words = generative_model.sample(g_lf).split()
            word_ids = np.array([env.word2idx[word]
                                    for word in words])

            g_ut = np.zeros(env.vocab_size)
            if len(word_ids):
                g_ut[word_ids] = 1

            # Run listener model q(z|u).
            l_lf, p = listener_model.sample(g_ut, words, temperature=0.5)
            # Literally dereference and see if we get the expected referent.
            # TODO: run multiple particles through this whole process!
            l_referent = env.resolve_lf(l_lf)
            if l_referent:
                success = l_referent[0] == referent

            print(
    """    Referent:\t\t%s
    z ~ p(z|r):\t\t%s
    u ~ p(u|z):\t\t%s
    z' ~ q(z|u):\t%s
    Deref:\t\t%s""" %
                (referent["attributes"][args.atom_attribute],
                env.describe_lf(g_lf),
                " ".join([env.vocab[idx] for idx, count in enumerate(g_ut) if count]),
                env.describe_lf(l_lf),
                [l_referent_i["attributes"][args.atom_attribute] for l_referent_i in l_referent]))

            i += 1
            if i > 1000:
                print("%sFailed to dream successfully after 1000 trials. Dying.%s"
                      % (colors.FAIL, colors.ENDC))
                break

        if success:
            # Construct an "observation" for the generative model.
            obs = (items, g_ut, words)
            generative_model.observe(obs, g_lf)


def sample_models(listener_model, speaker_model, env, args):
    listener_samples = []
    for num_trials in range(5):
        env.configure(dreaming=False)
        _, utterance_bag, words = env.reset()
        lf, p = listener_model.sample(utterance_bag, words)
        listener_samples.append((" ".join(words), env.describe_lf(lf)))

    speaker_samples = []
    for num_trials in range(5):
        env.reset()
        referent_idx = [i for i, referent in enumerate(env._domain)
                        if referent["target"]][0]
        lf = env.sample_lf(referent=referent_idx)
        words = speaker_model.sample(lf)
        speaker_samples.append((words, env.describe_lf(lf)))

    return listener_samples, speaker_samples


def eval_model(listener_model, examples, env):
    """
    Evaluate the listener model relative to ground-truth utterance-LF pairs.

    Args:
        listener_model:
        examples: Pairs of `(words_string, lf_candidates)`
    """
    results = []
    for words_string, lf_candidates in examples:
        words = words_string.strip().split()

        lf_candidate_tok_ids = []
        for lf_candidate in lf_candidates:
            tokens = re.split("[()]| AND ", lf_candidate.strip())
            ids = [env.lf_token_to_id[token] for token in tokens if token]
            lf_candidate_tok_ids.append(tuple(ids))

        # DEV: assumes we're using a non-BOW listener model
        sampled_lf, p = listener_model.sample(None, words, temperature=0.001, test=True)
        listener_model.reset()
        success = tuple(sampled_lf) in lf_candidate_tok_ids
        results.append((words_string, env.describe_lf(sampled_lf), success))

        color = colors.OKGREEN if success else colors.FAIL
        print("\t%s%30s => %30s%s" % (color, words_string, env.describe_lf(sampled_lf), colors.ENDC))

    return results


def build_train_graph(model, env, args, scope="train"):
    if args.learning_method == "rl":
        model.build_rl_gradients()
        gradients = model.rl_gradients
    elif args.learning_method == "xent":
        try:
            model.build_xent_gradients()
        except AttributeError:
            # Learning not defined.
            return None, None
        gradients = model.xent_gradients
    else:
        raise NotImplementedError("undefined learning method " + args.learning_method)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.MomentumOptimizer(args.learning_rate, args.momentum)
    train_op = opt.apply_gradients(gradients, global_step=global_step)

    # Make a dummy train_op that works with TF partial_run.
    with tf.control_dependencies([train_op]):
        train_op = tf.constant(0.0, name="dummy_train_op")

    return train_op, global_step


def train(args):
    env = TUNAWithLoTEnv(args.corpus_path, corpus_selection=args.corpus_selection,
                         bag=args.bag_env, functions=FUNCTIONS[args.fn_selection],
                         atom_attribute=args.atom_attribute)
    listener_model = SkipGramListenerModel(env)
    speaker_model = SPEAKER_MODELS[args.speaker_model](env, args.embedding_dim)

    listener_train_op, listener_global_step = \
            build_train_graph(listener_model, env, args, scope="train/listener")
    speaker_train_op, speaker_global_step = \
            build_train_graph(speaker_model, env, args, scope="train/speaker")
    # TODO does momentum / etc. of shared parameters get shared in this case?
    # check variables after build_train_graph
    listener_model.train_op = listener_train_op
    speaker_model.train_op = speaker_train_op
    accuracies = []
    with tf.Session() as sess:
        with sess.as_default():
            for run_i in range(args.num_runs):
                tqdm.write("%sBeginning training run %i.%s\n\n" % (colors.BOLD, run_i, colors.ENDC))
                sess.run(tf.initialize_all_variables())

                for i in trange(args.num_trials):
                    tqdm.write("\n%s==============\nLISTENER TRIAL\n==============%s"
                            % (colors.HEADER, colors.ENDC))
                    run_listener_trial(listener_model, speaker_model,
                                       env, sess, args)

                    if args.dream:
                        tqdm.write("\n%s===========\nDREAM TRIAL\n===========%s"
                                % (colors.HEADER, colors.ENDC))
                        run_dream_trial(listener_model, speaker_model,
                                        env, sess, args)

                # Print samples from listener, speaker model
                print("\n%s=======\nSAMPLES\n=======%s"
                      % (colors.HEADER, colors.ENDC))
                listener_samples, speaker_samples = \
                        sample_models(listener_model, speaker_model, env, args)
                print("%sListener model (u -> z)%s" % (colors.BOLD, colors.ENDC))
                for words, lf in listener_samples:
                    print("\t%30s => %30s" % (words, lf))
                print("%sSpeaker model (z -> u)%s" % (colors.BOLD, colors.ENDC))
                for words, lf in speaker_samples:
                    print("\t%30s => %30s" % (lf, words))

                if args.gold_path:
                    with open(args.gold_path, "r") as gold_f:
                        listener_examples = json.load(gold_f)

                        print("\n%s==========\nEVALUATION\n==========%s"
                              % (colors.HEADER, colors.ENDC))
                        eval_results = eval_model(listener_model, listener_examples, env)
                        n_success = len([result for _, _, result in eval_results if result])
                        accuracy = n_success / len(eval_results)
                        accuracies.append(accuracy)
                        print("%sAccuracy: %.3f%%%s" % (colors.BOLD, accuracy * 100, colors.ENDC))
    if args.gold_path:
        print("\n%s==========\nOVERALL EVALUATION\n==========%s"
              % (colors.HEADER, colors.ENDC))
        avg_accuracy = np.mean(accuracies)
        print("%sAverage accuracy: %.3f%%%s" % (colors.BOLD, avg_accuracy * 100, colors.ENDC))


SPEAKER_MODELS = {
    "naive": lambda env, _: NaiveGenerativeModel(env),
    "discrete": lambda env, _: DiscreteGenerativeModel(env),
    "window":  lambda env, emb_dim: WindowedSequenceSpeakerModel(env, embedding_dim=emb_dim)
}


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/tuna")
    p.add_argument("--analyze_weights", default=False, action="store_true")

    p.add_argument("--corpus_path", required=True)
    p.add_argument("--corpus_selection", default=None)
    p.add_argument("--fn_selection", default="spatial_simple",
                   choices=FUNCTIONS.keys())
    p.add_argument("--atom_attribute", default="shape")
    p.add_argument("--gold_path",
                   help=("Path to JSON file containing gold listener "
                         "utterance -> LF interpretations"))

    p.add_argument("--speaker_model", default="discrete",
                   choices=SPEAKER_MODELS.keys())
    p.add_argument("--bag_env", default=False, action="store_true")
    p.add_argument("--item_repr_dim", type=int, default=64)
    p.add_argument("--utterance_repr_dim", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=10)

    p.add_argument("--dream", default=False, action="store_true")
    p.add_argument("--num_listener_samples", type=int, default=5)
    p.add_argument("--max_rejections_after_trial", type=int, default=3)

    p.add_argument("--num_runs", default=1, type=int,
                   help="Number of times to repeat entire training process")
    p.add_argument("--learning_method", default="rl", choices=["rl", "xent"])
    p.add_argument("--num_trials", default=100, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)
    p.add_argument("--momentum", default=0.9, type=float)

    train(p.parse_args())
