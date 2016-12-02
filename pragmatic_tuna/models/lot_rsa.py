from argparse import ArgumentParser
from collections import Counter, defaultdict, namedtuple
from itertools import permutations

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import trange, tqdm
import nltk

from pragmatic_tuna.environments import TUNAWithLoTEnv
from pragmatic_tuna.environments.spatial import *
from pragmatic_tuna.reinforce import reinforce_episodic_gradients


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

    def __init__(self, vocab_size, max_conjuncts, env, smooth=True):
        self.smooth = smooth
        self.counter = defaultdict(lambda: Counter())
        self.bigramcounter = defaultdict(lambda: Counter())
        self.unigramcounter = Counter()
        self.vocab_size = vocab_size
        self.max_conjuncts = max_conjuncts
        #TODO: better way to pass the env to the model?
        self.env = env

    def observe(self, obs, gold_lf):
        u = obs[2]
        z = gold_lf

        for lf_token in z:
            self.counter[lf_token].update(u)

        words = []
        words.extend(u)
        words.append(self.END_TOKEN)

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
        words.append(self.END_TOKEN)
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
        if len(u) > len(z):
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

        return p_seq + p_trans


    def sample_with_alignment(self, z, alignment):
        unigram_denom = max(sum(self.unigramcounter.values()), 1.0)
        unigram_probs = np.array(list(self.unigramcounter.values())) * self.backoff_factor / unigram_denom
        keys = list(self.unigramcounter.keys())

        prev_word = self.START_TOKEN

        u = []
        ps = []
        i = 0
        while prev_word != self.END_TOKEN:
            #limit utterance length to the length of the lf
            if i == len(z):
               word = self.END_TOKEN
               u.append(word)
               prev_word = word
               #todo compute correct probability
               ps.append(1.0)
               break

            bigram_counts = np.array([self.bigramcounter[prev_word][w]
                                        for w in keys])
            bigram_denom = sum(bigram_counts) if len(bigram_counts) > 0 else 1.0
            bigram_probs = bigram_counts * (1 - self.backoff_factor) / bigram_denom


            cond_probs = np.array([self._score_word_atom(w, z[alignment[i]]) for w in keys])

            distr = (bigram_probs + unigram_probs) * cond_probs

            distr = distr / np.sum(distr)

            idx = np.random.choice(len(keys), p=distr)
            word = keys[idx]
            if len(u) < 1 and word == self.END_TOKEN:
                continue
            u.append(word)
            prev_word = word
            ps.append(distr[idx])
            i += 1

        p = np.exp(np.sum(np.log(distr)))
        return " ".join(u[0:-1]), p

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

        self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def build_rl_gradients(self):
        raise NotImplementedError

    def build_xent_gradients(self):
        raise NotImplementedError

    def sample(self, utterance_bag, words):
        raise NotImplementedError

    def observe(self, obs, lf_pred, reward, gold_lf, train_op):
        raise NotImplementedError


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

    def observe(self, obs, lf_pred, reward, gold_lf, train_op):
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
        sess.run(train_op, train_feeds)


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
            # TODO: padding representation?
            self.words = tf.placeholder(tf.int32, shape=(self.max_timesteps,),
                                        name="words")

            emb_shape = (self.env.vocab_size, self.embedding_dim)
            word_embeddings = tf.get_variable("word_embeddings", shape=emb_shape)

            word_window = tf.nn.embedding_lookup(word_embeddings, self.words)
            word_window = tf.reshape(word_window, (-1,))

            # Create embeddings for LF tokens + null/stop token (id 0)
            lf_emb_shape = (len(self.env.lf_vocab) + 1, self.embedding_dim)
            lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape)
            null_embedding = tf.gather(lf_embeddings, 0)

            # Now run a teeny LF decoder.
            outputs, samples = [], []
            output_dim = lf_emb_shape[0]
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(0, [prev_sample, word_window])
                    output_t = layers.fully_connected(tf.expand_dims(input_t, 0),
                                                      output_dim, tf.identity)

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1))
                    prev_sample = tf.nn.embedding_lookup(lf_embeddings, sample_t)

                    # TODO support stop token here?

                    outputs.append(output_t)
                    samples.append(sample_t)

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
        word_idxs += [0] * (self.max_timesteps - len(word_idxs))
        return word_idxs

    def sample(self, utterance_bag, words):
        sess = tf.get_default_session()
        feed = {self.words: self._get_word_idxs(words)}
        # Rejection-sample a valid LF (alternating fn-atom-fn-atom..)
        while True:
            sample = sess.run(self.samples, feed)
            ret_sample = []
            valid = True
            for i, sample_i in enumerate(sample):
                stop = sample_i == 0
                if stop and i > 0 and i % 2 == 0:
                    # Valid stopping point. Truncate and end.
                    ret_sample = ret_sample[:i]
                    break

                lf_voc_idx = sample_i - 1
                ret_sample.append(lf_voc_idx)
                sample_i_str = self.env.lf_vocab[lf_voc_idx]
                if i % 2 == 0 and sample_i_str not in self.env.lf_function_map:
                    valid = False
                elif i % 2 == 1 and sample_i_str not in self.env.lf_atoms:
                    valid = False

            if valid:
                return ret_sample

    def observe(self, obs, lf_pred, reward, gold_lf, train_op):
        if gold_lf is None:
            return

        # Convert LF to internal space (shift IDs; add STOP token)
        gold_lf = [idx + 1 for idx in gold_lf]
        real_length = min(self.max_timesteps, len(gold_lf) + 1) # train to output a single stop token
        if len(gold_lf) < self.max_timesteps:
            gold_lf.extend([0] * (self.max_timesteps - len(gold_lf)))

        feed = {self.words: self._get_word_idxs(obs[2]),
                self.xent_gold_lf_length: real_length}
        feed.update({self.xent_gold_lf_tokens[t]: gold_lf[t]
                     for t in range(self.max_timesteps)})

        sess = tf.get_default_session()
        sess.run(train_op, feed)


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
    """

    items, utterance_bag, words = obs

    # Sample N LFs from the predicted distribution, accepting only when they
    # resolve to a referent in the scene.
    lfs, g_lfs, weights = [], [], []
    while len(weights) < args.num_listener_samples:
        lf = listener_model.sample(utterance_bag, words)

        # Resolve referent.
        referent = env.resolve_lf(lf)
        if not referent:
            # Dereference failed. No object matched.
            continue
        referent = env._domain.index(referent[0])

        # Sample an LF z' ~ p(z|r).
        g_lf = env.sample_lf(referent=referent)

        # Record unnormalized score p(u, z)
        weight = speaker_model.score(g_lf, utterance_bag, words)

        lfs.append(lf)
        g_lfs.append(g_lf)
        weights.append(weight)

    # Debug logging.
    data = sorted(zip(lfs, g_lfs, weights), key=lambda xs: xs[2], reverse=True)
    for lf, g_lf, weight in data:
        print("LF %30s  =>  Referent %10s  =>  Gen LF %30s  =>  %f" %
              (env.describe_lf(lf),
               env.resolve_lf(lf)[0]["attributes"][args.atom_attribute],
               env.describe_lf(g_lf),
               weight))

    return lfs, weights


def run_listener_trial(listener_model, speaker_model, listener_train_op,
                       env, sess, args):
    """
    Run single recognition trial.

    1. Predict a referent give an utterance.
    2. Update listener model weights.
    3. Update speaker model weights.
    """
    env.configure(dreaming=False)
    obs = env.reset()

    lfs, lf_weights = infer_trial(env, obs, listener_model, speaker_model, args)
    lfs = sorted(zip(lfs, lf_weights), key=lambda el: el[1], reverse=True)

    # Now select action based on maximum generative score.
    lf_pred = lfs[0][0]
    _, reward, _, _ = env.step(lf_pred)
    tqdm.write("%f" % reward)

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
    listener_model.observe(obs, lf_pred, reward, gold_lf, listener_train_op)

    # Update speaker parameters.
    if gold_lf is not None:
        speaker_model.observe(obs, gold_lf)


def run_dream_trial(listener_model, generative_model, env, sess, args):
    """
    Run a single dream trial.
    """
    env.configure(dreaming=True)
    items, _, _ = env.reset()

    referent_idx = [i for i, referent in enumerate(env._domain)
                    if referent["target"]][0]
    referent = env._domain[referent_idx]

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
        g_ut[word_ids] = 1

        # Run listener model q(z|u).
        l_lf = listener_model.sample(g_ut, words)
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
              (env._domain[referent_idx],
               env.describe_lf(g_lf),
               " ".join([env.vocab[idx] for idx, count in enumerate(g_ut) if count]),
               env.describe_lf(l_lf),
               l_referent))

        i += 1

    # Construct an "observation" for the generative model.
    obs = (items, g_ut, words)
    generative_model.observe(obs, g_lf)


def build_train_graph(model, env, args):
    if args.learning_method == "rl":
        model.build_rl_gradients()
        gradients = model.rl_gradients
    elif args.learning_method == "xent":
        model.build_xent_gradients()
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
    model = WindowedSequenceListenerModel(env)
    train_op, global_step = build_train_graph(model, env, args)
    generative_model = DiscreteGenerativeModel(env.vocab_size, 3, env) # TODO fixed

    supervisor = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                                     summary_op=None)
    with supervisor.managed_session() as sess:
        with sess.as_default():
            for i in trange(args.num_trials):
                if supervisor.should_stop():
                    break

                tqdm.write("\n===========\nLISTENER TRIAL\n===========")
                run_listener_trial(model, generative_model, train_op, env, sess, args)

                if args.dream:
                    tqdm.write("\n===========\nDREAM TRIAL\n============")
                    run_dream_trial(model, generative_model, env, sess, args)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/tuna")
    p.add_argument("--analyze_weights", default=False, action="store_true")

    p.add_argument("--corpus_path", required=True)
    p.add_argument("--corpus_selection", default=None)
    p.add_argument("--fn_selection", default="spatial_simple",
                   choices=FUNCTIONS.keys())
    p.add_argument("--atom_attribute", default="shape")

    p.add_argument("--bag_env", default=False, action="store_true")
    p.add_argument("--item_repr_dim", type=int, default=64)
    p.add_argument("--utterance_repr_dim", type=int, default=64)

    p.add_argument("--dream", default=False, action="store_true")
    p.add_argument("--num_listener_samples", type=int, default=5)

    p.add_argument("--learning_method", default="rl", choices=["rl", "xent"])
    p.add_argument("--num_trials", default=100, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)
    p.add_argument("--momentum", default=0.9, type=float)

    train(p.parse_args())
