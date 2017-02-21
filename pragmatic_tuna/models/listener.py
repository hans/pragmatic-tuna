"""
Defines discriminative listener models.
"""

import copy
import itertools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers

from pragmatic_tuna.reinforce import reinforce_episodic_gradients
from pragmatic_tuna.util import orthogonal_initializer


EMBEDDING_INITIALIZER = orthogonal_initializer()
LF_EMBEDDING_INITIALIZER = orthogonal_initializer()


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

    def sample(self, words, temperature=None, argmax=False, evaluating=False):
        """
        Conditionally sample an LF given an input utterance.

        Returns:
            lf: LF token idx sequence
            p_lf: Probability of sampled LF under model
        """
        lfs, p_lfs = self.sample_batch([words])
        return lfs[0], p_lfs[0]

    def sample_batch(self, words, temperature=None, argmax=False, evaluating=False):
        """
        Conditionally sample a batch of LFs given a batch of input utterances.

        Args:
            words: List of token sequences (strings)
        Returns:
            lfs:
            p_lfs:
        """
        raise NotImplementedError

    def observe(self, obs, lf_pred, reward, gold_lf):
        raise NotImplementedError

    def score_batch(self, words, lfs):
        """
        Evaluate p(z|u) for a batch of word sequence inputs and sequence LF
        outputs.
        """
        raise NotImplementedError

    def marginalize(self, lf):
        """
        Compute the marginal p(z) by marginalizing out utterances p(z|u).
        """
        # For now, we'll just evaluate the listener model for every possible
        # utterance. Because, well, there aren't that many.
        valid_tokens = copy.copy(self.env.vocab)
        valid_tokens.remove(self.env.vocab[self.env.word_unk_id])
        valid_tokens.remove(self.env.vocab[self.env.word_eos_id])
        permutations = list(itertools.chain.from_iterable(
                itertools.permutations(valid_tokens, t)
                for t in range(1, self.max_timesteps + 1)))

        # Evaluate scores for the token permutations calculated above.
        lfs = [lf] * len(permutations)
        scores = self.score_batch(permutations, lfs)

        # TODO: This sometimes returns values > 1. Why?
        # This partition function actually doesn't even cover the entire space,
        # as we don't enumerate invalid options like "<eos> square"
        return sum(scores)

    def reset(self):
        pass


class EnsembledListenerModel(ListenerModel):

    def __init__(self, models):
        self.models = models

    def build_xent_gradients(self):
        gradients = []
        for model in self.models:
            model.build_xent_gradients()
            gradients.extend(model.xent_gradients)

        self.xent_gradients = gradients

    def sample(self, words, **kwargs):
        # TODO if argmax, should sample from all models
        model = self.models[np.random.choice(len(self.models))]
        return model.sample(words, **kwargs)

    def observe(self, obs, lf_pred, reward, gold_lf):
        raise NotImplementedError

    def reset(self):
        for model in self.models:
            model.reset()


class WindowedSequenceListenerModel(ListenerModel):

    """
    Parametric listener model $q_\\theta(z|u) which maps utterances (sequences)
    to LF representations (factored as sequences).

    This model takes a window of embedding inputs and outputs a sequence of LF
    tokens.
    """

    def __init__(self, env, scope="listener", max_timesteps=2, embedding_dim=10):
        self.env = env
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

        self._build_embeddings()
        super(WindowedSequenceListenerModel, self).__init__(env, scope=scope)

    def _build_embeddings(self):
        with tf.variable_scope("embeddings", initializer=EMBEDDING_INITIALIZER):
            # Word embeddings
            emb_shape = (self.env.vocab_size, self.embedding_dim)
            word_embeddings = tf.get_variable("word_embeddings", shape=emb_shape)
            self.word_embeddings = word_embeddings

            lf_emb_shape = (len(self.env.lf_vocab), self.embedding_dim)
            lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape)
            self.lf_embeddings = lf_embeddings

    def _build_graph(self):
        with self._scope:
            self.temperature = tf.constant(1.0, name="sampling_temperature")

            self.words = tf.placeholder(tf.int32, shape=(None, self.max_timesteps,),
                                        name="words")

            word_window = tf.nn.embedding_lookup(self.word_embeddings, self.words)
            word_window = tf.reshape(word_window, (-1, self.embedding_dim * self.max_timesteps))

            # Create embeddings for LF tokens
            batch_size = tf.shape(self.words)[0]
            null_embedding = tf.tile(
                    tf.expand_dims(tf.gather(self.lf_embeddings, self.env.lf_unk_id), 0),
                    (batch_size, 1))

            # Weight matrices mapping input -> ~p(fn), input -> ~p(atom)
            input_dim = self.embedding_dim + self.embedding_dim * self.max_timesteps
            W_fn = tf.get_variable("W_fn", shape=(input_dim, len(self.env.lf_functions)))
            W_atom = tf.get_variable("W_atom", shape=(input_dim, len(self.env.lf_atoms)))

            # Now run a teeny LF decoder.
            outputs, probs, samples = [], [], []
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(1, [prev_sample, word_window])

                    # Force-sample a syntactically valid LF.
                    # i.e. alternating fn,atom,fn,atom,...
                    #
                    # TODO: This is coupled with the ordering of the LF tokens
                    # in the env definition. That could be bad.
                    if t % 2 == 0:
                        fn_logits = tf.matmul(input_t, W_fn) / self.temperature
                        atom_logits = tf.zeros((batch_size, len(self.env.lf_atoms)))
                        sample_t = tf.multinomial(fn_logits, num_samples=1)
                        fn_probs = tf.nn.softmax(fn_logits)
                        atom_probs = atom_logits
                    else:
                        fn_logits = tf.zeros((batch_size, len(self.env.lf_functions)))
                        atom_logits = tf.matmul(input_t, W_atom) / self.temperature
                        sample_t = tf.multinomial(atom_logits, num_samples=1)
                        fn_probs = fn_logits
                        atom_probs = tf.nn.softmax(atom_logits)

                    output_t = tf.concat(1, (fn_logits, atom_logits),
                                         name="output_%i" % t)
                    probs_t = tf.concat(1, (fn_probs, atom_probs),
                                        name="probs_%i" % t)

                    sample_t = tf.squeeze(sample_t, [1])
                    if t % 2 == 1:
                        # Shift index to match standard vocabulary.
                        sample_t = tf.add(sample_t, len(self.env.lf_functions))
                    # Rename
                    sample_t = tf.identity(sample_t, name="sample_%i" % t)

                    # Hack shape.
                    prev_sample = tf.nn.embedding_lookup(self.lf_embeddings, sample_t)

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.probs = probs
            self.samples = samples

    def build_xent_gradients(self):
        gold_lf_tokens = [tf.placeholder(tf.int32, shape=(None,),
                                         name="gold_lf_%i" % t)
                          for t in range(self.max_timesteps)]
        gold_lf_length = tf.placeholder(tf.int32, shape=(None,),
                                        name="gold_lf_length")

        # `max_timesteps`-length list of 1D `batch_size` masks
        loss_weights = [tf.to_float(t < gold_lf_length)
                        for t in range(self.max_timesteps)]
        loss = tf.nn.seq2seq.sequence_loss(self.outputs, gold_lf_tokens,
                                           loss_weights)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_lf_tokens = gold_lf_tokens
        self.xent_gold_lf_length = gold_lf_length
        self.xent_gradients = zip(gradients, params)

        self.feeds.extend([self.xent_gold_lf_tokens, self.xent_gold_lf_length])

        return ((self.xent_gold_lf_tokens, self.xent_gold_lf_length),
                (self.xent_gradients,))

    def sample(self, words, temperature=1.0, argmax=False, evaluating=False):
        ret_lfs, total_probs = self.sample_batch([words])
        return ret_lfs[0], total_probs[0]

    def sample_batch(self, words, temperature=1.0, argmax=False, evaluating=False):
        # TODO handle argmax, evaluating
        batch_size = len(words)

        sess = tf.get_default_session()
        feed = {self.words: [self.env.get_word_idxs(words_i)
                             for words_i in words],
                self.temperature: temperature}

        rets = sess.run(self.samples + self.probs, feed)

        # Unpack.
        sample_toks = rets[:len(self.samples)]
        probs = rets[len(self.samples):]

        # Calculate sequence probability as batch.
        done = np.zeros(batch_size)
        total_probs = np.ones(batch_size)
        batch_range = np.arange(batch_size)
        ret_lfs = [[] for _ in range(batch_size)]
        for t, (samples_t, probs_t) in enumerate(zip(sample_toks, probs)):
            total_probs *= probs_t[batch_range, samples_t]
            done = np.logical_or(done, samples_t == self.env.lf_eos_id)
            for i, sample_t_i in enumerate(samples_t):
                if not done[i]:
                    ret_lfs[i].append(sample_t_i)

        return ret_lfs, total_probs

    def observe(self, obs, lf_pred, reward, gold_lf):
        if gold_lf is None:
            return

        # Pad gold output LF.
        real_length = min(self.max_timesteps, len(gold_lf) + 1) # train to output a single stop token
        gold_lf = self.env.pad_lf_idxs(gold_lf)

        word_idxs = self.env.get_word_idxs(obs[1])
        feed = {self.words: [word_idxs],
                self.xent_gold_lf_length: [real_length]}
        feed.update({self.xent_gold_lf_tokens[t]: [gold_lf[t]]
                     for t in range(self.max_timesteps)})
        feed.update({self.samples[t]: [lf_t]
                     for t, lf_t in enumerate(gold_lf)})

        sess = tf.get_default_session()
        sess.run(self.train_op, feed)

    def score_batch(self, words, lfs):
        words = [self.env.get_word_idxs(words_i) for words_i in words]

        lfs = [self.env.pad_lf_idxs(lf_i) for lf_i in lfs]
        # Transpose to num_timesteps * batch_size
        lfs = np.array(lfs).T

        feed = {self.words: np.array(words)}
        feed.update({self.samples[t]: lfs_t
                     for t, lfs_t in enumerate(lfs)})

        sess = tf.get_default_session()
        probs = sess.run(self.probs, feed)

        # Index the sampled word for each example at each timestep
        batch_size = len(words)
        probs = [probs_t[np.arange(batch_size), lf_sample_t]
                 for probs_t, lf_sample_t in zip(probs, lfs)]

        return np.prod(probs, axis=0)
