"""
Defines generative speaker models.
"""

from collections import Counter, defaultdict
from itertools import permutations

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers

from pragmatic_tuna.util import orthogonal_initializer


class SpeakerModel(object):

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

    def sample(self, lf):
        raise NotImplementedError

    def score(self, lf, words):
        raise NotImplementedError

    def observe(self, env_obs, gold_lf):
        raise NotImplementedError

    def batch_observe(self, words, gold_lfs):
        raise NotImplementedError


class EnsembledSpeakerModel(SpeakerModel):

    def __init__(self, models):
        self.models = models

    def build_xent_gradients(self):
        gradients = []
        for model in self.models:
            model.build_xent_gradients()
            gradients.extend(model.xent_gradients)

        self.xent_gradients = gradients

    def sample(self, lf):
        model = self.models[np.random.choice(len(self.models))]
        return model.sample(lf)

    def score(self, lf, words):
        scores = [model.score(lf, words) for model in self.models]
        return np.log(np.mean(np.exp(scores)))

    def observe(self, env_obs, gold_lf):
        for model in self.models:
            model.observe(env_obs, gold_lf)


class SequenceSpeakerModel(SpeakerModel):

    def build_xent_gradients(self):
        gold_words = [tf.placeholder(tf.int32, shape=(None,),
                                     name="gold_word_%i" % t)
                      for t in range(self.max_timesteps)]
        gold_length = tf.placeholder(tf.int32, shape=(None,),
                                     name="gold_lengths")

        loss_weights = [tf.to_float(t < gold_length)
                        for t in range(self.max_timesteps)]
        loss = tf.nn.seq2seq.sequence_loss(self.outputs, gold_words,
                                           loss_weights)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_words = gold_words
        self.xent_gold_length = gold_length
        self.xent_gradients = zip(gradients, params)

        return ((self.xent_gold_words, self.xent_gold_length),
                (self.xent_gradients,))

    def sample(self, z):
        sess = tf.get_default_session()
        feed = {self.lf_toks: [self.env.pad_lf_idxs(z)]}

        samples = sess.run(self.samples, feed)
        sample = [x[0] for x in samples]
        try:
            stop_idx = sample.index(self.env.word_eos_id)
            sample = sample[:stop_idx]
        except ValueError:
            # No stop token. No trimming necessary. Pass.
            pass

        return " ".join(self.env.vocab[idx] for idx in sample)

    def score(self, z, u):
        sess = tf.get_default_session()

        z = self.env.pad_lf_idxs(z)
        words = [self.env.word2idx[word] for word in u]

        feed = {self.lf_toks: [z]}
        feed.update({self.samples[t]: [word] for t, word in enumerate(words)})

        probs = sess.run(self.probs, feed)
        probs = [probs_t[0, word_t] for probs_t, word_t in zip(probs, words)]
        return np.log(np.prod(probs))

    def observe(self, obs, gold_lf):
        if gold_lf is None:
            return

        return self.batch_observe([obs[1]], [gold_lf])

    def batch_observe(self, words_lists, gold_lfs):
        n = len(gold_lfs)
        z = np.zeros(shape=(n, self.max_timesteps), dtype=np.int32)
        real_lengths = np.zeros(shape=n, dtype=np.int32)
        for i, gold_lf in enumerate(gold_lfs):
            padded_lf = self.env.pad_lf_idxs(gold_lf)
            for j in range(self.max_timesteps):
                z[i][j] = padded_lf[j]

        gold_words = [np.zeros(shape=(n,), dtype=np.int32) for _ in range(self.max_timesteps)]
        for i, words in enumerate(words_lists):
            real_lengths[i] = min(len(words) + 1, self.max_timesteps)
            words = self.env.get_word_idxs(words)
            for j in range(len(words)):
                gold_words[j][i] = words[j]

        sess = tf.get_default_session()
        feed = {self.lf_toks: z, self.xent_gold_length: real_lengths}
        feed.update({self.xent_gold_words[t]: word_t
                    for t, word_t in enumerate(gold_words)})
        feed.update({self.samples[t]: word_t
                     for t, word_t in enumerate(gold_words)})
        sess.run(self.train_op, feed)


class ShallowSequenceSpeakerModel(SequenceSpeakerModel):

    """
    A shallow sequence speaker model which is a simple CLM (no word embeddings)
    conditioned on some LF embeddings.
    """

    def __init__(self, env, scope="speaker", max_timesteps=2,
                 lf_embeddings=None, embedding_dim=10):
        self.env = env
        self._scope_name = scope
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

        self.train_op = None

        self._build_embeddings(lf_embeddings)
        self._build_graph()

    def _build_embeddings(self, lf_embeddings):
        with tf.variable_scope(self._scope_name):
            lf_emb_shape = (len(self.env.lf_vocab), self.embedding_dim)
            if lf_embeddings is None:
                lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape,
                                                initializer=orthogonal_initializer())
            assert tuple(lf_embeddings.get_shape().as_list()) == lf_emb_shape

            self.lf_embeddings = lf_embeddings

    def _build_graph(self):
        with tf.variable_scope(self._scope_name):
            self.lf_toks = tf.placeholder(tf.int32, shape=(None, self.max_timesteps),
                                          name="lf_toks")

            lf_window = tf.nn.embedding_lookup(self.lf_embeddings, self.lf_toks)
            lf_window = tf.reduce_mean(lf_window, 1)
            lf_window = tf.stop_gradient(lf_window)

            batch_size = tf.shape(self.lf_toks)[0]

            # CLM à la Andreas.

            # TODO: ugly trick to get the right dimension. is there a better way?
            prev_words = tf.zeros((batch_size, len(self.env.vocab)))
            last_word = tf.zeros((batch_size, len(self.env.vocab)))

            # Now run a teeny utterance decoder.
            outputs, probs, samples = [], [], []
            output_dim = self.env.vocab_size
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(1, [prev_words, last_word, lf_window])
                    output_t = layers.fully_connected(input_t,
                                                      output_dim, tf.identity)

                    probs_t = tf.nn.softmax(output_t)

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1), [1],
                                          name="sample_%i" % t)

                    last_word = tf.one_hot(sample_t, len(self.env.vocab))
                    prev_words += last_word

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.probs = probs
            self.samples = samples


class WindowedSequenceSpeakerModel(SequenceSpeakerModel):

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
            self.lf_toks = tf.placeholder(tf.int32, shape=(None, self.max_timesteps),
                                          name="lf_toks")

            batch_size = tf.shape(self.lf_toks)[0]
            lf_window_dim = self.max_timesteps * self.embedding_dim
            lf_window = tf.nn.embedding_lookup(self.lf_embeddings, self.lf_toks)
            lf_window = tf.reshape(lf_window, (batch_size, lf_window_dim))

            null_embedding = tf.gather(self.word_embeddings,
                                       tf.fill((batch_size,), self.env.word_unk_id))

            # Now run a teeny utterance decoder.
            outputs, probs, samples = [], [], []
            output_dim = self.env.vocab_size
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(1, [prev_sample, lf_window])
                    output_t = layers.fully_connected(input_t,
                                                      output_dim, tf.identity)
                    probs_t = tf.nn.softmax(output_t)

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1), [1],
                                          name="sample_%i" % t)
                    prev_sample = tf.nn.embedding_lookup(self.word_embeddings, sample_t)

                    # TODO support stop token here?

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.probs = probs
            self.samples = samples


class EnsembledSequenceSpeakerModel(EnsembledSpeakerModel):

    def __init__(self, env, n, cls=ShallowSequenceSpeakerModel,
                 scope="speaker", lf_embeddings=None, **kwargs):
        self.env = env

        # Build the first model.
        models = [cls(env, scope="%s_0" % scope, lf_embeddings=lf_embeddings,
                      **kwargs)]
        models.extend([cls(env, scope="%s_%i" % (scope, i),
                           #word_embeddings=models[0].word_embeddings,
                           lf_embeddings=models[0].lf_embeddings, **kwargs)
                        for i in range(1, n)])

        super(EnsembledSequenceSpeakerModel, self).__init__(models)

    def observe(self, env_obs, gold_lf):
        if gold_lf is None:
            return

        z = self.env.pad_lf_idxs(gold_lf)

        max_timesteps = self.models[0].max_timesteps
        words = [self.env.word2idx[word] for word in env_obs[1]]
        real_length = min(len(words) + 1, max_timesteps)
        # Add a EOS token to words
        if len(words) > max_timesteps:
            words.append(self.env.word_eos_id)

        sess = tf.get_default_session()
        feed = {}
        for model in self.models:
            feed[model.lf_toks] = z
            feed[model.xent_gold_length] = real_length

            feed.update({model.xent_gold_words[t]: word_t
                         for t, word_t in enumerate(words)})
            feed.update({model.samples[t]: word_t
                         for t, word_t in enumerate(words)})

        sess.run(self.train_op, feed)


class EnsembledSequenceSpeakerModel2(SequenceSpeakerModel):

    """
    Sequence speaker model which ensembles several distinct recurrent matrices.
    """

    def __init__(self, env, n=8, scope="speaker", max_timesteps=2,
                 lf_embeddings=None, embedding_dim=10):
        self.env = env
        self.n = n
        self._scope_name = scope
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

        self.train_op = None

        self._build_embeddings(lf_embeddings)
        self._build_graph()

    def _build_embeddings(self, lf_embeddings):
        with tf.variable_scope(self._scope_name):
            lf_emb_shape = (len(self.env.lf_vocab), self.embedding_dim)
            if lf_embeddings is None:
                lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape,
                                                initializer=orthogonal_initializer())
            assert tuple(lf_embeddings.get_shape().as_list()) == lf_emb_shape

            self.lf_embeddings = lf_embeddings

    def _build_graph(self):
        with tf.variable_scope(self._scope_name):
            self.lf_toks = tf.placeholder(tf.int32, shape=(self.max_timesteps,),
                                          name="lf_toks")
            lf_window = tf.nn.embedding_lookup(self.lf_embeddings, self.lf_toks)
            lf_window = tf.reduce_mean(lf_window, 0)
            lf_window = tf.stop_gradient(lf_window)

            # CLM à la Andreas.
            prev_words = tf.zeros((len(self.env.vocab),))
            last_word = tf.zeros((len(self.env.vocab),))

            # Now run a teeny utterance decoder.
            outputs, probs, samples = [], [], []
            output_dim = self.env.vocab_size
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(0, [prev_words, last_word, lf_window])
                    input_t = tf.expand_dims(input_t, 0)

                    outputs_t = [layers.fully_connected(input_t, output_dim, tf.identity,
                                                        scope="recurrence_%i" % i)
                                 for i in range(self.n)]
                    output_t = tf.add_n(outputs_t) / float(self.n)
                    probs_t = tf.squeeze(tf.nn.softmax(output_t))

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1))

                    last_word = tf.one_hot(sample_t, len(self.env.vocab))
                    prev_words += last_word

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.probs = probs
            self.samples = samples

