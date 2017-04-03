"""
Defines speaker model for VG.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from pragmatic_tuna.util import orthogonal_initializer


EMBEDDING_INITIALIZER = orthogonal_initializer()
GRAPH_EMBEDDING_INITIALIZER = orthogonal_initializer()


class SpeakerModel(object):

    def __init__(self, env, scope="listener"):
        self.env = env
        self._scope = tf.variable_scope(scope)
        self.train_op = None

        self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def build_rl_gradients(self):
        raise NotImplementedError

    def build_xent_gradients(self):
        raise NotImplementedError

    def sample(self, subgraph, argmax=False):
        raise NotImplementedError

    def score(self, utterance, subgraph):
        raise NotImplementedError

    def score_batch(self, utterances_batch, subgraphs):
        raise NotImplementedError

    def observe(self, utterance, pos_candidates, neg_candidates):
        raise NotImplementedError

    def observe_batch(self, utterances_batch, pos_cands_batch, neg_cands_batch):
        raise NotImplementedError


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

    def sample(self, z, argmax=False):
        sess = tf.get_default_session()
        feed = {self.graph_toks: [self.env.pad_graph_idxs(z)],
                self.temperature: 0.0001 if argmax else 1.0}

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

        z = self.env.pad_graph_idxs(z)
        words = [self.env.word2idx[word] for word in u]

        feed = {self.graph_toks: [z]}
        feed.update({self.samples[t]: [word] for t, word in enumerate(words)})

        probs = sess.run(self.probs, feed)
        probs = [probs_t[0, word_t] for probs_t, word_t in zip(probs, words)]
        return np.log(np.prod(probs))

    def observe(self, obs, gold_lf):
        if gold_lf is None:
            return

        return self.observe_batch([obs[1]], [gold_lf])

    def observe_batch(self, words_lists, gold_lfs):
        n = len(gold_lfs)
        z = np.zeros(shape=(n, self.max_timesteps), dtype=np.int32)
        real_lengths = np.zeros(shape=n, dtype=np.int32)
        for i, gold_lf in enumerate(gold_lfs):
            padded_lf = self.env.pad_graph_idxs(gold_lf)
            for j in range(self.max_timesteps):
                z[i][j] = padded_lf[j]

        gold_words = [np.zeros(shape=(n,), dtype=np.int32) for _ in range(self.max_timesteps)]
        for i, words in enumerate(words_lists):
            real_lengths[i] = min(len(words) + 1, self.max_timesteps)
            words = self.env.get_word_idxs(words)
            for j in range(len(words)):
                gold_words[j][i] = words[j]

        sess = tf.get_default_session()
        feed = {self.graph_toks: z, self.xent_gold_length: real_lengths}
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
                 graph_embeddings=None, embedding_dim=10):
        self.env = env
        self._scope_name = scope
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

        self.train_op = None

        self._build_embeddings(graph_embeddings)
        self._build_graph()

    def _build_embeddings(self, graph_embeddings):
        with tf.variable_scope(self._scope_name):
            graph_emb_shape = (len(self.env.graph_vocab), self.embedding_dim)
            if graph_embeddings is None:
                graph_embeddings = tf.get_variable("graph_embeddings", shape=graph_emb_shape,
                                                   initializer=GRAPH_EMBEDDING_INITIALIZER())
            assert tuple(graph_embeddings.get_shape().as_list()) == graph_emb_shape

            self.graph_embeddings = graph_embeddings

    def _build_graph(self):
        with tf.variable_scope(self._scope_name):
            self.graph_toks = tf.placeholder(tf.int32, shape=(None, self.max_timesteps),
                                          name="graph_toks")
            self.temperature = tf.constant(1.0, name="temperature")

            graph_window = tf.nn.embedding_lookup(self.graph_embeddings, self.graph_toks)
            graph_window = tf.reduce_mean(graph_window, 1)
            graph_window = tf.stop_gradient(graph_window)

            batch_size = tf.shape(self.graph_toks)[0]

            # CLM à la Andreas.

            # TODO: ugly trick to get the right dimension. is there a better way?
            prev_words = tf.zeros((batch_size, len(self.env.vocab)))
            last_word = tf.zeros((batch_size, len(self.env.vocab)))

            # Now run a teeny utterance decoder.
            outputs, probs, samples = [], [], []
            output_dim = self.env.vocab_size
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(1, [prev_words, last_word, graph_window])
                    output_t = layers.fully_connected(input_t,
                                                      output_dim, tf.identity)

                    probs_t = tf.nn.softmax(output_t / self.temperature)

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
                 embeddings=None, graph_embeddings=None, embedding_dim=10):
        self.env = env
        self._scope_name = scope
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

        self.train_op = None

        self._build_embeddings(embeddings, graph_embeddings)
        self._build_graph()

    def _build_embeddings(self, embeddings, graph_embeddings):
        with tf.variable_scope(self._scope_name):
            emb_shape = (self.env.vocab_size, self.embedding_dim)
            if embeddings is None:
                embeddings = tf.get_variable("embeddings", emb_shape)
            assert tuple(embeddings.get_shape().as_list()) == emb_shape

            graph_emb_shape = (len(self.env.graph_vocab), self.embedding_dim)
            if graph_embeddings is None:
                graph_embeddings = tf.get_variable("graph_embeddings", shape=graph_emb_shape)
            assert tuple(graph_embeddings.get_shape().as_list()) == graph_emb_shape

            self.embeddings = embeddings
            self.graph_embeddings = graph_embeddings

    def _build_graph(self):
        with tf.variable_scope(self._scope_name):
            self.graph_toks = tf.placeholder(tf.int32, shape=(None, self.max_timesteps),
                                          name="graph_toks")
            self.temperature = tf.constant(1.0, name="temperature")

            batch_size = tf.shape(self.graph_toks)[0]
            graph_window_dim = self.max_timesteps * self.embedding_dim
            graph_window = tf.nn.embedding_lookup(self.graph_embeddings, self.graph_toks)
            graph_window = tf.reshape(graph_window, (batch_size, graph_window_dim))

            null_embedding = tf.gather(self.embeddings,
                                       tf.fill((batch_size,), self.env.word_unk_id))

            # Now run a teeny utterance decoder.
            outputs, probs, samples = [], [], []
            output_dim = self.env.vocab_size
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat([prev_sample, graph_window], axis=1)
                    output_t = layers.fully_connected(input_t,
                                                      output_dim, tf.identity)
                    probs_t = tf.nn.softmax(output_t / self.temperature)

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1), [1],
                                          name="sample_%i" % t)
                    prev_sample = tf.nn.embedding_lookup(self.embeddings, sample_t)

                    # TODO support stop token here?

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.probs = probs
            self.samples = samples

