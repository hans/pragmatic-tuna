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
        self._scope = tf.variable_scope9scope)
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


class SequenceSpeakerModel(object):

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

    def score(self, z, u):
        sess = tf.get_default_session()

        z = self._pad_lf_idxs(z)
        words = [self.env.word2idx[word] for word in u]

        feed = {self.lf_toks: z}
        feed.update({self.samples[t]: word for t, word in enumerate(words)})

        probs = sess.run(self.probs[:len(words)], feed)
        probs = [probs_t[word_t] for probs_t, word_t in zip(probs, words)]
        return np.log(np.prod(probs))

    def observe(self, obs, gold_lf):
        if gold_lf is None:
            return

        z = self._pad_lf_idxs(gold_lf)

        words = [self.env.word2idx[word] for word in obs[1]]
        real_length = min(len(words) + 1, self.max_timesteps) # train to output a single EOS token
        # Add a EOS token to words
        if len(words) < self.max_timesteps:
            words.append(self.env.word_eos_id)

        sess = tf.get_default_session()
        feed = {self.lf_toks: z, self.xent_gold_length: real_length}
        feed.update({self.xent_gold_words[t]: word_t
                     for t, word_t in enumerate(words)})
        feed.update({self.samples[t]: word_t
                     for t, word_t in enumerate(words)})
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
                    output_t = layers.fully_connected(tf.expand_dims(input_t, 0),
                                                      output_dim, tf.identity)
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


