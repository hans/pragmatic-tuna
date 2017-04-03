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

    def sample(self, subgraphs, argmax=False):
        """
        Sample utterances using the given graph representations.

        Args:
            subgraphs: list of referent tuples
        Returns:
            utterances: `num_timesteps * batch_size` batch of utterances as
                vocab IDs
        """
        raise NotImplementedError

    def score(self, utterances, subgraphs, lengths, n_cands):
        """
        Score the given batch of utterances and potential referents.
        """
        raise NotImplementedError

    def observe(self, words_batch, candidates_batch, lengths, n_cands,
                true_referent_position=0):
        """
        Observe a batch of referents with positive candidates and negatively
        sampled false candidates.

        Args:
            words_batch:
            candidates_batch:
            true_referent_position:
        """
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

    def score(self, words, cands, lengths, n_cands):
        # TODO a lot of stuff in this method could happen in a TF graph. If
        # necessary..
        sess = tf.get_default_session()

        # Replicate word inputs so that we have a one-to-one
        # candidate-utterance relationship.
        num_timesteps, batch_size = words.shape
        num_candidates = len(cands[0])
        words = words.reshape((num_timesteps, batch_size, 1))
        words = np.tile(words, (1, 1, num_candidates))
        words = words.reshape(num_timesteps, -1)
        # Also replicate the lengths
        lengths_rep = lengths.reshape((batch_size, 1))
        lengths_rep = np.tile(lengths_rep, (1, num_candidates))
        lengths_rep = lengths_rep.reshape((-1,))

        feed = {self.samples[t]: words_t
                for t, words_t in enumerate(words)}
        feed[self.graph_toks] = cands

        # probs: `num_timesteps * (batch_size * max_cands) * vocab_size`
        probs = sess.run(self.probs, feed)

        # Collect cumulative probabilities for each input sequence (each
        # candidate in each example)
        n_inputs = len(probs[0])
        cum_probs = np.zeros(n_inputs)
        arange = np.arange(n_inputs)
        for t, (probs_t, words_t) in enumerate(zip(probs, words)):
            mask_t = t < lengths_rep
            probs_t = mask_t * probs_t[arange, words_t]
            cum_probs += np.log(probs_t)
        cum_probs = cum_probs.reshape((batch_size, num_candidates))

        scores = [cum_probs_i[:n_cands_i]
                  for cum_probs_i, n_cands_i in zip(cum_probs, n_cands)]
        return scores

    def observe(self, words_batch, candidates_batch, lengths, n_cands,
                true_referent_position=0):
        graph_toks = [candidates_i[true_referent_position]
                      for candidates_i in candidates_batch]

        sess = tf.get_default_session()
        feed = {self.xent_gold_words[t]: words_t
                for t, words_t in enumerate(words_batch)}
        feed.update({self.samples[t]: words_t
                     for t, words_t in enumerate(words_batch)})
        feed[self.graph_toks] = graph_toks
        feed[self.xent_gold_length] = lengths
        sess.run(self.train_op, feed)


class WindowedSequenceSpeakerModel(SequenceSpeakerModel):

    """
    Windowed sequence speaker/decoder model.
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
            self.graph_toks = tf.placeholder(tf.int32,
                                             shape=(None, None, 3),
                                             name="graph_toks")
            self.temperature = tf.constant(1.0, name="temperature")

            graph_window_dim = 3 * self.embedding_dim
            graph_toks = tf.reshape(self.graph_toks, (-1, 3))
            graph_window = tf.nn.embedding_lookup(self.graph_embeddings, graph_toks)
            graph_window = tf.reshape(graph_window, (-1, graph_window_dim))

            # num_inputs = batch_size * max_candidates
            num_inputs = tf.shape(graph_window)[0]
            null_embedding = tf.gather(self.embeddings,
                                       tf.fill((num_inputs,), self.env.word_unk_id))

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

