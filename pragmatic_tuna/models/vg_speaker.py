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

    def sample(self, subgraphs, argmax=False):
        sess = tf.get_default_session()
        feed = {self.graph_toks: subgraphs,
                self.temperature: 1e-8 if argmax else 1.0}

        samples = sess.run(self.samples, feed)
        # TODO: reshape to fit subgraphs shape to be nice.
        return samples

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
        scores = self._probs_to_scores(probs, words, lengths_rep)

        # Reshape and trim off any padded input candidates.
        scores = scores.reshape((batch_size, num_candidates))
        scores = [scores_i[:n_cands_i]
                  for scores_i, n_cands_i in zip(scores, n_cands)]
        return scores

    def _probs_to_scores(self, probs, words, lengths):
        """
        Fold over a sequence of probability distribution batches with a batch of
        actual word sequences to produce a batch of predictions p(sequence).
        """

        n_inputs = len(probs[0])
        cum_probs = np.zeros(n_inputs)
        arange = np.arange(n_inputs)
        for t, (probs_t, words_t) in enumerate(zip(probs, words)):
            mask_t = t < lengths
            probs_t = mask_t * probs_t[arange, words_t]
            cum_probs += np.log(probs_t + 1e-8)

        return cum_probs

    def observe(self, words_batch, candidates_batch, lengths, n_cands,
                true_referent_position=0):
        graph_toks = [[candidates_i[true_referent_position]]
                      for candidates_i in candidates_batch]

        sess = tf.get_default_session()
        feed = {self.gold_words[t]: words_t
                for t, words_t in enumerate(words_batch)}
        feed.update({self.samples[t]: words_t
                     for t, words_t in enumerate(words_batch)})
        feed[self.graph_toks] = graph_toks
        feed[self.gold_length] = lengths

        ret = sess.run([self.train_op, self.loss] + self.probs, feed)
        loss = ret[1]

        probs = ret[2:]
        scores = self._probs_to_scores(probs, words_batch, lengths)
        avg_prob = np.exp(scores).mean()

        return loss, avg_prob


class WindowedSequenceSpeakerModel(SequenceSpeakerModel):

    """
    Windowed sequence speaker/decoder model.
    """
    # could probably be unified with WindowedSequenceListenerModel if there is
    # sufficient motivation.

    def __init__(self, env, scope="speaker", max_timesteps=4,
                 embeddings=None, graph_embeddings=None, embedding_dim=10,
                 dropout_keep_prob=0.8):
        self.env = env
        self._scope_name = scope
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim
        self.dropout_keep_prob = dropout_keep_prob

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
            prev2_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    ######### Feedforward.

                    input_t = tf.concat(1, [prev_sample, prev2_sample, graph_window])
                    hidden_t = layers.fully_connected(input_t, 256)
                    hidden_t = tf.nn.dropout(hidden_t, self.dropout_keep_prob)
                    output_t = layers.fully_connected(input_t,
                                                      output_dim, tf.identity)
                    probs_t = tf.nn.softmax(output_t / self.temperature)

                    ######### Sampling / history update.

                    prev2_sample = prev_sample

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


            ######### Loss
            gold_words = [tf.placeholder(tf.int32, shape=(None,),
                                         name="gold_word_%i" % t)
                          for t in range(self.max_timesteps)]
            gold_length = tf.placeholder(tf.int32, shape=(None,),
                                         name="gold_lengths")
            self.gold_words = gold_words
            self.gold_length = gold_length

            loss_weights = [tf.to_float(t < gold_length)
                            for t in range(self.max_timesteps)]
            loss = tf.nn.seq2seq.sequence_loss(self.outputs, gold_words,
                                               loss_weights)
            self.loss = loss

