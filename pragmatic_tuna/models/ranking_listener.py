"""
Defines ranking listener models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from pragmatic_tuna.util import orthogonal_initializer


EMBEDDING_INITIALIZER = orthogonal_initializer()
LF_EMBEDDING_INITIALIZER = orthogonal_initializer()


class RankingListenerModel(object):

    """
    Parametric ranking listener model mapping from (utterance, referent) -> score.
    """

    def __init__(self, env, max_negative_samples=5, scope="listener"):
        self.env = env
        self.max_timesteps = env.max_timesteps

        self.max_negative_samples = max_negative_samples
        self.max_candidates = self.max_negative_samples + 1

        with tf.variable_scope(scope):
            self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def score(self, words_batch, candidates_batch, lengths, num_candidates):
        """
        Rank possible candidate referents for the given batch of word inputs.

        Args:
            words_batch: batch_size list of lists of vocab tokens
            candidates_batch: batch_size list of lists of candidate tuples
                (each sub-list corresponds to the possible candidates for a
                trial)
        Returns:
            scores: batch_size list of lists (each sub-list contains scores
                corresponding to provided candidates)
        """
        raise NotImplementedError

    def observe(self, words_batch, candidates_batch, lengths, num_candidates,
                true_referent_position=0):
        """
        Observe a batch of references with positive candidates and negatively
        sampled false candidates.

        Args:
            words_batch: batch_size list of lists of vocab tokens
            candidates_batch: batch_size list of lists of candidate tuples.
                The true referent should lie at position
                `true_referent_position` within each sublist.
            true_referent_position:
        Returns:
            None
        """
        raise NotImplementedError


class BoWRankingListener(RankingListenerModel):

    def __init__(self, env, embeddings=None, graph_embeddings=None,
                 embedding_dim=64, hidden_dim=256, **kwargs):
        self.embeddings = embeddings
        self.graph_embeddings = graph_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        super().__init__(env, **kwargs)

    def _init_embeddings(self):
        embeddings_shape = (len(self.env.vocab), self.embedding_dim)
        if self.embeddings is not None:
            assert tuple(self.embeddings.get_shape().as_list()) == embeddings_shape
        else:
            self.embeddings = tf.get_variable("embeddings", shape=embeddings_shape,
                                              initializer=orthogonal_initializer())

        graph_embeddings_shape = (len(self.env.graph_vocab), self.embedding_dim)
        if self.graph_embeddings is not None:
            assert tuple(self.graph_embeddings.get_shape().as_list()) == graph_embeddings_shape
        else:
            self.graph_embeddings = tf.get_variable("graph_embeddings", shape=graph_embeddings_shape,
                                                    initializer=orthogonal_initializer())

    def _build_graph(self):
        self._init_embeddings()

        self.lengths = tf.placeholder(tf.int32, shape=(None,), name="lengths")
        self.words = [tf.placeholder(tf.int32, shape=(None,), name="words_%i" % i)
                      for i in range(self.max_timesteps)]

        self.candidates = tf.placeholder(tf.int32, shape=(None, None, 3),
                                         name="candidates")
        num_candidates = tf.shape(self.candidates)[1]

        # Embed utterances.
        lengths_temp = tf.to_float(tf.expand_dims(self.lengths, 1))
        embedded = [tf.to_float(t < lengths_temp) \
                        * tf.nn.embedding_lookup(self.embeddings, words_t)
                    for t, words_t in enumerate(self.words)]
        embedded = tf.reduce_sum(embedded, axis=0) / lengths_temp
        embedded = layers.fully_connected(embedded, self.hidden_dim)

        # Embed candidates.
        embedded_cands = tf.nn.embedding_lookup(self.graph_embeddings, self.candidates)
        # Flatten candidates to 2d matrix.
        embedded_cands = tf.reshape(embedded_cands, (-1, 3 * self.embedding_dim))
        embedded_cands = layers.fully_connected(embedded_cands, self.hidden_dim)

        # Tile utterance representations.
        embedded = tf.reshape(embedded, (-1, 1, self.hidden_dim))
        embedded = tf.tile(embedded, (1, num_candidates, 1))
        embedded = tf.reshape(embedded, (-1, self.hidden_dim))

        # Concat and compute a bit more.
        concat = tf.concat(1, (embedded, embedded_cands))
        concat = layers.fully_connected(concat, self.hidden_dim)
        scores = tf.squeeze(layers.fully_connected(concat, 1, activation_fn=tf.tanh), [1])

        # Take dot product to yield scores.
        self.scores = tf.reshape(scores, (-1, num_candidates))

        ########## Loss
        # Assumes that the positive candidate is always provided first in the
        # candidate list.
        batch_size = tf.shape(self.lengths)[0]
        pos_indices = tf.range(batch_size) * num_candidates

        neg_indices = tf.tile(tf.reshape(tf.range(1, num_candidates), (1, -1)), (batch_size, 1))
        neg_indices += tf.reshape(tf.range(batch_size) * num_candidates, (-1, 1))
        neg_indices = tf.reshape(neg_indices, (-1,))

        pos_scores = tf.reshape(tf.gather(scores, pos_indices), (-1, 1))
        neg_scores = tf.reshape(tf.gather(scores, neg_indices), (-1, num_candidates - 1))
        margin = tf.constant(0.5, name="loss_margin")
        margin_diffs = tf.maximum(0., margin + neg_scores - pos_scores)
        margin_diffs = tf.reduce_sum(margin_diffs, axis=1)
        self.loss = tf.reduce_mean(margin_diffs)
        tf.summary.scalar("loss", self.loss)

        self.train_op = None

    def score(self, words_batch, candidates_batch, lengths_batch, num_candidates):
        feed = {self.words[t]: words_batch[t] for t in range(self.max_timesteps)}
        feed[self.candidates] = candidates_batch
        feed[self.lengths] = lengths_batch

        sess = tf.get_default_session()
        scores = sess.run(self.scores, feed)

        scores = [scores_i[:num_candidates_i]
                  for scores_i, num_candidates_i in zip(scores, num_candidates)]
        return scores

    def observe(self, words_batch, candidates_batch, lengths_batch, num_candidates,
                true_referent_position=0):
        # TODO don't update on false candidates
        # TODO monitor avg embedding norm to make sure we're not going crazy
        feed = {self.words[t]: words_batch[t] for t in range(self.max_timesteps)}
        feed[self.candidates] = candidates_batch
        feed[self.lengths] = lengths_batch

        sess = tf.get_default_session()
        _, loss = sess.run((self.train_op, self.loss), feed)
        return loss
