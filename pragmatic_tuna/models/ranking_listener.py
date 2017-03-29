"""
Defines ranking listener models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers

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

        self.max_candidates = self.max_negative_samples = max_negative_samples

        with tf.variable_scope(scope):
            self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def rank(self, words_batch, candidates_batch):
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

    def observe(self, words_batch, true_referent_batch, false_referents_batch):
        """
        Observe a batch of references from words_batch -> true_referent_batch
        with negatively sampled false_referents_batch.

        Args:
            words_batch: batch_size list of lists of vocab tokens
            true_referent_batch: batch_size list of candidate tuples
            false_referents_batch: batch_size list of lists of negative
                candidate tuples (each sub-list corresponds to negatively
                sampled candidates for a trial)
        Returns:
            None
        """
        # TODO: update docs for true_referent_batch (it's nested like
        # false_referents just because symmetry is easier)
        raise NotImplementedError

    def _prepare_batch(self, words_batch, candidates_batch):
        """
        Pad in-place.

        Returns:
            words: num_timesteps * batch_size padded indices
            candidates: batch_size * num_candidates * 3
        """
        # Pad words.
        lengths = np.empty(len(words_batch))
        eos_id = self.env.vocab2idx[self.env.EOS]
        for i, words_i in enumerate(words_batch):
            lengths[i] = len(words_i)
            if lengths[i] < self.max_timesteps:
                words_i.extend([eos_id] * (self.max_timesteps - lengths[i]))
        words_batch = np.asarray(words_batch).T

        # Pad candidates.
        num_candidates = np.empty(len(candidates_batch))
        for i, candidates_i in enumerate(candidates_batch):
            num_candidates[i] = len(candidates_i)
            if num_candidates[i] < self.max_candidates:
                pad_length = self.max_candidates - num_candidates[i]
                candidates_i.extend([(0, 0, 0)] * (pad_length))

        return words_batch, candidates_batch, lengths, num_candidates


class BoWRankingListener(RankingListenerModel):

    def __init__(self, env, embeddings=None, graph_embeddings=None,
                 embedding_dim=50, hidden_dim=128, **kwargs):
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

        self.lengths = tf.placeholder(tf.int32, shape=(None,))
        self.words = [tf.placeholder(tf.int32, shape=(None,), name="words_%i" % i)
                      for i in range(self.max_timesteps)]

        self.candidates = tf.placeholder(tf.int32, shape=(None, None, 3),
                                         name="candidates")
        num_candidates = tf.shape(self.candidates)[1]
        # TODO support training graph as well

        # Embed utterances.
        embedded = [tf.nn.embedding_lookup(self.embeddings, words_i)
                    for words_i in self.words]
        # TODO handle variable lengths better
        embedded = tf.reduce_mean(embedded, axis=0)
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

        # Take dot product to yield scores.
        scores = tf.reduce_sum(embedded * embedded_cands, axis=1)
        self.scores = tf.reshape(scores, (-1, num_candidates))

        ########## Loss
        # Assumes that the positive candidate is always provided first in the
        # candidate list.
        batch_size = tf.shape(self.lengths)[0]
        pos_indices = tf.range(batch_size) * num_candidates

        neg_indices = tf.tile(tf.reshape(tf.range(1, num_candidates), (1, -1)), (batch_size, 1))
        neg_indices += tf.reshape(tf.range(batch_size) * num_candidates, (-1, 1))
        neg_indices = tf.reshape(neg_indices, (-1,))

        objective = tf.reduce_mean(tf.gather(scores, pos_indices)) - tf.reduce_mean(tf.gather(scores, neg_indices))
        # TODO L2
        self.loss = -objective

        ######### Training.
        opt = tf.train.MomentumOptimizer(0.01, 0.9)
        self._train_op = opt.minimize(self.loss)

    def rank(self, words_batch, candidates_batch):
        words_batch, candidates_batch, lengths, num_candidates = \
                self._prepare_batch(words_batch, candidates_batch)

        feed = {self.words[t]: words_batch[t] for t in range(self.max_timesteps)}
        feed[self.candidates] = candidates_batch
        feed[self.lengths] = lengths

        sess = tf.get_default_session()
        scores = sess.run(self.scores, feed)

        scores = [scores_i[:num_candidates_i]
                  for scores_i, num_candidates_i in zip(scores, num_candidates)]
        return scores

    def observe(self, words_batch, true_referents_batch, false_referents_batch):
        words_batch, false_referents_batch, lengths, num_false_referents = \
                self._prepare_batch(words_batch, false_referents_batch)

        candidates_batch = [true_referent_i + false_referents_i
                            for true_referent_i, false_referents_i
                            in zip(true_referents_batch, false_referents_batch)]

        feed = {self.words[t]: words_batch[t] for t in range(self.max_timesteps)}
        feed[self.candidates] = candidates_batch
        feed[self.lengths] = lengths

        sess = tf.get_default_session()
        _, loss = sess.run((self._train_op, self.loss), feed)
        return loss
