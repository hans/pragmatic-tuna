"""
Defines ranking listener models.
"""

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
        raise NotImplementedError


class BoWRankingListener(RankingListenerModel):

    def __init__(self, env, embeddings=None, graph_embeddings=None,
                 embedding_dim=50, **kwargs):
        self.embeddings = embeddings
        self.graph_embeddings = graph_embeddings
        self.embedding_dim = embedding_dim

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

        self.candidates = tf.placeholder(tf.int32, shape=(None, self.max_candidates, 3),
                                         name="candidates")
        # TODO support training graph as well

        embedded = [tf.nn.embedding_lookup(self.embeddings, words_i)
                    for words_i in self.words]
        # TODO handle variable lengths better
        embedded = tf.reduce_mean(embedded, axis=0)

        # Embed candidates.
        embedded_cands = tf.nn.embedding_lookup(self.graph_embeddings, self.candidates)
        embedded_cands = tf.reshape(embedded_cands, (-1, self.max_candidates, 3 * self.embedding_dim))
        print(embedded_cands.get_shape())
