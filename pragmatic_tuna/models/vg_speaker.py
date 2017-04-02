"""
Defines speaker model for VG.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from pragmatic_tuna.util import orthogonal_initializer


EMBEDDING_INITIALIZER = orthogonal_initializer()
LF_EMBEDDING_INITIALIZER = orthogonal_initializer()


class SpeakerModel(object):
    
    def __init__(self, env, embedding_dim=128, lf_len=3, lf_embeddings=None):
        
        self.env = env
        self.lf_len = lf_len
        self.embedding_dim = embedding_dim
        
        self._build_embeddings(lf_embeddings)
        self._bulld_graph()

    def _build_embeddings(self, lf_embeddings):
        lf_emb_shape = (len(self.env.graph_vocab), self.embedding_dim)
        if lf_embeddings is None:
            lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape,
                                                initializer=LF_EMBEDDING_INITIALIZER)
            assert tuple(lf_embeddings.get_shape().as_list()) == lf_emb_shape

        self.lf_embeddings = lf_embeddings
        
        #TODO: word embeddings

    def _build_graph(self):

        self.lf_toks = tf.placeholder(tf.int32, shape=(None, self.lf_len),
                                      name="lf_toks")
        lf_window = tf.nn.embedding_lookup(self.lf_embeddings, self.lf_toks)
        lf_window = tf.reshape(lf_window, (-1, self.lf_len * self.embedding_dim))
        
        
        #TODO: implement seq output layer
        
        pass

    def score(self, lf, words):
        pass


    def sample(self, lf):
        pass
        
    def observe(self, lf, words):
        pass

    def batch_observe(self, lfs, words):
        pass
        
