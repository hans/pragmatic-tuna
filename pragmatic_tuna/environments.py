import json
import random

import gym
from gym import spaces
import nltk
import numpy as np


UNK = "<unk>"


class TUNAEnv(gym.Env):

    def __init__(self, corpus_path, corpus_selection="furniture", bag=False):
        with open(corpus_path, "r") as corpus_f:
            corpus = json.load(corpus_f)[corpus_selection]
            self._trials = corpus["trials"]
            self._attributes = corpus["attributes"]

        self.domain_size = len(self._trials[0]["domain"])
        self.attr_dim = sum(len(values) for values in self._attributes.values())

        self.vocab = [UNK] + corpus["vocab"]
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.word_unk_id = self.word2idx[UNK]
        self.vocab_size = len(self.vocab)

        self.attributes_to_idx = {key: {value: idx for idx, value
                                        in enumerate(self._attributes[key])}
                                  for key in sorted(self._attributes.keys())}

        self.bag = bag
        # Observations: product of
        #   1. for each object, a set of one-hot vectors concatenated into one
        #      `attr_dim`-long vector.
        #   2. a bag-of-words representation of the string utterance
        if self.bag:
            # For each item, observe the cross product of
            # `utterance * attributes`.
            #
            # This allows us to get nice interpretable weights for simple models.
            shape = (self.domain_size, self.attr_dim * self.vocab_size)
            self._observation_space = spaces.Box(low=0, high=1, shape=shape)
        else:
            self._observation_space = spaces.Tuple(
                    (spaces.Box(low=0, high=1, shape=(self.domain_size, self.attr_dim)),
                     spaces.Box(low=0, high=1, shape=(self.vocab_size,))))

    @property
    def action_space(self):
        return spaces.Discrete(self.domain_size)

    @property
    def observation_space(self):
        return self._observation_space

    def _observe(self):
        items = [self._item_to_vector(item) for item in self._trial["domain"]]

        desc_words = nltk.word_tokenize(self._trial["string_description"])
        desc_word_ids = np.array([self.word2idx[desc_word]
                                  for desc_word in desc_words])

        bag_of_words = np.zeros(self.vocab_size)
        bag_of_words[desc_word_ids] = 1

        if self.bag:
            # For each item, build a flattened product space
            # of utterance features * attribute features.
            ret = np.empty((self.domain_size, self.attr_dim * self.vocab_size))
            # Prep for broadcasting.
            bag_of_words = bag_of_words[:, np.newaxis]
            for i, item in enumerate(items):
                ret[i]= (bag_of_words * item).flatten()

            # ret = np.transpose(bag_of_words[:, np.newaxis, np.newaxis] * items, (1, 0, 2))

            return ret
        else:
            return np.array(items), bag_of_words

    def _item_to_vector(self, item):
        vec = np.zeros(self.attr_dim)
        offset = 0

        for attribute in sorted(self._attributes.keys()):
            val = item["attributes"][attribute]
            mapping_dict = self.attributes_to_idx[attribute]

            vec[offset + mapping_dict[val]] = 1
            offset += len(mapping_dict)

        return vec

    def _reset(self):
        self._trial = random.choice(self._trials)
        return self._observe()

    def _step(self, action):
        chosen = self._trial["domain"][action]

        reward = 0.5 if chosen["target"] else -0.5
        done = True
        info = {}

        return None, reward, done, info

    def describe_features(self):
        """
        Return a list of string descriptions for the features in observations
        returned by this environment.
        """
        if not self.bag:
            raise NotImplementedError

        idx_to_attribute = {}
        i = 0
        for attribute in sorted(self._attributes.keys()):
            attr_values = self.attributes_to_idx[attribute]
            for value, value_idx in attr_values.items():
                idx_to_attribute[i + value_idx] = (attribute, value)

            i += len(attr_values)

        descs = [(word,) + idx_to_attribute[attr_idx]
                 for word in self.vocab
                 for attr_idx in range(len(idx_to_attribute))]

        return descs


if __name__ == "__main__":
    env = TUNAEnv("data/tuna.json")

    np.set_printoptions(threshold=np.inf)
    print(env.reset())
    print(env.step(2))
