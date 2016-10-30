import json
import random

import gym
from gym import spaces
import numpy as np


class MuteTUNAEnv(gym.Env):

    def __init__(self, corpus_path, corpus_selection="furniture"):
        with open(corpus_path, "r") as corpus_f:
            corpus = json.load(corpus_f)[corpus_selection]
            self._trials = corpus["trials"]
            self._attributes = corpus["attributes"]

        self.domain_size = len(self._trials[0]["domain"])
        self.attr_dim = sum(len(values) for values in self._attributes.values())

        self.attributes_to_idx = {key: {value: idx for idx, value
                                        in enumerate(self._attributes[key])}
                                  for key in sorted(self._attributes.keys())}

        # Observations: for each object, a set of one-hot vectors concatenated
        # into one `attr_dim`-long vector.
        self._observation_space = spaces.Box(low=0, high=1,
                                             shape=(self.domain_size, self.attr_dim))

    @property
    def action_space(self):
        return spaces.Discrete(self.domain_size)

    @property
    def observation_space(self):
        return self._observation_space

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
        items = [self._item_to_vector(item) for item in self._trial["domain"]]
        return np.array(items)

    def _step(self, action):
        chosen = self._trial["domain"][action]

        reward = 1.0 if chosen["target"] else 0.0
        done = True
        info = {}

        return None, reward, done, info


if __name__ == "__main__":
    env = MuteTUNAEnv("data/tuna.json")

    np.set_printoptions(threshold=np.inf)
    print(env.reset())
    print(env.step(2))
