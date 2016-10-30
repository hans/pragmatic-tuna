import random

import gym
from gym import spaces


class MuteTUNAEnv(gym.Env):

    def __init__(self, corpus_path):
        # TODO load corpus
        self._scenes = []

        self.domain_size = 5
        self.attr_dim = 50

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

    def _reset(self):
        self._scene = random.choice(self._scenes)

    def _step(self, action):
        chosen = self._scene[action]

        reward = 1.0 if chosen.target else 0.0
        done = True
        info = {}

        return None, reward, done, info
