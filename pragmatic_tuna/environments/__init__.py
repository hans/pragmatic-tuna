import copy
import itertools
import json
import random

from frozendict import frozendict
import gym
from gym import spaces
import nltk
import numpy as np


UNK = "<unk>"


def freeze(val):
    """
    Recursively freeze dictionaries in the given structure.
    """
    if isinstance(val, dict):
        return frozendict({k: freeze(v) for k, v in val.items()})
    elif isinstance(val, list):
        return [freeze(v) for v in val]
    return val


class TUNAEnv(gym.Env):

    def __init__(self, corpus_path, corpus_selection=None, bag=False,
                 randomize=False, repeat_until_success=False):
        with open(corpus_path, "r") as corpus_f:
            corpus_data = json.load(corpus_f)
            if corpus_selection is None:
                corpus = next(iter(corpus_data.values()))
            else:
                corpus = corpus_data[corpus_selection]
            corpus = freeze(corpus)
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

        self.randomize = randomize
        self.repeat_until_success = repeat_until_success

        self._cursor = 0
        self.dreaming = False

    @property
    def action_space(self):
        return spaces.Discrete(self.domain_size)

    @property
    def observation_space(self):
        return self._observation_space

    def _observe(self):
        items = [self._item_to_vector(item) for item in self._domain]

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
            return np.array(items), bag_of_words, desc_words

    def _item_to_vector(self, item):
        vec = np.zeros(self.attr_dim)
        offset = 0

        for attribute in sorted(self._attributes.keys()):
            val = item["attributes"][attribute]
            mapping_dict = self.attributes_to_idx[attribute]

            vec[offset + mapping_dict[val]] = 1
            offset += len(mapping_dict)

        return vec

    def _configure(self, dreaming=False):
        """Runtime configuration."""
        self.dreaming = dreaming

    @property
    def _essential_attributes(self):
        """
        A collection of attributes which should not be modified in dream trials.
        """
        return []

    def _dream_trial(self):
        """
        "Dream" a trial by recalling a seen trial and making minor
        modifications. This is basically a data-augmentation trick.
        """
        # We don't support dreaming after randomized trials right now.
        # Possible to implement; just need to track what has been observed
        # so far explicitly
        assert not self.randomize

        # Dream about the most recent trial by default.
        trial = copy.deepcopy(self._trials[self._cursor - 1])

        # TODO: randomly change referent?

        # Modify trial, leaving "essential" attributes unchanged.
        to_change = set(self._attributes.keys()) - set(self._essential_attributes)
        for item in trial["domain"]:
            for attribute in to_change:
                values = self._attributes[attribute]
                item["attributes"][attribute] = random.choice(values)

        return trial

    def _reset(self):
        if self.randomize:
            self._trial = random.choice(self._trials)
        elif self.dreaming:
            self._trial = self._dream_trial()
        else:
            self._cursor = self._cursor % len(self._trials)
            self._trial = self._trials[self._cursor]
            self._cursor += 1

        self._domain = self._trial["domain"]
        return self._observe()

    def _step(self, action):
        chosen = self._domain[action]

        reward = 0.5 if chosen["target"] else -0.5
        done = True
        info = {}

        if self.repeat_until_success and not chosen:
            self._cursor -= 1

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


class TUNAWithLoTEnv(TUNAEnv):
    """
    A TUNA environment in which "actions" consist of decoding internal logical
    forms / language-of-thought code.

    The code representation is very limited right now; we decode strings of the
    form `FN(ATOM)`. Actions consist then of two classification decisions: over
    the function space (possibly selecting a null/identity function) and the
    atom-space.
    """

    def __init__(self, corpus_path, functions=None, atom_attribute="shape",
                 max_conjuncts=1, **kwargs):
        """
        Args:
            functions: List of possible LF function types. Each list item is a
                tuple of the form `(name, lambda)`, where `name` is a string
                `name` and `lambda` ... TODO
        """
        super(TUNAWithLoTEnv, self).__init__(corpus_path, **kwargs)

        self.atom_attribute = atom_attribute
        self.max_conjuncts = max_conjuncts
        self._build_lfs(functions, atom_attribute)

    def _build_lfs(self, functions, atom_attribute):
        """
        Prepare definition of logical form space.

        Args:
            atom_attribute: Item attribute which should be yield to produce
                a collection of unique atoms. For example, if we use the TUNA
                `TYPE` attribute here, we will derive unique atoms like
                `CHAIR`, `DESK`, etc.
        """

        def id_fn(sources, candidate):
            if len(sources) != 1: return False
            return sources[0] == candidate
        id_function = ("id", id_fn)
        lf_functions = [id_function] + sorted(functions or [])
        self.lf_functions = lf_functions
        self.lf_functions_map = dict(lf_functions)

        self.lf_atoms = sorted(self.attributes_to_idx[atom_attribute])

        # Also prepare a unified LF-language vocabulary
        self.lf_vocab = list(self.lf_functions_map.keys()) + self.lf_atoms
        # No overlap between function and atom names
        assert len(self.lf_vocab) == len(set(self.lf_vocab))
        self.lf_token_to_id = {token: idx for idx, token in enumerate(self.lf_vocab)}

    def _resolve_atom(self, atom_str):
        return [item for item in self._domain
                if item["attributes"][self.atom_attribute] == atom_str]

    @property
    def action_space(self):
        return spaces.DiscreteSequence(len(self.lf_vocab), self.max_conjuncts * 2)

    def resolve_lf_part(self, lf_function, lf_atom):
        atom_objs = self._resolve_atom(lf_atom)
        if atom_objs:
            return [item for item in self._domain
                    if lf_function(atom_objs, item)]
        return []

    def resolve_lf(self, id_list):
        matches = self._domain
        for fn_id, atom_id in zip(id_list[::2], id_list[1::2]):
            fn = self.lf_functions_map[self.lf_vocab[fn_id]]
            atom = self.lf_vocab[atom_id]
            assert atom in self.lf_atoms

            matches = self._intersect_list(matches,
                                           self.resolve_lf_part(fn, atom))

        return matches

    def _intersect_list(self, xs, ys):
        ret = []
        for x in xs:
            if x in ys:
                ret.append(x)
        return ret

    def describe_lf(self, id_list):
        parts = ["%s(%s)" % (self.lf_vocab[fn_id], self.lf_vocab[atom_id])
                 for fn_id, atom_id in zip(id_list[::2], id_list[1::2])]
        return " AND ".join(parts)

    def sample_part(self, available_atoms=None):
        """
        Uniformly sample an LF part (optionally scoping atom space).
        """
        if available_atoms is None:
            available_atoms = self.lf_atoms
        fn_name, _ = random.choice(self.lf_functions)
        atom = random.choice(available_atoms)

        # Convert to LF token IDs.
        return [self.lf_token_to_id[fn_name], self.lf_token_to_id[atom]]

    def sample_lf(self, referent=None):
        """
        Sample a logical form representation `z ~ p(z|r, w)` for the current
        world `w` with referent `r`.

        Args:
            referent: ID of referent. If not given, the target referent for
                this trial is used.
        """
        if referent is None:
            referent = [item for item in self._domain if item["target"]][0]
        else:
            referent = self._domain[referent]

        # Find possible atoms to use.
        available_atoms = list(set([item["attributes"][self.atom_attribute]
                                    for item in self._domain]))

        # Rejection-sample an LF.
        i = 0
        while True:
            if i > 100:
                raise RuntimeError("Failed to sample a valid LF after 100 "
                                   "attempts")

            # TODO magic here: sampling # of parts
            n_parts = min(np.random.geometric(0.5), self.max_conjuncts)
            lf = list(itertools.chain.from_iterable(
                self.sample_part(available_atoms) for _ in range(n_parts)))

            matches = self.resolve_lf(lf)
            if matches and matches[0] == referent:
                return lf

            i += 1

    @property
    def _essential_attributes(self):
        return [self.atom_attribute]

    def _step(self, action):
        # DEBUG: print string_desc -> sampled fn(atom)
        print("%s => %s" % (self._trial["string_description"],
                            self.describe_lf(action)))

        matches = self.resolve_lf(action)
        finished = len(matches) == 1

        success = finished and matches[0]["target"]
        reward = 0.5 if success else -0.5
        done = True
        info = {}

        if self.repeat_until_success and not success:
            self._cursor -= 1

        return None, reward, done, info


if __name__ == "__main__":
    env = TUNAEnv("data/tuna.json")

    np.set_printoptions(threshold=np.inf)
    print(env.reset())
    print(env.step(2))
