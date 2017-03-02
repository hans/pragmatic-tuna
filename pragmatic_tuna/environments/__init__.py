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
EOS = "<eos>"


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

    UNK = UNK
    EOS = EOS

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

        self.vocab = [UNK, EOS] + corpus["vocab"]
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.word_unk_id = self.word2idx[UNK]
        self.word_eos_id = self.word2idx[EOS]
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

    def _observe(self, domain=None):
        if domain == None:
            domain = self._domain

        items = [self._item_to_vector(item) for item in domain]

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
            return np.array(items), desc_words

    def _item_to_vector(self, item):
        vec = np.zeros(self.attr_dim)
        offset = 0

        for attribute in sorted(self._attributes.keys()):
            val = item["attributes"][attribute]
            mapping_dict = self.attributes_to_idx[attribute]

            vec[offset + mapping_dict[val]] = 1
            offset += len(mapping_dict)

        return vec

    def _configure(self, dreaming=False, reset_cursor=False):
        """Runtime configuration."""
        self.dreaming = dreaming

        if reset_cursor:
            self._cursor = 0

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
        trial = self._trials[self._cursor - 1]

        # TODO: randomly change referent?

        # Modify trial, leaving "essential" attributes unchanged.
        # TODO: this is super hacky because of frozendict. Change it.
        to_change = set(self._attributes.keys()) - set(self._essential_attributes)
        new_items = []
        for item in trial["domain"]:
            new_attributes = {attr: random.choice(self._attributes[attr])
                              for attr in to_change}
            attributes = item["attributes"].copy(**new_attributes)
            new_items.append(item.copy(attributes=attributes))

        return trial.copy(items=new_items)

    def sample_prev_trials(self, k):
        if self.randomize:
            raise RuntimeError("Sampling previous trials is not compatible "
                                          "with random traversal of examples.")

        k = min(k, self._cursor)
        trial_idxs = np.random.choice(self._cursor, size=k, replace=False)
        print(trial_idxs)
        y = [self._trials[i] for i in trial_idxs]
        return y

    def _set_trial(self, trial):
        self._trial = trial
        self._domain = self._trial["domain"]

        return self._observe()


    def _reset(self):
        if self.randomize:
            self._trial = random.choice(self._trials)
        elif self.dreaming:
            self._trial = self._dream_trial()
        else:
            self._cursor = self._cursor % len(self._trials)
            self._trial = self._trials[self._cursor]
            self._cursor += 1

        return self._set_trial(self._trial)

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
        self.max_tokens = self.max_conjuncts * 2
        self._build_lfs(functions, atom_attribute)

    def _build_lfs(self, functions, atom_attribute):
        """
        Prepare definition of logical form space.

        Args:
            functions: List of pairs `(name, fn)`
            atom_attribute: Item attribute which should be yield to produce
                a collection of unique atoms. For example, if we use the TUNA
                `TYPE` attribute here, we will derive unique atoms like
                `CHAIR`, `DESK`, etc.
        """

        def id_fn(sources, candidate):
            if len(sources) != 1: return False
            return sources[0] == candidate

        function_map = dict(functions or [])
        function_map["id"] = id_fn
        function_map[EOS] = EOS

        self.lf_functions = sorted(function_map.keys())
        self.lf_function_map = function_map

        self.lf_atoms = sorted(self.attributes_to_idx[atom_attribute]) + [UNK]

        # Also prepare a unified LF-language vocabulary
        self.lf_vocab = self.lf_functions + self.lf_atoms
        # No overlap between function and atom names
        assert len(self.lf_vocab) == len(set(self.lf_vocab))
        self.lf_token_to_id = {token: idx for idx, token in enumerate(self.lf_vocab)}

        self.lf_unk_id = self.lf_token_to_id[UNK]
        self.lf_eos_id = self.lf_token_to_id[EOS]

    def _resolve_atom(self, atom_str, domain=None):
        if domain == None:
            domain = self._domain

        return [item for item in domain
                if item["attributes"][self.atom_attribute] == atom_str]

    @property
    def action_space(self):
        return spaces.DiscreteSequence(len(self.lf_vocab), self.max_conjuncts * 2)

    def resolve_lf_part(self, lf_function, lf_atom, domain=None):
        if domain == None:
            domain = self._domain

        atom_objs = self._resolve_atom(lf_atom)
        if atom_objs:
            items = []
            for atom_obj in atom_objs:
                items.extend([item for item in domain
                    if lf_function(atom_objs, item)])
            return items
        return []

    def resolve_lf(self, id_list, domain=None):
        """
        Resolve an LF token sequence to a list of referents in the current
        domain.

        Returns:
            referents: List whose elements are a subset of `domain`
            domain: List of potential referents or `self._domain` by default
        """
        if len(id_list) == 0 or id_list[0] == self.lf_eos_id:
            return []

        if domain == None:
            domain = self._domain

        matches = domain
        for fn_id, atom_id in zip(id_list[::2], id_list[1::2]):
            if fn_id == self.lf_eos_id:
                break

            fn = self.lf_function_map[self.lf_vocab[fn_id]]
            atom = self.lf_vocab[atom_id]
            assert atom in self.lf_atoms

            matches = self._intersect_list(matches,
                                           self.resolve_lf_part(fn, atom, domain=domain))

        return matches

    def check_lf(self, id_list):
        """
        Check whether the given LF token sequence resolves to a target referent.

        Returns:
            bool
        """
        referents = self.resolve_lf(id_list)
        if not referents:
            return False

        return referents[0]["target"]

    def _intersect_list(self, xs, ys):
        ret = []
        for x in xs:
            if x in ys:
                ret.append(x)
        return ret

    def describe_lf(self, id_list):
        try:
            eos_pos = id_list.index(self.lf_eos_id)
        except ValueError:
            pass
        else:
            id_list = id_list[:eos_pos]

        parts = ["%s(%s)" % (self.lf_vocab[fn_id], self.lf_vocab[atom_id])
                 for fn_id, atom_id in zip(id_list[::2], id_list[1::2])]
        return " AND ".join(parts)

    def sample_part(self, available_atoms=None):
        """
        Uniformly sample an LF part (optionally scoping atom space).
        """
        if available_atoms is None:
            available_atoms = self.lf_atoms
        fn_name = EOS
        while fn_name == EOS:
            fn_name = random.choice(self.lf_functions)
        atom = random.choice(available_atoms)

        # Convert to LF token IDs.
        return [self.lf_token_to_id[fn_name], self.lf_token_to_id[atom]]

    def sample_lf(self, referent=None, n_parts=None, domain=None):
        """
        Sample a logical form representation `z ~ p(z|r, w)` for the current
        world `w` with referent `r`.

        Args:
            referent: ID of referent. If not given, the target referent for
                this trial is used.
        """

        if domain is None:
            domain = self._domain

        if referent is None:
            referent = [item for item in domain if item["target"]][0]
        elif referent != "any":
            referent = domain[referent]

        # Find possible atoms to use.
        if referent != "any":
            available_atoms = list(set([item["attributes"][self.atom_attribute]
                                    for item in domain]))
        else:
            available_atoms = None

        # Rejection-sample an LF...
        i = 0
        while True:
            if i > 1000:
                raise RuntimeError("Failed to sample a valid LF after 1000 "
                                   "attempts")

            # TODO magic here: sampling # of parts
            #n_parts = min(np.random.geometric(0.5), self.max_conjuncts)
            if n_parts == None:
                n_parts = self.max_conjuncts
            lf = list(itertools.chain.from_iterable(
                self.sample_part(available_atoms) for _ in range(n_parts)))

            if len(lf) > 3 and lf[0:2] == lf[2:4]:
                lf = lf[0:2]

            if referent == "any":
                return lf

            matches = self.resolve_lf(lf)
            if matches and matches[0] == referent:
                return lf

            i += 1

    def enumerate_lfs(self, lf_prefix=(), includeOnlyPossibleReferents=True, domain=None):
        """
            Enumerate all possible LF function-atom combinations.
            If lf_prefix contains part of an LF, the function returns
            the conjunction of lf_prefix and every possible combination
            of fn(atom).
        """
        lfs = []

        if includeOnlyPossibleReferents:
            referents = domain if domain != None else self._domain


        for fn_name in self.lf_functions:
            if fn_name == EOS:
                continue
            for atom in self.lf_atoms:
                lf = lf_prefix
                lf += (self.lf_token_to_id[fn_name],
                       self.lf_token_to_id[atom])
                if len(lf) > 3 and lf[1] == lf[3]:
                    lf = lf[0:2]
                if includeOnlyPossibleReferents:
                    matches = self.resolve_lf(lf)
                    if matches and matches[0] in referents:
                        lfs.append(lf)
                else:
                    lfs.append(lf)
        return lfs

    def get_word_idxs(self, words, pad_to_length=None):
        """
        Convert a list of vocabulary words to word indices and pad at right.
        """
        idxs = [self.word2idx[word] for word in words]
        if pad_to_length is None:
            pad_to_length = self.max_tokens

        assert len(idxs) <= pad_to_length
        if len(idxs) < pad_to_length:
            idxs += [self.word_eos_id] * (pad_to_length - len(idxs))

        return idxs

    def pad_lf_idxs(self, lf_idxs, pad_to_length=None):
        """
        Pad an LF idx sequence.
        """
        lf_idxs = lf_idxs[:]

        if pad_to_length is None:
            pad_to_length = self.max_tokens

        assert len(lf_idxs) <= pad_to_length
        missing_conjuncts = (pad_to_length - len(lf_idxs)) / 2
        assert int(missing_conjuncts) == missing_conjuncts
        lf_idxs += [self.lf_eos_id, self.lf_unk_id] * int(missing_conjuncts)
        return lf_idxs

    def _intersect_lists(self, list1, list2):
        result = []
        for item in list1:
            if item in list2:
                result.append(item)
        return result

    @property
    def _essential_attributes(self):
        return [self.atom_attribute]

    def _step(self, action):
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
