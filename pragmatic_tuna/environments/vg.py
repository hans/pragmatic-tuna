from collections import Counter
import json

try: import cPickle as pickle
except: import pickle

import gym
import numpy as np


UNK = "<unk>"
EOS = "<eos>"


class VGEnv(gym.Env):

    UNK = UNK
    EOS = EOS

    def __init__(self, corpus_path, max_negative_samples=5):
        if corpus_path.endswith(".pkl"):
            with open(corpus_path, "rb") as corpus_pkl:
                self.corpora, self.vocab, self.graph_vocab = pickle.load(corpus_pkl)
        else:
            self.corpora, self.vocab, self.graph_vocab = self._process_corpus(corpus_path)
            pkl_path = corpus_path[:corpus_path.rindex(".")] + ".pkl"
            with open(pkl_path, "wb") as corpus_pkl:
                pickle.dump((self.corpora, self.vocab, self.graph_vocab), corpus_pkl,
                            pickle.HIGHEST_PROTOCOL)

        self.vocab2idx = {w: idx for idx, w in enumerate(self.vocab)}
        self.graph_vocab2idx = {w: idx for idx, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab2idx)
        self.word_unk_id = self.vocab2idx[UNK]

        self.max_timesteps = max([len(trial["utterance"])
                                  for corpus in self.corpora
                                  for trial in self.corpora[corpus]])

        # Assumes 1 positive candidate per example
        self.max_negative_samples = max_negative_samples
        self.max_candidates = max_negative_samples + 1

    def _process_corpus(self, corpus_path):
        with open(corpus_path, "r") as corpus_f:
            corpus_data = json.load(corpus_f)

        corpora = {}
        vocab_counts, graph_vocab = Counter(), set()
        for trial in corpus_data:
            if trial["type"] not in corpora:
                corpora[trial["type"]] = []

            utterance = trial["utterance"].lower().strip(".!?").split()

            # DEV: Skip trials with long utterances.
            if len(utterance) > 10:
                continue

            for word in utterance:
                vocab_counts[word] += 1

            domain_positive, domain_negative = [], []
            for subgraph in trial["domain"]:
                obj1 = subgraph["object1"]
                obj2 = subgraph["object2"]
                reln = subgraph["reln"]
                graph_vocab.add(obj1)
                graph_vocab.add(obj2)
                graph_vocab.add(reln)

                domain = domain_positive if subgraph["target"] else domain_negative
                domain.append((reln, obj1, obj2))

            corpora[trial["type"]].append({
                "utterance": utterance,
                "domain_positive": domain_positive,
                "domain_negative": domain_negative
            })


        vocab = [UNK, EOS] + list(sorted([word for word, freq in vocab_counts.items()
                                          if freq > 1]))
        vocab2idx = {w: idx for idx, w in enumerate(vocab)}
        graph_vocab = [EOS] + list(sorted(graph_vocab))
        graph_vocab2idx = {w: idx for idx, w in enumerate(graph_vocab)}

        # Now reprocess trials, replacing strings with IDs.
        unk_id = vocab2idx[UNK]
        for corpus_name, corpus in corpora.items():
            for trial in corpus:
                trial["utterance"] = [vocab2idx.get(word, unk_id)
                                      for word in trial["utterance"]]
                trial["domain_positive"] = [tuple([graph_vocab2idx[x] for x in subgraph])
                                            for subgraph in trial["domain_positive"]]
                trial["domain_negative"] = [tuple([graph_vocab2idx[x] for x in subgraph])
                                            for subgraph in trial["domain_negative"]]

        return corpora, vocab, graph_vocab

    def _prepare_batch(self, words_batch, candidates_batch):
        """
        Pad a batch (not in-place).

        Returns:
            words: num_timesteps * batch_size padded indices
            candidates: batch_size * num_candidates * 3
            lengths: batch_size vector of original utterance lengths
            num_candidates: batch_size vector of original num. candidates per
                example
        """

        # Pad words.
        lengths = np.empty(len(words_batch))
        eos_id = self.vocab2idx[self.EOS]
        words_batch_ret = []
        for i, words_i in enumerate(words_batch):
            lengths[i] = len(words_i)
            ret_i = words_i[:]
            if lengths[i] < self.max_timesteps:
                ret_i.extend([eos_id] * (self.max_timesteps - lengths[i]))
            words_batch_ret.append(ret_i)
        words_batch_ret = np.asarray(words_batch_ret).T

        # Pad candidates.
        num_candidates = np.empty(len(candidates_batch))
        candidates_batch_ret = []
        for i, candidates_i in enumerate(candidates_batch):
            num_candidates[i] = len(candidates_i)
            ret_i = candidates_i[:]
            if num_candidates[i] < self.max_candidates:
                pad_length = self.max_candidates - num_candidates[i]
                ret_i.extend([(0, 0, 0)] * (pad_length))
            candidates_batch_ret.append(ret_i)

        return words_batch_ret, candidates_batch_ret, lengths, num_candidates


    def get_batch(self, corpus, batch_size=64, negative_samples=5):
        """
        Return a training batch.

        Returns: see `self._prepare_batch`
            words: `max_timesteps * batch_size` vocab token ndarray
            candidates: `batch_size * max_candidates` list of lists. In each
                sublist, the positive candidate always comes first.
            lengths:
            num_candidates:
        """
        # Assumes 1 positive candidate per example
        assert negative_samples <= self.max_negative_samples

        corpus = self.corpora[corpus]
        assert len(corpus) >= batch_size
        idxs = np.random.choice(len(corpus), size=batch_size, replace=False)

        utterances, candidates = [], []
        for idx in idxs:
            trial = corpus[idx]

            assert len(trial["domain_positive"]) == 1, len(trial["domain_positive"]) # At least for now
            candidates_i = trial["domain_positive"][:]

            neg_samples = min(negative_samples, len(trial["domain_negative"]))
            if neg_samples > 0:
                neg_idxs = np.random.choice(len(trial["domain_negative"]), size=neg_samples, replace=False)
                candidates_i.extend([trial["domain_negative"][neg_idx] for neg_idx in neg_idxs])
            else:
                # TODO how to handle this?
                eos_id = self.graph_vocab2idx[EOS]
                candidates_i.append((eos_id, eos_id, eos_id))

            utterances.append(trial["utterance"])
            candidates.append(candidates_i)

        return self._prepare_batch(utterances, candidates)

    def get_silent_batch(self, relation, batch_size=64):
        """
        Return a batch for "dreaming" of grounded relations without paired
        utterances.

        Args:
            relation: instances of relation to fetch
        """
        # TODO: we should have a separate corpus for this
        # -- one where the constraint that only relevant relations appear is
        # not enforced
        corpus = self.corpora["fast_mapping"]

        # TODO: exclude examples encountered during fast mapping
        idxs = np.random.choice(len(corpus), size=batch_size, replace=False)
        observations = []
        for idx in idxs:
            trial = corpus[idx]
            referent = trial["domain_positive"][0]
            assert referent["target"] is True

            observations.append(trial)

        return observations


if __name__ == "__main__":
    env = VGEnv("data/vg_processed.pkl") # VGEnv("data/vg_processed.json")

    print(env.get_batch("train", batch_size=2))
