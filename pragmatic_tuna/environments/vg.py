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
        self.graph_vocab2idx = {w: idx for idx, w in enumerate(self.graph_vocab)}
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

    def _extract_candidates(self, trial, negative_samples=5):
        """
        Extract candidate referents from a trial using negative sampling.
        """
        positive = trial["domain_positive"]
        negative = trial["domain_negative"]

        # For now: only work with single positive referent
        assert len(positive) == 1, len(positive)
        candidates = positive[:]

        neg_samples = min(negative_samples, len(negative))
        if neg_samples > 0:
            neg_idxs = np.random.choice(len(negative), size=neg_samples,
                                        replace=False)
            candidates.extend([negative[neg_idx] for neg_idx in neg_idxs])
        else:
            # TODO how to handle this..?
            eos_id = self.graph_vocab2idx[EOS]
            candidates.append((eos_id, eos_id, eos_id))

        return candidates

    def _pad_words_batch(self, words_batch):
        lengths = np.empty(len(words_batch))
        eos_id = self.vocab2idx[self.EOS]
        ret = []
        for i, words_i in enumerate(words_batch):
            lengths[i] = len(words_i)
            ret_i = words_i[:]
            if lengths[i] < self.max_timesteps:
                ret_i.extend([eos_id] * (self.max_timesteps - lengths[i]))
            ret.append(ret_i)

        ret = np.asarray(ret).T
        return ret, lengths

    def _pad_candidates_batch(self, candidates_batch, max_candidates=None):
        max_candidates = max_candidates or self.max_candidates

        num_candidates = np.empty(len(candidates_batch))
        candidates_batch_ret = []
        for i, candidates_i in enumerate(candidates_batch):
            num_candidates[i] = len(candidates_i)
            ret_i = candidates_i[:]

            if num_candidates[i] < max_candidates:
                pad_length = max_candidates - num_candidates[i]
                ret_i.extend([(0, 0, 0)] * (pad_length))
            candidates_batch_ret.append(ret_i)

        return candidates_batch_ret, num_candidates

    def _pad_batch(self, words_batch, candidates_batch):
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
        words_batch_ret, lengths = \
                self._pad_words_batch(words_batch)

        # Pad candidates.
        candidates_batch_ret, num_candidates = \
                self._pad_candidates_batch(candidates_batch)

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

            utterances.append(trial["utterance"])
            candidates.append(self._extract_candidates(trial, negative_samples=negative_samples))

        return self._pad_batch(utterances, candidates)

    def get_silent_batch(self, relation, batch_size=64, negative_samples=5):
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
        reln_id = self.graph_vocab2idx[relation]

        # TODO: exclude examples encountered during fast mapping
        idxs = np.random.choice(len(corpus), size=batch_size, replace=False)
        candidates = []
        for idx in idxs:
            trial = corpus[idx]
            referent = trial["domain_positive"][0]
            assert referent[0] == reln_id, " ".join(self.graph_vocab[g_idx] for g_idx in referent)

            candidates.append(self._extract_candidates(trial, negative_samples=negative_samples))

        return self._pad_candidates_batch(candidates, max_candidates=negative_samples + 1)


if __name__ == "__main__":
    env = VGEnv("data/vg_processed.pkl") # VGEnv("data/vg_processed.json")

    print(env.get_batch("train", batch_size=2))
