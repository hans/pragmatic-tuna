from collections import Counter
from copy import copy
import json
from pathlib import Path, PurePath
import subprocess
import tempfile

try: import cPickle as pickle
except: import pickle

import gym
import numpy as np

from pragmatic_tuna import glove_util


UNK = "<unk>"
EOS = "<eos>"


class VGEnv(gym.Env):

    UNK = UNK
    EOS = EOS

    def __init__(self, corpus_path, graph_embeddings_path=None,
                 embedding_dim=64, max_negative_samples=5):
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
        self.word_eos_id = self.vocab2idx[EOS]

        self.max_timesteps = max([len(trial["utterance"])
                                  for corpus in self.corpora
                                  for trial in self.corpora[corpus]])

        self.advfm_corpora = {split: [name for name in self.corpora
                                      if name.startswith("adv_fast_mapping_" + split)]
                              for split in ["train", "dev", "test"]}
        self.advfm_corpora_flat = [name for name in self.corpora
                                   if name.startswith("adv_fast_mapping")]

        # Assumes 1 positive candidate per example
        self.max_negative_samples = max_negative_samples
        self.max_candidates = max_negative_samples + 1

        self.embedding_dim = embedding_dim
        if graph_embeddings_path is None:
            graph_embeddings_path = \
                    corpus_path[:corpus_path.rindex(".")] + ".graph_embeddings.npz"
        self.graph_embeddings = self._load_graph_embeddings(graph_embeddings_path)

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

            domain_positive, domain_negative = [], []
            for subgraph in trial["domain"]:
                obj1 = subgraph["object1"]
                obj2 = subgraph["object2"]

                reln = subgraph["reln"]
                # Make sure this is a single token.
                reln = reln.replace(" ", "_")

                graph_vocab.add(obj1)
                graph_vocab.add(obj2)
                graph_vocab.add(reln)

                domain = domain_positive if subgraph["target"] else domain_negative
                domain.append((reln, obj1, obj2))

            # Don't double-count words in fast-mapping + adversarial
            # fast-mapping (adversarial are directly duped from
            # non-adversarial paired trials)
            skip = "adv_fast_mapping" in trial["type"]
            if not skip:
                for word in utterance:
                    vocab_counts[word] += 1

            corpora[trial["type"]].append({
                "utterance": utterance,
                "domain_positive": domain_positive,
                "domain_negative": domain_negative
            })


        vocab = [UNK, EOS] + list(sorted([word for word, freq in vocab_counts.items()
                                          if freq > 1]))
        vocab2idx = {w: idx for idx, w in enumerate(vocab)}
        graph_vocab = [UNK, EOS] + list(sorted(graph_vocab))
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
        lengths = np.empty(len(words_batch), dtype=np.int32)
        eos_id = self.vocab2idx[self.EOS]
        ret = []
        for i, words_i in enumerate(words_batch):
            ret_i = words_i[:]

            # Train to output at most a single EOS token.
            if len(ret_i) < self.max_timesteps:
                ret_i.append(eos_id)

            lengths[i] = len(ret_i)
            if lengths[i] < self.max_timesteps:
                ret_i.extend([eos_id] * (self.max_timesteps - lengths[i]))
            ret.append(ret_i)

        ret = np.asarray(ret).T
        return ret, lengths

    def _pad_candidates_batch(self, candidates_batch, max_candidates=None):
        max_candidates = max_candidates or self.max_candidates

        num_candidates = np.empty(len(candidates_batch), dtype=np.int32)
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

    def iter_batches(self, corpus, batch_size=64, negative_samples=5):
        """
        Iterate in batches over an ordered corpus.

        The final batch may be smaller than `batch_size`.

        Yields elements just like the return from `self.get_batch`.
        """
        assert negative_samples <= self.max_negative_samples

        corpus = self.corpora[corpus]

        offset = 0
        while offset < len(corpus):
            trials = corpus[offset:offset + batch_size]
            utterances = [trial["utterance"] for trial in trials]
            candidates = [self._extract_candidates(trial, negative_samples=negative_samples)
                          for trial in trials]

            yield self._pad_batch(utterances, candidates)
            offset += batch_size

    def get_silent_batch(self, batch_size=64, negative_samples=5, p_swap=0.75):
        """
        Return a batch for "dreaming" of grounded relations without paired
        utterances.

        Args:
            batch_size:
            negative_samples:
            p_swap: independent probability that, for each example, we swap one
                of the negative referents to be the positive referent in the
                returned batch
        """
        corpus = self.corpora["dreaming_train"]

        idxs = np.random.choice(len(corpus), size=batch_size, replace=False)
        candidates = []
        for idx in idxs:
            trial = corpus[idx]
            if p_swap > 0 and np.random.random() < p_swap:
                trial = copy(trial)
                new_positive = np.random.choice(len(trial["domain_negative"]))
                old_positive = trial["domain_positive"][0]

                trial["domain_positive"] = [trial["domain_negative"][new_positive]]
                trial["domain_negative"] = \
                        trial["domain_negative"][:new_positive] + \
                        [old_positive] + \
                        trial["domain_negative"][new_positive + 1:]

            candidates.append(self._extract_candidates(trial, negative_samples=negative_samples))

        return self._pad_candidates_batch(candidates, max_candidates=negative_samples + 1)

    def _load_graph_embeddings(self, graph_embeddings_path):
        if Path(graph_embeddings_path).exists():
            data = np.load(graph_embeddings_path)
            saved_vocab = list(data["graph_vocab"])
            if len(saved_vocab) != len(self.graph_vocab):
                raise ValueError("vocab of saved graph embeddings has different"
                                 " size than vocab loaded from corpus: %i != %i"
                                 % (len(saved_vocab), len(self.graph_vocab)))
            elif saved_vocab != self.graph_vocab:
                raise ValueError("values in vocab of saved graph embeddings "
                                 "differ from those loaded from corpus")

            embs = data["graph_embeddings"]
        else:
            embs = self._compute_graph_embeddings()
            np.savez(graph_embeddings_path,
                     graph_vocab=self.graph_vocab,
                     graph_embeddings=embs)

        assert embs.shape[1] == self.embedding_dim, embs.shape[1]
        return embs

    def _compute_graph_embeddings(self):
        """
        Pre-train embedding representations for tokens in graph vocabulary.
        """
        # We generate a corpus of collocations, then call GloVe (C++ version)
        # programmatically to create embeddings from this corpus.
        #
        # Every "utterance" in this corpus is of length 3 --- they are
        # generated from the subgraph triples. We use a GloVe window of size
        # 3, padding each utterance so that edge tokens don't mistakenly
        # track co-occurrences with nearby triples.
        with tempfile.TemporaryDirectory() as d:
            # Generate corpus.
            pad_str = " ".join([EOS] * 3)
            corpus_path = Path(d, "corpus")
            with corpus_path.open("w") as corpus_f:
                for corpus_name, corpus in self.corpora.items():
                    if corpus_name in self.advfm_corpora_flat:
                        # Skip adversarial corpora, which are duplicated from
                        # real fast-mapping data and contain lots of bogus
                        # relations.
                        continue

                    for trial in corpus:
                        subgraphs = trial["domain_positive"] + trial["domain_negative"]
                        for subgraph in subgraphs:
                            subgraph = tuple(self.graph_vocab[idx] for idx in subgraph)
                            # Rearrange so that we have (obj, reln, obj) order
                            subgraph = (subgraph[1], subgraph[0], subgraph[2])
                            subgraph_str = "%s %s %s %s %s\n" \
                                    % ((pad_str,) + subgraph + (pad_str,))
                            corpus_f.write(subgraph_str)

            return glove_util.learn_embeddings(corpus_path, self.graph_vocab2idx,
                                               self.embedding_dim)

    def utterance_to_tokens(self, u):
        """
        Convert an utterance formatted as an ID sequence to a list of tokens.
        """
        u = list(u)
        try:
            u = u[:u.index(self.word_eos_id)]
        except ValueError: pass

        return [self.vocab[idx] for idx in u]


if __name__ == "__main__":
    env = VGEnv("data/vg_processed_2_2.split_adv.dedup.json") # VGEnv("data/vg_processed.json")
