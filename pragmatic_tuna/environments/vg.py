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

    def __init__(self, corpus_path):
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

        self.max_timesteps = max([len(trial["utterance"])
                                  for corpus in self.corpora
                                  for trial in self.corpora[corpus]])

    def _process_corpus(self, corpus_path):
        with open(corpus_path, "r") as corpus_f:
            corpus_data = json.load(corpus_f)

        corpora = {}
        vocab, graph_vocab = set(), set()
        for trial in corpus_data:
            if trial["type"] not in corpora:
                corpora[trial["type"]] = []

            for word in trial["utterance"].split():
                vocab.add(word)

            domain_positive, domain_negative = [], []
            for subgraph in trial["domain"]:
                obj1 = subgraph["object1"][:subgraph["object1"].index(".")]
                obj2 = subgraph["object2"][:subgraph["object2"].index(".")]
                reln = subgraph["reln"][:subgraph["reln"].index(".")]
                graph_vocab.add(obj1)
                graph_vocab.add(obj2)
                graph_vocab.add(reln)

                domain = domain_positive if subgraph["target"] else domain_negative
                domain.append((reln, obj1, obj2))

            corpora[trial["type"]].append({
                "utterance": trial["utterance"],
                "domain_positive": domain_positive,
                "domain_negative": domain_negative
            })

        vocab = [UNK, EOS] + list(sorted(vocab))
        vocab2idx = {w: idx for idx, w in enumerate(vocab)}
        graph_vocab = [EOS] + list(sorted(graph_vocab))
        graph_vocab2idx = {w: idx for idx, w in enumerate(graph_vocab)}

        # Now reprocess trials, replacing strings with IDs.
        for corpus_name, corpus in corpora.items():
            for trial in corpus:
                trial["utterance"] = [vocab2idx[word] for word in trial["utterance"].split()]
                trial["domain_positive"] = [tuple([graph_vocab2idx[x] for x in subgraph])
                                            for subgraph in trial["domain_positive"]]
                trial["domain_negative"] = [tuple([graph_vocab2idx[x] for x in subgraph])
                                            for subgraph in trial["domain_negative"]]

        return corpora, vocab, graph_vocab

    def get_batch(self, corpus, batch_size=64, negative_samples=5):
        corpus = self.corpora[corpus]
        assert len(corpus) >= batch_size
        idxs = np.random.choice(len(corpus), size=batch_size, replace=False)

        utterances = []
        positive_candidates, negative_candidates = [], []
        for idx in idxs:
            trial = corpus[idx]
            utterances.append(trial["utterance"])
            positive_candidates.append(trial["domain_positive"][0])

            neg_samples = min(negative_samples, len(trial["domain_negative"]))
            neg_idxs = np.random.choice(len(trial["domain_negative"]), size=neg_samples, replace=False)
            negative_candidates.append([trial["domain_negative"][neg_idx] for neg_idx in neg_idxs])

        return utterances, positive_candidates, negative_candidates


if __name__ == "__main__":
    env = VGEnv("data/vg_processed.pkl") # VGEnv("data/vg_processed.json")

    print(env.get_batch("train", batch_size=2))
