"""
Utility module which supports shelling out to GloVe to learn embeddings for a
corpus.
"""

from pathlib import Path, PurePath
import subprocess

import numpy as np


GLOVE_PATH = Path(Path(__file__).absolute().parent.parent,
                  "tools", "glove", "build")
assert GLOVE_PATH.exists(), GLOVE_PATH

VOCAB_FILE = "vocab.txt"
COOCCURRENCE_FILE = "cooccurrence.bin"
COOCCURRENCE_SHUF_FILE = "cooccurrence.shuf.bin"
VECTORS_FILE = "vectors"

MEMORY = "16.0"
NUM_THREADS = "8"


def learn_embeddings(corpus_path, vocab2idx, embedding_dim, workdir=None):
    if workdir is None:
        workdir = corpus_path.parent

    do_vocab_count(corpus_path, workdir)
    do_cooccur(corpus_path, workdir)
    do_shuffle(workdir)
    do_glove(workdir, embedding_dim)

    vectors_path = Path(workdir, VECTORS_FILE + ".txt")
    vocab_vectors = []
    with vectors_path.open() as vectors_f:
        for line in vectors_f:
            fields = line.strip().split()
            word = fields[0]
            vector = [float(x) for x in fields[1:]]
            vocab_vectors.append((word, vector))

    # Resort according to provided vocabulary.
    vocab_vectors = list(sorted(vocab_vectors, key=lambda x: vocab2idx[x[0]]))
    embeddings = np.array([vector for word, vector in vocab_vectors])
    return embeddings


def do_vocab_count(corpus_path, workdir):
    vocab_bin = Path(GLOVE_PATH, "vocab_count")
    vocab_out = Path(workdir, VOCAB_FILE)
    args = [str(vocab_bin),
            "-min-count 1",
            "-verbose 2"]
    with corpus_path.open() as corpus_f, vocab_out.open("w") as vocab_f:
        subprocess.run(args, stdin=corpus_f, stdout=vocab_f)


def do_cooccur(corpus_path, workdir):
    cooccur_bin = Path(GLOVE_PATH, "cooccur")
    vocab_path = Path(workdir, VOCAB_FILE)
    cooccur_out = Path(workdir, COOCCURRENCE_FILE)
    args = [str(cooccur_bin),
            "-memory", MEMORY,
            "-vocab-file", str(vocab_path),
            "-verbose", "2",
            "-window-size", "3"]
    with corpus_path.open() as corpus_f, cooccur_out.open("w") as cooccur_f:
        subprocess.run(args, stdin=corpus_f, stdout=cooccur_f)


def do_shuffle(workdir):
    shuffle_bin = Path(GLOVE_PATH, "shuffle")
    cooccur_path = Path(workdir, COOCCURRENCE_FILE)
    shuffle_out = Path(workdir, COOCCURRENCE_SHUF_FILE)
    args = [str(shuffle_bin),
            "-memory", MEMORY,
            "-verbose", "2"]
    with cooccur_path.open() as cooccur_f, shuffle_out.open("w") as shuffle_f:
        subprocess.run(args, stdin=cooccur_f, stdout=shuffle_f)


def do_glove(workdir, vector_size):
    glove_bin = Path(GLOVE_PATH, "glove")
    shuffle_path = Path(workdir, COOCCURRENCE_SHUF_FILE)
    vocab_path = Path(workdir, VOCAB_FILE)
    vectors_out = Path(workdir, VECTORS_FILE)
    args = [str(glove_bin),
            "-save-file", str(vectors_out),
            "-threads", NUM_THREADS,
            "-input-file", str(shuffle_path),
            "-x-max", "10", # TODO
            "-iter", "30", # TODO
            "-vector-size", str(vector_size),
            "-binary", "0",
            "-vocab-file", str(vocab_path),
            "-verbose", "2"]
    subprocess.run(args)
