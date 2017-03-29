from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments.vg import VGEnv
from pragmatic_tuna.models.ranking_listener import BoWRankingListener


def infer_trial(batch, listener_model, speaker_model):
    utterances, pos_candidates, neg_candidates = batch

    pos_scores = listener_model.rank(utterances, pos_candidates)
    neg_scores = listener_model.rank(utterances, neg_candidates)
    results = []
    for example in zip(pos_candidates, pos_scores, neg_candidates, neg_scores):
        pos_cand_i, pos_scores_i, neg_cand_i, neg_scores_i = example

        pos_argmax = np.argmax(pos_scores_i)
        neg_argmax = np.argmax(neg_scores_i)
        if pos_scores_i[pos_argmax] > neg_scores_i[neg_argmax]:
            results.append(pos_cand_i[pos_argmax])
            success = True
        else:
            results.append(neg_cand_i[neg_argmax])
            success = False

    return results

def main(args):
    env = VGEnv(args.corpus_path)

    listener_model = BoWRankingListener(env, max_negative_samples=args.negative_samples)
    speaker_model = None

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        for i in trange(args.n_iters):
            batch = env.get_batch("train", batch_size=args.batch_size,
                                  negative_samples=args.negative_samples)
            predictions = infer_trial(batch, listener_model, speaker_model)


if __name__ == '__main__':
    p = ArgumentParser()

    p.add_argument("--corpus_path")

    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--negative_samples", type=int, default=5)

    main(p.parse_args())
