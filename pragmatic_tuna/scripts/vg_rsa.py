from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments.vg import VGEnv
from pragmatic_tuna.models.ranking_listener import BoWRankingListener


LEARNING_RATE = 0.005
def run_trial(batch, listener_model, speaker_model):
    utterances, pos_candidates, neg_candidates = batch

    # Run inference.
    pos_scores = listener_model.rank(utterances, pos_candidates)
    neg_scores = listener_model.rank(utterances, neg_candidates)
    results, successes = [], []
    for example in zip(pos_candidates, pos_scores, neg_candidates, neg_scores):
        pos_cand_i, pos_scores_i, neg_cand_i, neg_scores_i = example

        pos_argmax = np.argmax(pos_scores_i)
        neg_argmax = np.argmax(neg_scores_i)
        if pos_scores_i[pos_argmax] > neg_scores_i[neg_argmax]:
            result = pos_cand_i[pos_argmax]
            success = True
        else:
            result = neg_cand_i[neg_argmax]
            success = False

        results.append(result)
        successes.append(success)

    # Observe.
    global LEARNING_RATE
    loss = listener_model.observe(utterances, pos_candidates, neg_candidates, learning_rate=LEARNING_RATE)
    if loss < 2.0 and LEARNING_RATE == 0.005:
        pass#tqdm.write("========== Cutting learning rate.")
        #LEARNING_RATE = 0.0005

    pct_success = np.mean(successes)
    tqdm.write("%5f\t%.2f" % (loss, pct_success * 100))

    return results, loss, pct_success

def main(args):
    env = VGEnv(args.corpus_path)

    listener_model = BoWRankingListener(env, max_negative_samples=args.negative_samples)
    speaker_model = None

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        losses, pct_successes = [], []
        for i in trange(args.n_iters):
            batch = env.get_batch("train", batch_size=args.batch_size,
                                  negative_samples=args.negative_samples)
            predictions, loss, pct_success = run_trial(batch, listener_model, speaker_model)

            losses.append(loss)
            pct_successes.append(pct_success)

            if i == args.n_iters - 1:
                # Debug: print utterances
                correct, false = [], []
                b_utt, b_pos, b_neg = batch
                for utterance, pos_cands, neg_cands, prediction in zip(b_utt, b_pos, b_neg, predictions):
                    utterance = " ".join([env.vocab[idx] for idx in utterance])

                    dest = correct if prediction == pos_cands[0] else false
                    dest.append("%30s\t%s" % (utterance, " ".join([env.graph_vocab[idx] for idx in prediction])))

                print("=========== Correct:")
                print("\n".join(correct))

                print("\n========== False:")
                print("\n".join(false))


if __name__ == '__main__':
    p = ArgumentParser()

    p.add_argument("--corpus_path")

    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--negative_samples", type=int, default=5)

    main(p.parse_args())
