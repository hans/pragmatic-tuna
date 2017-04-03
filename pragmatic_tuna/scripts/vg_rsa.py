from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments.vg import VGEnv
from pragmatic_tuna.models.ranking_listener import BoWRankingListener
from pragmatic_tuna.models.vg_speaker import WindowedSequenceSpeakerModel


def infer_trial(candidates, listener_scores, speaker_scores):
    """
    Rerank batch of candidates based on listener + speaker scores.
    Return top-scoring candidates.
    """
    results = [candidates_i[l_scores_i.argmax()]
               for candidates_i, l_scores_i in zip(candidates, listener_scores)]
    return results


def run_trial(batch, listener_model, speaker_model):
    utterances, candidates = batch

    # Fetch model scores and rank for pragmatic listener inference.
    listener_scores = listener_model.score(utterances, candidates)
    speaker_scores = None#speaker_scores = speaker_model.score_batch(utterances, candidates)
    results = infer_trial(candidates, listener_scores, speaker_scores)

    successes = [result_i == candidates_i[0]
                 for result_i, candidates_i in zip(results, candidates)]

    # Observe.
    loss = listener_model.observe(utterances, candidates)

    pct_success = np.mean(successes)
    tqdm.write("%5f\t%.2f" % (loss, pct_success * 100))

    return results, loss, pct_success


def main(args):
    env = VGEnv(args.corpus_path)

    listener_model = BoWRankingListener(env,
            embedding_dim=args.embedding_dim,
            max_negative_samples=args.negative_samples)
    speaker_model = WindowedSequenceSpeakerModel(
            env, max_timesteps=3,
            embedding_dim=args.embedding_dim,
            embeddings=listener_model.embeddings,
            graph_embeddings=listener_model.graph_embeddings)

    global_step = tf.Variable(0, name="global_step")
    opt = tf.train.MomentumOptimizer(args.learning_rate, 0.9)
    train_op = opt.minimize(listener_model.loss, global_step=global_step)
    listener_model.train_op = train_op

    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             summary_op=None)

    with sv.managed_session() as sess:
        with sess.as_default():
            losses, pct_successes = [], []
            for i in trange(args.n_iters):
                batch = env.get_batch("train", batch_size=args.batch_size,
                                      negative_samples=args.negative_samples)
                predictions, loss, pct_success = \
                        run_trial(batch, listener_model, speaker_model)

                losses.append(loss)
                pct_successes.append(pct_success)

                if i == args.n_iters - 1:
                    # Debug: print utterances
                    correct, false = [], []
                    b_utt, b_cands = batch
                    for utterance, cands, prediction in zip(b_utt, b_cands, predictions):
                        utterance = " ".join([env.vocab[idx] for idx in utterance])

                        dest = correct if prediction == cands[0] else false
                        dest.append("%30s\t%s" % (utterance, " ".join([env.graph_vocab[idx] for idx in prediction])))

                    print("=========== Correct:")
                    print("\n".join(correct))

                    print("\n========== False:")
                    print("\n".join(false))

            sv.request_stop()


if __name__ == '__main__':
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/vg")
    p.add_argument("--corpus_path")

    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--negative_samples", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=0.001)

    p.add_argument("--embedding_dim", type=int, default=64)

    main(p.parse_args())
