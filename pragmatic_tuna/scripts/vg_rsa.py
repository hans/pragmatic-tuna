from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments.vg import VGEnv
from pragmatic_tuna.models.ranking_listener import BoWRankingListener
from pragmatic_tuna.models.vg_speaker import WindowedSequenceSpeakerModel


FAST_MAPPING_RELATION = "behind.r.01"


def infer_trial(candidates, listener_scores, speaker_scores, infer_with_speaker=False):
    """
    Rerank batch of candidates based on listener + speaker scores.
    Return top-scoring candidates.
    """
    scores = speaker_scores if infer_with_speaker else listener_scores
    results = [candidates_i[scores_i.argmax()]
               for candidates_i, scores_i in zip(candidates, scores)]
    return results


def run_trial(batch, listener_model, speaker_model, update=True, infer_with_speaker=False):
    utterances, candidates, lengths, n_cands = batch

    # Fetch model scores and rank for pragmatic listener inference.
    listener_scores = listener_model.score(*batch)
    speaker_scores = speaker_model.score(*batch)
    results = infer_trial(candidates, listener_scores, speaker_scores,
                          infer_with_speaker=infer_with_speaker)

    successes = [result_i == candidates_i[0]
                 for result_i, candidates_i in zip(results, candidates)]

    if update:
        # Observe.
        # TODO: joint optimization?
        l_loss = listener_model.observe(*batch)
        s_loss, avg_prob = speaker_model.observe(*batch)
    else:
        l_loss = s_loss = avg_prob = 0.0

    pct_success = np.mean(successes)
    return results, (l_loss, s_loss), avg_prob, pct_success


def sample_utterances(env, silent_batch, speaker_model):
    # Get true referents for each batch element
    candidates_batch, num_candidates = silent_batch
    referents = [batch_i[:1] for batch_i in candidates_batch]
    utterances = speaker_model.sample(referents)

    utterances = np.asarray(utterances)
    utterances_t = utterances.T[:3]
    for cand, utt in zip(referents, utterances):
        cand = cand[0]
        cand_str = " ".join(env.graph_vocab[idx] for idx in cand)
        utt_str = " ".join(env.vocab[idx] for idx in utt)
        print("%40s\t%100s" % (cand_str, utt_str))

    print()

    return utterances


def main(args):
    env = VGEnv(args.corpus_path)

    listener_model = BoWRankingListener(env,
            embedding_dim=args.embedding_dim,
            max_negative_samples=args.negative_samples)
    speaker_model = WindowedSequenceSpeakerModel(
            env, max_timesteps=env.max_timesteps,
            embedding_dim=args.embedding_dim,
            embeddings=listener_model.embeddings,
            graph_embeddings=listener_model.graph_embeddings)

    opt = tf.train.MomentumOptimizer(args.learning_rate, 0.9)
    l_global_step = tf.Variable(0, name="global_step_listener")
    listener_model.train_op = opt.minimize(listener_model.loss,
                                           global_step=l_global_step)
    s_opt = tf.train.MomentumOptimizer(args.learning_rate * 100., 0.9)
    s_global_step = tf.Variable(0, name="global_step_speaker")
    speaker_model.train_op = s_opt.minimize(speaker_model.loss,
                                          global_step=s_global_step)

    global_step = l_global_step + s_global_step
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             summary_op=None)

    with sv.managed_session() as sess:
        with sess.as_default():
            losses, pct_successes = [], []
            for i in trange(args.n_iters):
                batch = env.get_batch("train", batch_size=args.batch_size,
                                      negative_samples=args.negative_samples)
                predictions, losses_i, avg_prob, pct_success = \
                        run_trial(batch, listener_model, speaker_model, update=True)

                losses.append(losses_i)
                pct_successes.append(pct_success)

                # Try fast-mapping.
                fm_batch = env.get_batch("fast_mapping", batch_size=args.batch_size,
                                         negative_samples=args.negative_samples)
                _, _, _, pct_fm_success = \
                        run_trial(fm_batch, listener_model, speaker_model,
                                  update=False, infer_with_speaker=True)

                tqdm.write("%5f\t%5f\t%5g\t%.2f\t\t%3f"
                           % (losses_i[0], losses_i[1], avg_prob, pct_success * 100,
                              pct_fm_success * 100))

                if i == args.n_iters - 1:
                    # Debug: print utterances
                    correct, false = [], []
                    b_utt, b_cands, _, _ = batch
                    for utterance, cands, prediction in zip(b_utt.T, b_cands, predictions):
                        utterance = list(utterance)
                        try:
                            utterance = utterance[:utterance.index(env.vocab2idx[env.EOS])]
                        except ValueError: pass

                        utterance = " ".join([env.vocab[idx] for idx in utterance])

                        dest = correct if prediction == cands[0] else false
                        dest.append("%30s\t%s" % (utterance, " ".join([env.graph_vocab[idx] for idx in prediction])))

                    print("=========== Correct:")
                    print("\n".join(correct))

                    print("\n========== False:")
                    print("\n".join(false))

            # Now DREAM!
            for i in trange(args.n_iters):
                silent_batch = env.get_silent_batch(FAST_MAPPING_RELATION,
                        batch_size=args.batch_size,
                        negative_samples=0)
                b_utterances = sample_utterances(env, silent_batch, speaker_model)

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
