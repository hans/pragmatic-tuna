from argparse import ArgumentParser
import os.path
from pprint import pprint

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments.vg import VGEnv
from pragmatic_tuna.models.ranking_listener import BoWRankingListener
from pragmatic_tuna.models.vg_speaker import WindowedSequenceSpeakerModel
from pragmatic_tuna.util import make_summary


FAST_MAPPING_RELATION = "behind"


def infer_trial(candidates, listener_scores, speaker_scores, infer_with_speaker=False):
    """
    Rerank batch of candidates based on listener + speaker scores.
    Return top-scoring candidates.
    """
    scores = speaker_scores if infer_with_speaker else listener_scores
    results = [candidates_i[scores_i.argmax()]
               for candidates_i, scores_i in zip(candidates, scores)]
    return results


def run_trial(batch, listener_model, speaker_model, update_listener=True,
              update_speaker=True, infer_with_speaker=False):
    utterances, candidates, lengths, n_cands = batch

    # Fetch model scores and rank for pragmatic listener inference.
    listener_scores = listener_model.score(*batch)
    speaker_scores = speaker_model.score(*batch)
    results = infer_trial(candidates, listener_scores, speaker_scores,
                          infer_with_speaker=infer_with_speaker)

    successes = [result_i == candidates_i[0]
                 for result_i, candidates_i in zip(results, candidates)]

    # TODO: joint optimization?
    l_loss = s_loss = 0.0
    if update_listener:
        l_loss = listener_model.observe(*batch)
    if update_speaker:
        s_loss = speaker_model.observe(*batch)

    pct_success = np.mean(successes)
    return results, (l_loss, s_loss), pct_success


def run_train_phase(sv, env, listener_model, speaker_model, args):
    losses, pct_successes = [], []
    for i in trange(args.n_iters):
        batch = env.get_batch("pre_train_train", batch_size=args.batch_size,
                              negative_samples=args.negative_samples)
        predictions, losses_i, pct_success = \
                run_trial(batch, listener_model, speaker_model)

        losses.append(losses_i)
        pct_successes.append(pct_success)

        # Try fast-mapping (adversarial batch!).
        fm_batch = env.get_batch("adv_fast_mapping", batch_size=args.batch_size,
                                 negative_samples=args.negative_samples)
        _, _, pct_fm_success = \
                run_trial(fm_batch, listener_model, speaker_model,
                          update_listener=False, update_speaker=False,
                          infer_with_speaker=True)

        tqdm.write("%5f\t%5f\t%.2f\t\t%3f"
                    % (losses_i[0], losses_i[1], pct_success * 100,
                       pct_fm_success * 100))

        if i % args.summary_interval == 0:
            summary = make_summary({
                "listener_loss": losses_i[0],
                "speaker_loss": losses_i[1],
                "listener_success": pct_success,
                "speaker_fm_success": pct_fm_success,
            })
            sv.summary_computed(tf.get_default_session(), summary)

        if i % args.eval_interval == 0 or i == args.n_iters - 1:
            tqdm.write("====================== DEV EVAL AT %i" % i)
            do_eval(sv, env, listener_model, speaker_model)


def do_eval(sv, env, listener_model, speaker_model, args, batch=None,
            corpus="pre_train_dev"):
    d_batch = batch
    if d_batch is None:
        d_batch = env.get_batch(corpus, batch_size=args.batch_size,
                                negative_samples=args.negative_samples)
    d_utt, d_cands, d_lengths, d_n_cands = d_batch

    d_predictions, d_losses, d_pct_success = \
            run_trial(d_batch, listener_model, speaker_model,
                      update_listener=False, update_speaker=False)

    # Test: draw some samples for this new input
    silent_batch = (d_cands, d_n_cands)
    s_utt, s_lengths = sample_utterances(env, silent_batch, speaker_model)

    # Debug: print utterances
    correct, false = [], []
    for utterance, cands, prediction, sample in zip(d_utt.T, d_cands,
                                                    d_predictions, s_utt.T):
        utterance = " ".join(env.utterance_to_tokens(utterance))
        sample = " ".join(env.utterance_to_tokens(sample))

        dest = correct if prediction == cands[0] else false
        dest.append("%40s\t%40s\t%s" %
                    (utterance, sample,
                        " ".join([env.graph_vocab[idx] for idx in prediction])))

    tqdm.write("=========== Correct:")
    tqdm.write("\n".join(correct))

    tqdm.write("\n========== False:")
    tqdm.write("\n".join(false))


def run_fm_phase(sv, env, listener_model, speaker_model, args,
                 corpus="fast_mapping_train"):
    """
    Run the "fast mapping" (== zero-shot inference) phase.
    """
    # Number of examples to use for learning.
    # TODO: make parameter
    N = 1#args.batch_size

    batch = env.get_batch(corpus, batch_size=N,
                          negative_samples=args.negative_samples)
    # NB: 100 is suitable for N=64
    for i in range(10):#100): # TODO
        predictions, losses_i, pct_success = \
                run_trial(batch, listener_model, speaker_model,
                          update_speaker=True, update_listener=False,
                          infer_with_speaker=True)
        tqdm.write("%5f\t%5f\tS:%.2f" % (losses_i[0], losses_i[1], pct_success * 100))

    do_eval(sv, env, listener_model, speaker_model, args, batch=batch)


def sample_utterances(env, silent_batch, speaker_model):
    """
    Sample utterances from the speaker referring to the true referent for each
    example in the given "silent" batch.

    Returns:
        utterances: `max_timesteps * batch_size` padded batch
        lengths:
    """
    # Get true referents for each batch element
    candidates_batch, num_candidates = silent_batch
    referents = [batch_i[:1] for batch_i in candidates_batch]
    utterances = speaker_model.sample(referents, argmax=True)

    # Compute lengths
    batch_size = len(candidates_batch)
    max_length = env.max_timesteps
    lengths = np.ones((batch_size,), dtype=np.int32) * max_length
    eos_positions = np.array(np.where(utterances == env.word_eos_id)).T
    for example, eos_idx in eos_positions:
        lengths[example] = max(lengths[example], eos_idx)

    # Mask to make sure these examples make sense
    mask = np.tile(np.arange(max_length).reshape((-1, 1)), (1, batch_size))
    mask = lengths.reshape((1, -1)) > mask
    utterances *= mask

    return utterances, lengths


def synthesize_dream_batch(env, speaker_model, batch_size,
                           dream_ratio=0.5, negative_samples=5):
    """
    Synthesize a "dreaming" training batch where some of the examples are
    sampled from the speaker model itself.
    """

    dreamed_size = int(batch_size * dream_ratio)
    real_size = batch_size - dreamed_size

    # Pull a silent batch and draw utterances.
    silent_batch = env.get_silent_batch(FAST_MAPPING_RELATION,
            batch_size=dreamed_size,
            negative_samples=negative_samples)
    model_utterances, model_lengths = sample_utterances(env, silent_batch,
                                                        speaker_model)
    silent_candidates, silent_num_candidates = silent_batch

    real_batch = env.get_batch("pre_train_train", batch_size=real_size,
                               negative_samples=negative_samples)

    utterances = np.concatenate((model_utterances, real_batch[0]), axis=1)
    candidates = silent_candidates + real_batch[1]
    lengths = np.concatenate((model_lengths, real_batch[2]))
    num_candidates = np.concatenate((silent_num_candidates, real_batch[3]))

    synthesized_batch = (utterances, candidates, lengths, num_candidates)
    return synthesized_batch


def run_dream_phase(sv, env, listener_model, speaker_model, args):
    n_iters = 100
    for i in trange(n_iters):
        batch = synthesize_dream_batch(env, speaker_model, args.batch_size,
                                       dream_ratio=0.75, # TODO
                                       negative_samples=args.negative_samples)

        if i % 10 == 0:
            # DEV: debug print a few strings
            # Assumes that silent / sampled candidates come first from
            # synthesize_dream_batch (they do)
            utterances = np.asarray(batch[0]).T[:5]
            candidates = batch[1]
            for cands, utt in zip(candidates, utterances):
                cand = cands[0]
                cand_str = " ".join(env.graph_vocab[idx] for idx in cand)
                utt_str = " ".join(env.utterance_to_tokens(utt))
                print("%40s\t%100s" % (cand_str, utt_str))
            print()

        predictions, losses_i, pct_success = \
                run_trial(batch, listener_model, speaker_model)

        tqdm.write("%5f\t%5f\tL:%.2f"
                   % (losses_i[0], losses_i[1], pct_success * 100))

        if i % args.eval_interval == 0 or i == n_iters - 1:
            print("======== FM DEV EVAL")
            do_eval(sv, env, listener_model, speaker_model, args,
                    corpus="fast_mapping_dev")


def main(args):
    env = VGEnv(args.corpus_path, embedding_dim=args.embedding_dim)
    graph_embeddings = tf.Variable(env.graph_embeddings, name="graph_embeddings",
                                   dtype=tf.float32, trainable=False)

    listener_model = BoWRankingListener(env,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.listener_hidden_dim,
            graph_embeddings=graph_embeddings,
            max_negative_samples=args.negative_samples)
    speaker_model = WindowedSequenceSpeakerModel(
            env, max_timesteps=env.max_timesteps,
            embedding_dim=args.embedding_dim,
            embeddings=listener_model.embeddings,
            graph_embeddings=graph_embeddings,
            hidden_dim=args.speaker_hidden_dim,
            dropout_keep_prob=args.dropout_keep_prob)

    if args.optimizer == "momentum":
        opt_f = lambda lr: tf.train.MomentumOptimizer(lr, 0.9)
    elif args.optimizer == "adagrad":
        opt_f = lambda lr: tf.train.AdagradOptimizer(lr)

    l_opt = opt_f(args.listener_learning_rate)
    l_global_step = tf.Variable(0, name="global_step_listener")
    listener_model.train_op = l_opt.minimize(listener_model.loss,
                                             global_step=l_global_step)

    speaker_lr = args.listener_learning_rate * args.speaker_lr_factor
    s_opt = opt_f(speaker_lr)
    s_global_step = tf.Variable(0, name="global_step_speaker")
    speaker_model.train_op = s_opt.minimize(speaker_model.loss,
                                            global_step=s_global_step)

    global_step = l_global_step + s_global_step
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             summary_op=None)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with sv.managed_session(config=config) as sess:
        # Dump params.
        params_path = os.path.join(args.logdir, "params")
        with open(params_path, "w") as params_f:
            pprint(vars(args), params_f)

        with sess.as_default():
            # print("============== TRAINING")
            # run_train_phase(sv, env, listener_model, speaker_model, args)

            print("============== FAST MAPPING")
            run_fm_phase(sv, env, listener_model, speaker_model, args)

            print("============== DREAMING")
            run_dream_phase(sv, env, listener_model, speaker_model, args)

            sv.request_stop()


if __name__ == '__main__':
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/vg")
    p.add_argument("--corpus_path")

    p.add_argument("--summary_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)

    p.add_argument("--optimizer", choices=["momentum", "adagrad"],
                   default="momentum")

    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--negative_samples", type=int, default=5)
    p.add_argument("--listener_learning_rate", type=float, default=0.001)
    p.add_argument("--speaker_lr_factor", type=float, default=100)

    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--listener_hidden_dim", type=int, default=256)
    p.add_argument("--speaker_hidden_dim", type=int, default=256)
    p.add_argument("--dropout_keep_prob", type=float, default=0.8)

    main(p.parse_args())
