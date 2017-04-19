from argparse import ArgumentParser
import os.path
from pprint import pprint
import sys

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments.vg import VGEnv
from pragmatic_tuna.models.ranking_listener import BoWRankingListener
from pragmatic_tuna.models.vg_speaker import WindowedSequenceSpeakerModel
from pragmatic_tuna.util import make_summary


# DEV
l_norm = s_norm = None


def infer_trial(candidates, listener_scores, speaker_scores,
                infer_with_speaker=False):
    """
    Rerank batch of candidates based on listener + speaker scores.

    Returns:
        scores: Final ranks used to arbitrate. batch_size list of candidate
            score lists
        result: Selections. batch_size list of candidate indices
    """
    scores = speaker_scores if infer_with_speaker else listener_scores
    results = [candidates_i[scores_i.argmax()]
               for candidates_i, scores_i in zip(candidates, scores)]
    return scores, results


def run_trial(batch, listener_model, speaker_model, update_listener=True,
              update_speaker=True, infer_with_speaker=False, verbose=False):
    utterances, candidates, lengths, n_cands = batch

    # Fetch model scores and rank for pragmatic listener inference.
    listener_scores = listener_model.score(*batch)
    speaker_scores = speaker_model.score(*batch)
    scores, results = infer_trial(candidates, listener_scores, speaker_scores,
                                  infer_with_speaker=infer_with_speaker)

    if verbose:
        print(scores)

    successes = [result_i == candidates_i[0]
                     and scores_i.min() != scores_i.max()
                 for result_i, candidates_i, scores_i
                 in zip(results, candidates, scores)]

    # TODO: joint optimization?
    l_loss = s_loss = 0.0
    l_gn = s_gn = 0.0
    if update_listener:
        l_loss, l_gn = listener_model.observe(*batch, norm_op=l_norm)
    if update_speaker:
        s_loss, s_gn = speaker_model.observe(*batch, norm_op=s_norm)

    pct_success = np.mean(successes)
    losses = (l_loss, s_loss)
    norms = (l_gn, s_gn)
    return results, losses, norms, pct_success


def run_train_phase(sv, env, listener_model, speaker_model, args):
    for i in trange(args.n_iters):
        batch = env.get_batch("pre_train_train", batch_size=args.batch_size,
                              negative_samples=args.negative_samples)
        predictions, losses, _, pct_success = \
                run_trial(batch, listener_model, speaker_model)

        tqdm.write("%5f\t%5f\t%.2f" % (*losses, pct_success * 100))

        if i % args.summary_interval == 0:
            eval_args = (env, listener_model, speaker_model, args)

            summary = make_summary({
                "listener_loss": losses[0],
                "speaker_loss": losses[1],
                "listener_train_success": pct_success,
                "speaker_advfm_success": eval_success("adv_fast_mapping_dev",
                    *eval_args, n_batches=5, infer_with_speaker=True),
                "listener_dev_success": eval_success("pre_train_dev",
                    *eval_args, n_batches=5),
            })
            sv.summary_computed(tf.get_default_session(), summary)

        if i % args.eval_interval == 0 or i == args.n_iters - 1:
            tqdm.write("====================== DEV EVAL AT %i" % i)
            do_eval(sv, env, listener_model, speaker_model, args)


def eval_success(corpus, env, l_model, s_model, args, n_batches=1,
                 infer_with_speaker=False):
    """
    Quick evaluation: fetch a few batches and compute average success.
    """
    pct_successes = []

    for _ in range(n_batches):
        batch = env.get_batch(corpus, batch_size=args.batch_size,
                            negative_samples=args.negative_samples)
        _, _, _, pct_success = \
                run_trial(batch, l_model, s_model,
                        update_listener=False, update_speaker=False,
                        infer_with_speaker=infer_with_speaker)

        pct_successes.append(pct_success)

    return np.mean(pct_successes)


def do_eval(sv, env, listener_model, speaker_model, args, batch=None,
            corpus="pre_train_dev", verbose=False):
    d_batch = batch
    if d_batch is None:
        d_batch = env.get_batch(corpus, batch_size=args.batch_size,
                                negative_samples=args.negative_samples)
    d_utt, d_cands, d_lengths, d_n_cands = d_batch

    d_predictions, _, _, d_pct_success = \
            run_trial(d_batch, listener_model, speaker_model,
                      update_listener=False, update_speaker=False,
                      verbose=verbose)

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
        dest.append("%40s\t%60s\t%40s\t%40s" %
                    (utterance, sample,
                     " ".join([env.graph_vocab[idx] for idx in cands[0]]),
                     " ".join([env.graph_vocab[idx] for idx in prediction])))

    tqdm.write("=========== Correct:")
    tqdm.write("\n".join(correct))

    tqdm.write("\n========== False:")
    tqdm.write("\n".join(false))


def run_fm_phase(sv, env, listener_model, speaker_model, args,
                 k=None, corpus="fast_mapping_train"):
    """
    Run the "fast mapping" (== zero-shot inference) phase.

    Arguments:
        sv: Supervisor
        env:
        listener_model:
        speaker_model:
        args:
        k: Number of labeled samples to use for learning
        corpus: Corpus from which to draw fast-mapping trials
    """
    if k is None:
        k = args.batch_size

    batch = env.get_batch(corpus, batch_size=k,
                          negative_samples=args.negative_samples)

    speaker_loss = np.inf
    i = 0
    while speaker_loss > args.fast_mapping_threshold:
        predictions, losses, norms, pct_success = \
                run_trial(batch, listener_model, speaker_model,
                          update_speaker=True, update_listener=False,
                          infer_with_speaker=True)
        listener_loss, speaker_loss = losses
        listener_norm, speaker_norm = norms

        tqdm.write("%5f\t%5f\tS:%.2f\t\t%5f\t%5f" %
                   (listener_loss, speaker_loss, pct_success * 100,
                    listener_norm, speaker_norm))

        i += 1
        if i == 2000:
            print("HALT: Fast mapping did not converge after 2000 iterations.")
            print("Dying.")
            sys.exit(1)

    do_eval(sv, env, listener_model, speaker_model, args, batch=batch)
    return batch


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
    utterances = np.asarray(speaker_model.sample(referents, argmax=True))

    # Compute lengths
    batch_size = len(candidates_batch)
    max_length = env.max_timesteps
    lengths = np.ones((batch_size,), dtype=np.int32) * max_length
    eos_positions = np.array(np.where(utterances == env.word_eos_id)).T
    for example, eos_idx in eos_positions:
        lengths[example] = max(lengths[example], eos_idx)

    # # DEV: keep the lengths, but randomly replace all non-EOS tokens with other
    # # tokens
    # valid_tokens = [x for x in range(env.vocab_size)
    #                 if x not in [env.word_eos_id, env.word_unk_id]]
    # utterances = np.random.choice(np.array(valid_tokens), replace=True,
    #                               size=utterances.shape)
    # # END DEV

    # Mask to make sure these examples make sense
    mask = np.tile(np.arange(max_length).reshape((-1, 1)), (1, batch_size))
    mask = lengths.reshape((1, -1)) > mask
    utterances *= mask

    return utterances, lengths


def sample_batch(batch, n):
    utterances, candidates, lengths, n_candidates = batch
    idxs = np.random.choice(len(lengths), size=n, replace=False)
    return (utterances[:, idxs],
            [candidates[idx] for idx in idxs],
            lengths[idxs], n_candidates[idxs])


def concat_batches(batch1, batch2):
    return (np.concatenate((batch1[0], batch2[0]), axis=1), # utterances
            batch1[1] + batch2[1], # candidates
            np.concatenate((batch1[2], batch2[2])), # lengths
            np.concatenate((batch1[3], batch2[3])), # n_candidates
            )


def synthesize_dream_batch(env, speaker_model, batch_size,
                           dream_ratio=0.5, negative_samples=5,
                           p_silent_swap=0.75):
    """
    Synthesize a "dreaming" training batch where some of the examples are
    sampled from the speaker model itself.
    """

    dreamed_size = int(batch_size * dream_ratio)
    real_size = batch_size - dreamed_size

    # Pull a silent batch and draw utterances.
    silent_batch = env.get_silent_batch(batch_size=dreamed_size,
                                        negative_samples=negative_samples,
                                        p_swap=p_silent_swap)
    model_utterances, model_lengths = sample_utterances(env, silent_batch,
                                                        speaker_model)
    dreamed_batch = (model_utterances, silent_batch[0],
                     model_lengths, silent_batch[1])

    real_batch = env.get_batch("pre_train_train", batch_size=real_size,
                               negative_samples=negative_samples)

    synthetic_batch = concat_batches(dreamed_batch, real_batch)
    return synthetic_batch


def run_dream_phase(sv, env, listener_model, speaker_model, fm_batch, args):
    verbose_interval = 50
    for i in trange(args.n_dream_iters):
        ####### DATA PREP

        # Synthetic dream batch.
        batch = synthesize_dream_batch(env, speaker_model, args.batch_size,
                                       dream_ratio=args.dream_ratio,
                                       negative_samples=args.negative_samples,
                                       p_silent_swap=args.dream_p_swap)

        # "Stabilizer" batch for the speaker.
        if fm_batch is None:
            stabilizer_batch = env.get_batch("pre_train_train", batch_size=args.batch_size,
                                             negative_samples=args.negative_samples)
        else:
            n_fm = min(len(fm_batch[2]),
                       int(args.dream_stabilizer_factor * args.batch_size))
            stabilizer_batch = concat_batches(
                    sample_batch(fm_batch, n_fm),
                    env.get_batch("pre_train_train",
                                  batch_size=args.batch_size - n_fm,
                                  negative_samples=args.negative_samples))

        ####### EVALUATION

        if i % args.eval_interval == 0 or i == args.n_dream_iters - 1:
            print("========= eval: synth training batch")
            do_eval(sv, env, listener_model, speaker_model, args,
                    batch=batch)

            for corpus in ["fast_mapping_dev", "adv_fast_mapping_dev",
                           "pre_train_dev"]:
                print("======= eval: %s" % corpus)
                do_eval(sv, env, listener_model, speaker_model, args,
                        corpus=corpus)

        if i % verbose_interval == 0:
            ####### MORE EVALUATION

            eval_args = (env, listener_model, speaker_model, args)

            # Eval with an adversarial batch, using listener for inference.
            pct_advfm_success = eval_success("adv_fast_mapping_dev",
                                             *eval_args, n_batches=5)

            # Eval with a non-adversarial FM batch.
            pct_fm_success = eval_success("fast_mapping_dev",
                                          *eval_args, n_batches=5)

            # Finally, eval on pre-train dev.
            pct_pt_success = eval_success("pre_train_dev",
                                          *eval_args, n_batches=5)

        ###### UPDATES

        # Update listener with synthetic batch.
        predictions, losses, norms, pct_success = \
                run_trial(batch, listener_model, speaker_model,
                          update_listener=True, update_speaker=False)

        # Update speaker with mixture of FM batch, pre-train.
        _, losses2, norms2, _ = \
                run_trial(stabilizer_batch, listener_model, speaker_model,
                          update_listener=False, update_speaker=True)


        ######## LOGGING

        # Pull listener losses/norms from first, speaker losses/norms from
        # second
        losses = (losses[0], losses2[1])
        norms = (norms[0], norms2[1])

        out_str = "%5f\t%5f\tL_SYNTH:% 3.2f"
        vals = losses + (pct_success * 100,)
        if i % verbose_interval == 0:
            out_str += "\tL_FM:% 3.2f\tL_ADVFM:% 3.2f\tL_PT:% 3.2f"
            vals += (pct_fm_success * 100, pct_advfm_success * 100,
                     pct_pt_success * 100)
        else:
            out_str += "\t\t\t\t\t\t"
        out_str += "\t%5f\t%5f"
        vals += norms

        tqdm.write(out_str % vals)


def main(args):
    env = VGEnv(args.corpus_path, embedding_dim=args.embedding_dim,
                fm_neg_synth=args.fast_mapping_neg_synth)
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
    elif args.optimizer == "sgd":
        opt_f = lambda lr: tf.train.GradientDescentOptimizer(lr)

    l_opt = opt_f(args.listener_learning_rate)
    l_global_step = tf.Variable(0, name="global_step_listener")

    l_grads = l_opt.compute_gradients(listener_model.loss)
    listener_model.train_op = l_opt.apply_gradients(l_grads,
                                                    global_step=l_global_step)

    speaker_lr = args.listener_learning_rate * args.speaker_lr_factor
    s_opt = opt_f(speaker_lr)
    s_global_step = tf.Variable(0, name="global_step_speaker")

    s_grads = s_opt.compute_gradients(speaker_model.loss)
    speaker_model.train_op = s_opt.apply_gradients(s_grads,
                                                   global_step=s_global_step)

    from pprint import pprint
    print("Listener gradients:")
    pprint([v.name for grad, v in l_grads if grad is not None])
    print("\nSpeaker gradients:")
    pprint([v.name for grad, v in s_grads if grad is not None])

    global l_norm, s_norm
    l_norm = tf.global_norm([grad for grad, _ in l_grads])
    s_norm = tf.global_norm([grad for grad, _ in s_grads])

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

            if args.fast_mapping_k > 0:
                print("============== FAST MAPPING")
                fm_batch = run_fm_phase(sv, env, listener_model, speaker_model, args,
                                        k=args.fast_mapping_k)
            else:
                fm_batch = None

            print("============== DREAMING")
            run_dream_phase(sv, env, listener_model, speaker_model, fm_batch,
                            args)

            sv.request_stop()


if __name__ == '__main__':
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/vg")
    p.add_argument("--corpus_path")

    p.add_argument("--summary_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)

    p.add_argument("--optimizer", choices=["momentum", "adagrad", "sgd"],
                   default="momentum")

    # Basic training details.
    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--negative_samples", type=int, default=5)
    p.add_argument("--listener_learning_rate", type=float, default=0.001)
    p.add_argument("--speaker_lr_factor", type=float, default=100)

    # Model architecture.
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--listener_hidden_dim", type=int, default=256)
    p.add_argument("--speaker_hidden_dim", type=int, default=256)
    p.add_argument("--dropout_keep_prob", type=float, default=0.8)

    # Fast mapping.
    p.add_argument("--fast_mapping_k", type=int, default=64)
    p.add_argument("--fast_mapping_threshold", type=float, default=1.5,
                   help=("Continue FM training until speaker loss drops "
                         "below this value"))
    p.add_argument("--fast_mapping_neg_synth", type=int, default=3,
                   help=("Number of negative examples with the FM relation "
                         "to synthesize for FM batches."))

    # Dreaming.
    p.add_argument("--n_dream_iters", type=int, default=501)
    p.add_argument("--dream_ratio", type=float, default=0.5)
    p.add_argument("--dream_stabilizer_factor", type=float, default=0.5,
                   help=("% of stabilizer batch fed to speaker during "
                         "dreaming which should be drawn from observed fast "
                         "mapping data"))
    p.add_argument("--dream_p_swap", type=float, default=0.75,
                   help=("Probability of swapping positive referent with a "
                         "negative one in silent batches"))

    main(p.parse_args())
