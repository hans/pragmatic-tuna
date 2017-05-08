from argparse import ArgumentParser
from collections import namedtuple
import itertools
import os.path
from pprint import pprint
import sys

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments.vg import VGEnv
from pragmatic_tuna.models.ranking_listener import BoWRankingListener
from pragmatic_tuna.models.vg_speaker import WindowedSequenceSpeakerModel
from pragmatic_tuna.util import make_summary, reset_momentum


Model = namedtuple("Model", ["listener", "speaker"])


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


def run_trial(batch, model, update_listener=True, update_speaker=True,
              infer_with_speaker=False):
    utterances, candidates, lengths, n_cands = batch

    # Fetch model scores and rank for pragmatic listener inference.
    listener_scores = model.listener.score(*batch)
    speaker_scores = model.speaker.score(*batch)
    scores, results = infer_trial(candidates, listener_scores, speaker_scores,
                                  infer_with_speaker=infer_with_speaker)

    successes = [result_i == candidates_i[0]
                     and scores_i.min() != scores_i.max()
                 for result_i, candidates_i, scores_i
                 in zip(results, candidates, scores)]

    # TODO: joint optimization?
    l_loss = s_loss = 0.0
    l_gn = s_gn = 0.0
    if update_listener:
        l_loss, l_gn = model.listener.observe(*batch, norm_op=l_norm)
    if update_speaker:
        s_loss, s_gn = model.speaker.observe(*batch, norm_op=s_norm)

    pct_success = np.mean(successes)
    losses = (l_loss, s_loss)
    norms = (l_gn, s_gn)
    return results, losses, norms, pct_success


def run_train_phase(sv, env, model, args):
    for i in trange(args.n_iters):
        batch = env.get_batch("pre_train_train", batch_size=args.batch_size,
                              negative_samples=args.negative_samples)
        predictions, losses, _, pct_success = run_trial(batch, model)

        tqdm.write("%5f\t%5f\t%.2f" % (*losses, pct_success * 100))

        if i % args.summary_interval == 0:
            eval_args = (env, model, args)

            summary_points = {
                "listener_loss": losses[0],
                "speaker_loss": losses[1],
                "listener_train_success": pct_success,
                "listener_dev_success": do_eval(*eval_args,
                    corpus="pre_train_dev", n_batches=5),
            }
            summary_points.update({
                ("speaker_advfm_%s_success") % (corpus[corpus.rindex("_")+1:]):
                    do_eval(*eval_args, corpus=corpus, n_batches=2,
                            infer_with_speaker=True)
                for corpus in env.advfm_corpora["dev"]})
            summary = make_summary(summary_points)
            sv.summary_computed(tf.get_default_session(), summary)

        if i % args.eval_interval == 0 or i == args.n_iters - 1:
            tqdm.write("====================== DEV EVAL AT %i" % i)
            do_eval(env, model, args, verbose=True)


def do_eval(env, model, args, batch=None, corpus="pre_train_dev",
            n_batches=1, resample_utterances=False, verbose=False,
            **kwargs):
    """
    Evaluate models on minibatch(es) or a whole corpus.

    Arguments:
        env:
        model:
        args: CLI args
        batch: Optional specific batch on which to evaluate
        corpus: Corpus from which to fetch batches
        n_batches: Number of batches over which to compute success statistics.
            If 0, use the entire corpus.
        resample_utterances: If True, resample utterances from the speaker
            model before performing inference. (Ignores the gold utterances in
            the corpus.)
        verbose: If `True`, print per-example detail on success for one of the
            batches
        kwargs: kwargs forwarded to `run_trial`

    Returns:
        avg_success: Average success across all examples
    """

    # Prepare kwargs for `run_trial`
    kwargs_ = kwargs
    kwargs = {
        "update_listener": False,
        "update_speaker": False,
        "infer_with_speaker": False,
    }
    kwargs.update(kwargs_)

    if batch is not None:
        batches = [batch]
    else:
        # Fetch `n` batches
        if n_batches == 0:
            batches = env.iter_batches(corpus, batch_size=args.batch_size,
                                       negative_samples=args.negative_samples)
        else:
            batches = [env.get_batch(corpus, batch_size=args.batch_size,
                                     negative_samples=args.negative_samples)
                       for _ in range(n_batches)]

    batch_sizes, predictions, pct_success = [], [], []
    for batch in batches:
        if resample_utterances:
            _, b_cands, _, b_n_cands = batch
            silent_batch = (b_cands, b_n_cands)
            b_utt, b_lengths = sample_utterances(env, silent_batch, model.speaker)
            batch = (b_utt, b_cands, b_lengths, b_n_cands)

        b_pred, _, _, b_pct = run_trial(batch, model, **kwargs)
        batch_sizes.append(len(batch[1]))
        predictions.append(b_pred)
        pct_success.append(b_pct)

    # Average % successes
    mean_success = sum([pct_i * size_i for pct_i, size_i
                        in zip(pct_success, batch_sizes)])
    mean_success /= sum(batch_sizes)

    if verbose:
        # Verbose output for one of the batches
        i = np.random.choice(len(batches))
        batch_i, predictions_i = batches[i], predictions[i]
        print_verbose_eval(env, model, batch_i, predictions_i)

    return mean_success


def print_verbose_eval(env, model, batch, predictions):
    utt, cands, lengths, n_cands = batch

    # Test: draw some samples for this new input
    silent_batch = (cands, n_cands)
    utt_s, lengths_s = sample_utterances(env, silent_batch, model.speaker)

    # Prepare to print per-example results
    correct, false = [], []
    for utterance, cands, prediction, sample in zip(utt.T, cands,
                                                    predictions, utt_s.T):
        utterance = " ".join(env.utterance_to_tokens(utterance))
        sample = " ".join(env.utterance_to_tokens(sample))

        pos_cand = cands[0]
        dest = correct if prediction == pos_cand else false
        dest.append("%40s\t%60s\t%40s\t%40s" %
                    (utterance, sample,
                     " ".join([env.graph_vocab[idx] for idx in pos_cand]),
                     " ".join([env.graph_vocab[idx] for idx in prediction])))

    tqdm.write("=========== Correct:")
    tqdm.write("\n".join(correct))

    tqdm.write("\n========== False:")
    tqdm.write("\n".join(false))


def run_fm_phase(sv, env, model, args, k=None, corpus="fast_mapping_train"):
    """
    Run the "fast mapping" (== zero-shot inference) phase.

    Arguments:
        sv: Supervisor
        env:
        model:
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
    if args.fast_mapping_threshold > 0:
        while speaker_loss > args.fast_mapping_threshold:
            predictions, losses, norms, pct_success = \
                    run_trial(batch, model, infer_with_speaker=True,
                            update_speaker=True, update_listener=False)
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

    do_eval(env, model, args, batch=batch)
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
    for eos_idx, example in eos_positions:
        # Every example must have at least one token -- otherwise TF graph
        # breaks
        if eos_idx == 0: continue
        lengths[example] = min(lengths[example], eos_idx)

    # # DEV: keep the lengths, but randomly replace all non-EOS tokens with other
    # # tokens
    # valid_tokens = [x for x in range(env.vocab_size)
    #                 if x not in [env.word_eos_id, env.word_unk_id]]
    # utterances = np.random.choice(np.array(valid_tokens), replace=True,
    #                               size=utterances.shape)
    # # END DEV

    # Mask to make sure these examples make sense
    mask = np.tile(np.arange(max_length).reshape((-1, 1)), (1, batch_size))
    mask = lengths.reshape((1, -1)) <= mask
    utterances[mask] = env.word_eos_id

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


def run_dream_phase(sv, env, model, fm_batch, args):
    verbose_interval = 100
    full_eval_interval = 500
    for i in trange(args.n_dream_iters):
        ####### DATA PREP

        # Synthetic dream batch.
        batch = synthesize_dream_batch(env, model.speaker, args.batch_size,
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
            do_eval(env, model, args, batch=batch, verbose=True)

            dev_corpora = ["dreaming_dev", "pre_train_dev"]
            dev_corpora += env.advfm_corpora["dev"]
            for corpus in dev_corpora:
                print("======= eval: %s" % corpus)
                do_eval(env, model, args, corpus=corpus, verbose=True)

        if i % verbose_interval == 0 or i % full_eval_interval == 0:
            ####### MORE EVALUATION

            # Full eval: evaluate on entire corpus.
            full_eval = i % full_eval_interval == 0

            eval_args = (env, model, args)

            # Eval with an adversarial batch, using listener for inference.
            advfm_successes = {corpus: do_eval(*eval_args, corpus=corpus,
                                               n_batches=0 if full_eval else 2)
                               for corpus in env.advfm_corpora["dev"]}

            # Compute ADVFM(new) - ADFM(old)
            old_advfm = ["adv_fast_mapping_dev_in", "adv_fast_mapping_dev_on"]
            new_advfm = set(env.advfm_corpora["dev"]) - set(old_advfm)
            advfm_success_diff = \
                    np.mean([advfm_successes[name] for name in new_advfm]) \
                    - np.mean([advfm_successes[name] for name in old_advfm])

            # Eval with a non-adversarial FM batch.
            pct_fm_success = do_eval(*eval_args, corpus="dreaming_dev",
                                     n_batches=0 if full_eval else 5)

            # Eval with a non-adversarial FM batch, sampling utterances from
            # speaker!
            pct_fms_success = do_eval(*eval_args, corpus="dreaming_dev",
                                      n_batches=0 if full_eval else 5,
                                      resample_utterances=True)

            # Finally, eval on pre-train dev.
            pct_pt_success = do_eval(*eval_args, corpus="pre_train_dev",
                                     n_batches=0 if full_eval else 5)

        ###### UPDATES

        # Update listener with synthetic batch.
        predictions, losses, norms, pct_success = \
                run_trial(batch, model,
                          update_listener=True, update_speaker=False)

        # Update speaker with mixture of FM batch, pre-train.
        _, losses2, norms2, _ = \
                run_trial(stabilizer_batch, model,
                          update_listener=False, update_speaker=True)


        ######## LOGGING

        # Pull listener losses/norms from first, speaker losses/norms from
        # second
        losses = (losses[0], losses2[1])
        norms = (norms[0], norms2[1])

        is_verbose = i % verbose_interval == 0
        is_full_eval = i % full_eval_interval == 0

        if is_verbose or is_full_eval:
            out_str = "%5f\t%5f\tL_SYNTH:% 3.2f\tL_FM:% 3.2f\tL_FMS:% 3.2f\tL_PT:% 3.2f"
            if is_full_eval:
                out_str = "%%%%" + out_str
            vals = losses + (pct_success * 100, pct_fm_success * 100,
                             pct_fms_success * 100, pct_pt_success * 100)
        else:
            out_str = "%5f\t%5f\tL_SYNTH:% 3.2f\t\t\t\t\t\t\t"
            vals = losses + (pct_success * 100,)
        out_str += "\t%5f\t%5f"
        vals += norms

        tqdm.write(out_str % vals)

        if is_verbose:
            out_fields = []
            for adv_corpus in sorted(env.advfm_corpora["dev"]):
                name = adv_corpus[adv_corpus.rindex("_")+1:]
                score = advfm_successes[adv_corpus] * 100
                out_fields.append("%s\t% 3.2f" % (name, score))

            out_fields.append("NEW-OLD\t% 3.2f" % (advfm_success_diff * 100))

            out_str = "\t" + "\t".join(out_fields)
            if is_full_eval:
                out_str = "%%%%" + out_str
            tqdm.write(out_str)


def rig_embedding_gradients(opt, loss, embedding_var, rig_idxs, scale=100.):
    """
    Compute gradients on an embedding variable.

    Upscale certain embedding indices (or mask out non-specified embedding
    updates altogether).

    Arguments:
        opt:
        loss:
        embedding_var:
        rig_idxs: Embeddings (rows of `embedding_var`) to select
        scale: Float scalar for selected embeddings. If zero, mask non-selected
            embedding gradients to zero and leave selected embeddings untouched.
    """
    grads = {v: grad for grad, v in opt.compute_gradients(loss)
             if grad is not None}
    emb_grads = grads[embedding_var]

    scale_mask = tf.zeros_like(emb_grads.indices)
    for rig_idx in rig_idxs:
        scale_mask = tf.logical_or(scale_mask,
                                   tf.equal(emb_grads.indices, rig_idx))
    scale_mask = tf.expand_dims(tf.to_float(scale_mask), 1)

    if scale == 0:
        scale = scale_mask
    else:
        scale = scale_mask * (scale - 1.0) + 1.0
    grads[embedding_var] = tf.IndexedSlices(indices=emb_grads.indices,
                                            values=emb_grads.values * scale)
    return [(grad, v) for v, grad in grads.items()]


def prepare_opt(model, args):
    if args.optimizer == "momentum":
        opt_f = lambda lr: tf.train.MomentumOptimizer(lr, 0.9)
    elif args.optimizer == "adagrad":
        opt_f = lambda lr: tf.train.AdagradOptimizer(lr)
    elif args.optimizer == "sgd":
        opt_f = lambda lr: tf.train.GradientDescentOptimizer(lr)

    l_opt = opt_f(args.listener_learning_rate)
    l_global_step = tf.Variable(0, name="global_step_listener")

    l_grads = l_opt.compute_gradients(model.listener.loss)
    model.listener.train_op = l_opt.apply_gradients(l_grads,
                                                    global_step=l_global_step)

    speaker_lr = args.listener_learning_rate * args.speaker_lr_factor
    s_opt = opt_f(speaker_lr)
    s_global_step = tf.Variable(0, name="global_step_speaker")

    s_grads = s_opt.compute_gradients(model.speaker.loss)
    model.speaker.train_op = s_opt.apply_gradients(s_grads,
                                                   global_step=s_global_step)

    global_step = l_global_step + s_global_step

    return l_grads, s_grads, global_step


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
            hidden_layers=args.speaker_hidden_layers,
            dropout_keep_prob=args.dropout_keep_prob)
    model = Model(listener_model, speaker_model)

    l_grads, s_grads, global_step = prepare_opt(model, args)

    global l_norm, s_norm
    l_norm = tf.global_norm([grad for grad, _ in l_grads])
    s_norm = tf.global_norm([grad for grad, _ in s_grads])

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
            if args.mode == "pretrain":
                print("============== TRAINING")
                run_train_phase(sv, env, model, args)

            if args.mode == "dream":
                reset_momentum()

                if args.fast_mapping_k > 0:
                    print("============== FAST MAPPING")
                    fm_batch = run_fm_phase(sv, env, model, args,
                                            k=args.fast_mapping_k)
                else:
                    fm_batch = None

                print("============== DREAMING")
                run_dream_phase(sv, env, model, fm_batch,
                                args)

            sv.request_stop()


if __name__ == '__main__':
    file_p = ArgumentParser()
    file_p.add_argument("--args_file")
    file_args, argv = file_p.parse_known_args()

    if file_args.args_file is not None:
        with open(file_args.args_file, "r") as args_f:
            prior_args = eval(args_f.read())

        # Prepend these to argv
        prepend_argv = list(itertools.chain(*[("--%s" % arg, str(value))
                                              for arg, value in prior_args.items()]))
        argv = prepend_argv + argv

    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/vg")
    p.add_argument("--corpus_path")

    p.add_argument("--summary_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)

    p.add_argument("--mode", choices=["pretrain", "dream"],
                   default="dream")

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
    p.add_argument("--speaker_hidden_layers", type=int, default=1)
    p.add_argument("--speaker_hidden_dim", type=int, default=256)
    p.add_argument("--dropout_keep_prob", type=float, default=0.8)

    # Fast mapping.
    p.add_argument("--fast_mapping_k", type=int, default=64)
    p.add_argument("--fast_mapping_threshold", type=float, default=1.5,
                   help=("Continue FM training until speaker loss drops "
                         "below this value"))

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

    args = p.parse_args(argv)
    main(args)
