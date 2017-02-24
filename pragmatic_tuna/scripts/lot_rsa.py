"""
Combine a listener and speaker model to perform RSA inference in reference games.

The inference is mediated by a logical form language / language of thought (LoT).
"""

from argparse import ArgumentParser
import json
import logging
import os.path
from pprint import pprint
import re

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from pragmatic_tuna.environments import TUNAWithLoTEnv
from pragmatic_tuna.models.listener import *
from pragmatic_tuna.models.speaker import *
from pragmatic_tuna.environments.spatial import *
from pragmatic_tuna.util import colors


LOGGER = logging.getLogger(__name__)


def infer_trial(env, obs, listener_model, speaker_model,
                num_listener_samples=128, debug=True, evaluating=False):
    """
    Run RSA inference for a given trial.

    Args:
        env:
        obs: Observation from environment
        listener_model:
        speaker_model:
        num_listener_samples:
        evaluating: If `True`, will run argmax-inference rather than sampling

    Returns:
        lfs: LF IDs sampled from `listener_model`
        weights: Associated combined model weight for each LF
        rejs_per_sample:
    """

    items, words = obs

    # Sample N LFs from the predicted distribution, accepting only when they
    # resolve to a referent in the scene.
    lfs, g_lfs, weights = [], [], []
    lf_score_cache = {}
    num_rejections = 0

    # Pre-sample a batch, and refresh as necessary.
    pseudo_batch = [words] * args.num_listener_samples
    sample_batch_cursor = len(pseudo_batch)

    while len(weights) < args.num_listener_samples:
        if sample_batch_cursor == len(pseudo_batch):
            sample_batch = listener_model.sample_batch(
                    pseudo_batch, argmax=evaluating, evaluating=evaluating)
            sample_batch_cursor = 0
            continue
        sample_batch_lfs, sample_batch_probs = sample_batch
        lf = sample_batch_lfs[sample_batch_cursor]
        p_lf = sample_batch_probs[sample_batch_cursor]
        sample_batch_cursor += 1

        # Resolve referent.
        referent = env.resolve_lf(lf)
        if not referent:
            # Dereference failed. No object matched.
            num_rejections += 1
            continue
        referent = env._domain.index(referent[0])

        # Sample an LF z' ~ p(z|r).
        g_lf = lf #env.sample_lf(referent=referent, n_parts=len(words) // 2)

        # Retrieved unnormalized likelihood p~(u|z), partition p(z)
        cache_key = tuple(g_lf)
        try:
            p_utterance = lf_score_cache[cache_key]
        except KeyError:
            p_utterance = speaker_model.score(g_lf, words)
            lf_score_cache[cache_key] = p_utterance

        lfs.append(lf)
        g_lfs.append(g_lf)
        weights.append((np.exp(p_utterance), p_lf))

    # Mix listener+speaker scores.
    # TODO: customizable
    mixed_weights = [speaker_weight / listener_weight
                     for speaker_weight, listener_weight in weights]
    data = sorted(zip(lfs, mixed_weights, weights), key=lambda el: el[1],
                  reverse=True)

    rejs_per_sample = num_rejections / num_listener_samples

    # Debug printing.
    if debug:
        seen = set()
        for lf, mixed_weight, weight in data:
            lf = tuple(lf)
            if lf in seen:
                continue
            seen.add(lf)

            LOGGER.debug("LF %30s  =>  Referent %10s  =>  (%.3g, %.3g, %.3g)",
                         env.describe_lf(lf),
                         env.resolve_lf(lf)[0]["attributes"][env.atom_attribute],
                         #env.describe_lf(g_lf),
                         weight[0], weight[1], mixed_weight)
        LOGGER.debug("%sRejections per sample: %.2f%s", colors.BOLD + colors.WARNING,
                                                        rejs_per_sample, colors.ENDC)

    listener_model.reset()
    return data, rejs_per_sample


def run_listener_trial(listener_model, speaker_model, env, sess, args,
                       evaluating=False, silent=False):
    env.configure(dreaming=False)
    obs = env.reset()

    n_iterations, success, first_success = 0, False, -1
    first_successful_lf_pred = None
    rejs_per_sample = np.inf
    terminate = False
    while not terminate:
        if n_iterations > 1000:
            if not silent:
                LOGGER.error("%sFailed to converge after 1000 listener trials. Dying.%s",
                             colors.FAIL, colors.ENDC)
            break

        lfs, rejs_per_sample = \
                infer_trial(env, obs, listener_model, speaker_model,
                            num_listener_samples=args.num_listener_samples,
                            debug=args.debug and not silent,
                            evaluating=evaluating)

        # Now select action based on maximum score.
        lf_pred = lfs[0][0]
        _, reward, _, _ = env.step(lf_pred)

        success = reward > 0
        if not silent:
            color = colors.OKGREEN if success else colors.FAIL
            LOGGER.info("%s%s => %s%s", colors.BOLD + color, env._trial["string_description"],
                                        env.describe_lf(lf_pred), colors.ENDC)

        if success and first_success == -1:
            first_success = n_iterations
            first_successful_lf_pred = lf_pred
        n_iterations += 1

        # Find the highest-scoring LF that dereferences to the correct
        # referent.
        gold_lf, gold_lf_pos = None, -1
        try:
            gold_lf, gold_lf_pos = \
                    next((lf_i, i) for i, (lf_i, _, _) in enumerate(lfs)
                         if env.check_lf(lf_i))
        except StopIteration:
            gold_lf, gold_lf_pos = None, -1
        else:
            if not silent:
                LOGGER.info("gold %s %i", env.describe_lf(gold_lf), gold_lf_pos)

            # Update model parameters.
            if not evaluating:
                listener_model.observe(obs, gold_lf)
                speaker_model.observe(obs, gold_lf)

        listener_model.reset()

        # Termination condition
        terminate = evaluating \
                or (success and rejs_per_sample <= args.max_rejections_after_trial)

    return first_success, first_successful_lf_pred, gold_lf_pos

def run_batch_dream_trials(listener_model, generative_model, env, sess, args):

    trials = env.sample_prev_trials(args.dream_samples)
    gold_lfs = []
    gold_utterances = []

    for trial in trials:
        items, gold_words = env._set_trial(trial)
        referent_idx = [i for i, referent in enumerate(env._domain)
                        if referent["target"]][0]
        referent = env._domain[referent_idx]
        success = False
        string_matches = False
        i = 0
        while not success or not string_matches:
            LOGGER.info("Dream trial %i" % i)

            # Sample an LF from p(z|r).
            g_lf = env.sample_lf(referent=referent_idx)

            # Sample utterances from p(u|z).
            words = generative_model.sample(g_lf, argmax=True).split()
            word_ids = np.array([env.word2idx[word]
                                    for word in words])

            # Build a fake observation object for inference.
            obs = (items, words)

            # Run listener model q(z|u).
            l_lfs, rejs_per_sample = \
                    infer_trial(env, obs, listener_model, generative_model,
                                num_listener_samples=args.num_listener_samples,
                                debug=False)
            # Literally dereference and see if we get the expected referent.
            l_referent = env.resolve_lf(l_lfs[0][0])
            if l_referent:
                success = l_referent[0] == referent

            color = colors.OKGREEN if success else colors.FAIL
            LOGGER.info("%s%s => %s%s", colors.BOLD + color, " ".join(words),
                                        env.describe_lf(l_lfs[0][0]), colors.ENDC)
            LOGGER.info("%sRejections per sample: %.2f%s",
                        colors.BOLD + colors.WARNING, rejs_per_sample, colors.ENDC)

            # TODO: This is only a good stopping criterion when we force the LF to
            # be the same as the gold LF. Otherwise it's too strict!
            string_matches = gold_words == words

            i += 1
            if i > 1000:
                LOGGER.error("%sFailed to dream successfully after 1000 trials. Dying.%s",
                             colors.FAIL, colors.ENDC)
                break

            listener_model.reset()

        if success and string_matches:
            gold_lfs.append(l_lfs[0][0])
            gold_utterances.append(gold_words)

    generative_model.observe_batch(gold_utterances, gold_lfs)


def run_dream_trial(listener_model, generative_model, env, sess, args):
    """
    Run a single dream trial.
    """
    env.configure(dreaming=True)
    items, gold_words = env.reset()

    referent_idx = [i for i, referent in enumerate(env._domain)
                    if referent["target"]][0]
    referent = env._domain[referent_idx]

    success = False
    string_matches = False
    i = 0
    while not success or not string_matches:
        LOGGER.info("Dream trial %i" % i)

        # Sample an LF from p(z|r).
        g_lf = env.sample_lf(referent=referent_idx)

        # Sample utterances from p(u|z).
        words = generative_model.sample(g_lf).split()
        word_ids = np.array([env.word2idx[word]
                                for word in words])

        # Build a fake observation object for inference.
        obs = (items, words)

        # Run listener model q(z|u).
        l_lfs, _ = infer_trial(env, obs, listener_model, generative_model,
                               num_listener_samples=args.num_listener_samples,
                               debug=False)
        # Literally dereference and see if we get the expected referent.
        l_referent = env.resolve_lf(l_lfs[0][0])
        if l_referent:
            success = l_referent[0] == referent

        color = colors.OKGREEN if success else colors.FAIL
        LOGGER.info("%s%s => %s%s", colors.BOLD + color, " ".join(words),
                                    env.describe_lf(l_lfs[0][0]), colors.ENDC)

        # TODO: This is only a good stopping criterion when we force the LF to
        # be the same as the gold LF. Otherwise it's too strict!
        string_matches = gold_words == words

        i += 1
        if i > 1000:
            LOGGER.error("%sFailed to dream successfully after 1000 trials. Dying.%s",
                         colors.FAIL, colors.ENDC)
            break

        listener_model.reset()

        if success:
            # TODO: can use `obs` here if we know the listener model to be
            # stable
            input_obs = (items, gold_words)
            generative_model.observe(input_obs, l_lfs[0][0])


def eval_offline_ctx(listener_model, speaker_model, examples, env, sess, args):
    """
    Evaluate the listener model relative to ground-truth utterance-LF pairs.
    This tests listener model predictions within the same environment contexts
    in which it was learning.
    """
    listener_model.reset()
    env.configure(reset_cursor=True)

    learned_mapping = {}
    for i in trange(args.num_test_trials, desc="Test trial"):
        first_success, best_lf, gold_lf_pos = run_listener_trial(
                listener_model, speaker_model, env, sess, args,
                evaluating=True, silent=True)
        if best_lf is None:
            continue

        # TODO: Assumes each string description maps to a unique trial
        string_desc = env._trial["string_description"]
        learned_mapping[string_desc] = env.describe_lf(best_lf)

    check_results = []
    for words_string, lf_candidates in examples:
        try:
            learned_mapping_i = learned_mapping[words_string]
        except KeyError:
            check_i = False
            learned_mapping_i = "???"
        else:
            check_i = learned_mapping_i in lf_candidates
        check_results.append((words_string, check_i))

        color = colors.OKGREEN if check_i else colors.FAIL
        LOGGER.info("\t%s%30s => %30s%s"
              % (color, words_string, learned_mapping_i, colors.ENDC))

    return check_results


def eval_offline_cf(listener_model, examples, env):
    """
    Evaluate the listener model relative to ground-truth utterance-LF pairs.
    This tests listener model predictions outside of any particular environment
    context.

    Args:
        listener_model:
        examples: Pairs of `(words_string, lf_candidates)`
        env:
    """
    results = []
    listener_model.reset()
    for words_string, lf_candidates in examples:
        words = words_string.strip().split()

        lf_candidate_tok_ids = []
        for lf_candidate in lf_candidates:
            tokens = re.split("[()]| AND ", lf_candidate.strip())
            ids = [env.lf_token_to_id[token] for token in tokens if token]
            lf_candidate_tok_ids.append(tuple(ids))

        # DEV: assumes we're using a non-BOW listener model
        sampled_lf, p = listener_model.sample(words, argmax=True, evaluating=True)
        listener_model.reset()
        success = tuple(sampled_lf) in lf_candidate_tok_ids
        results.append((words_string, env.describe_lf(sampled_lf), success))

        color = colors.OKGREEN if success else colors.FAIL
        LOGGER.info("\t%s%30s => %30s%s" % (color, words_string, env.describe_lf(sampled_lf), colors.ENDC))

    return results


def eval_offline(listener_model, speaker_model, env, sess, args):
    if not args.gold_path:
        return None, None

    with open(args.gold_path, "r") as gold_f:
        listener_examples = json.load(gold_f)

        LOGGER.info("\n%s==========\nOFFLINE EVALUATION\n==========%s",
                    colors.HEADER, colors.ENDC)

        # Context-grounded offline evaluation.
        LOGGER.info("%sWith context:%s", colors.HEADER, colors.ENDC)
        ctx_results = eval_offline_ctx(
                listener_model, speaker_model, listener_examples,
                env, sess, args)
        ctx_successes = [s for _, s in ctx_results]

        # Context-free offline evaluation.
        LOGGER.info("\n%sWithout context:%s", colors.HEADER, colors.ENDC)
        cf_results = eval_offline_cf(
                listener_model, listener_examples, env)
        cf_successes = [s for _, _, s in cf_results]

    return ctx_successes, cf_successes


def build_train_graph(model, env, args, scope="train"):
    if args.learning_method == "rl":
        model.build_rl_gradients()
        gradients = model.rl_gradients
    elif args.learning_method == "xent":
        try:
            model.build_xent_gradients()
        except AttributeError:
            # Learning not defined.
            return None, None
        gradients = model.xent_gradients
    else:
        raise NotImplementedError("undefined learning method " + args.learning_method)

    with tf.variable_scope(scope):
        global_step = tf.Variable(0, name="global_step", dtype=tf.int64,
                                  trainable=False)
        opt = tf.train.AdagradOptimizer(args.learning_rate)
        train_op = opt.apply_gradients(gradients, global_step=global_step)

        # Make a dummy train_op that works with TF partial_run.
        with tf.control_dependencies([train_op]):
            train_op = tf.constant(0.0, name="dummy_train_op")

    return train_op, global_step


def train(args):
    max_conjuncts = args.max_timesteps / 2
    assert int(max_conjuncts) == max_conjuncts
    max_conjuncts = int(max_conjuncts)

    env = TUNAWithLoTEnv(args.corpus_path, corpus_selection=args.corpus_selection,
                         bag=args.bag_env, functions=FUNCTIONS[args.fn_selection],
                         max_conjuncts=max_conjuncts,
                         atom_attribute=args.atom_attribute)
    listener_model = LISTENER_MODELS[args.listener_model](env, args)
    speaker_model = SPEAKER_MODELS[args.speaker_model](env, listener_model, args)

    listener_train_op, listener_global_step = \
            build_train_graph(listener_model, env, args, scope="train/listener")
    speaker_train_op, speaker_global_step = \
            build_train_graph(speaker_model, env, args, scope="train/speaker")
    # TODO does momentum / etc. of shared parameters get shared in this case?
    # check variables after build_train_graph
    listener_model.train_op = listener_train_op
    speaker_model.train_op = speaker_train_op

    # Prepare for logging/summaries/etc.
    tf.gfile.MakeDirs(args.logdir)
    LOGGER.propagate = False

    # Offline metrics
    ctx_results, cf_results = [], []
    # Online metrics
    online_results = []
    with tf.Session() as sess:
        with sess.as_default():
            for run_i in range(args.num_runs):
                logfile = os.path.join(args.logdir, "run_%i.log" % run_i)
                LOGGER.handlers = [logging.FileHandler(logfile)]
                env.configure(reset_cursor=True)

                tqdm.write("%sBeginning training run %i.%s\n\n" % (colors.BOLD, run_i, colors.ENDC))
                sess.run(tf.global_variables_initializer())

                run_online_results = []

                for i in trange(args.num_trials, desc="Trial"):
                    LOGGER.info("\n%s==============\nLISTENER TRIAL\n==============%s",
                                colors.HEADER, colors.ENDC)

                    first_success, _, _ = run_listener_trial(listener_model, speaker_model,
                                                             env, sess, args)
                    run_online_results.append(first_success == 0)

                    if args.dream:
                        LOGGER.info("\n%s===========\nDREAM TRIAL\n===========%s",
                                    colors.HEADER, colors.ENDC)

                        if args.dream_samples == 1:
                            run_dream_trial(listener_model, speaker_model,
                                            env, sess, args)
                        else:
                            run_batch_dream_trials(listener_model, speaker_model,
                                                  env, sess, args)

                online_results.append(run_online_results)

                ctx_successes, cf_successes = eval_offline(
                        listener_model, speaker_model, env, sess, args)
                ctx_results.append(ctx_successes)
                cf_results.append(cf_successes)

    LOGGER.info("\n%s==========\nOVERALL EVALUATION\n==========%s",
                colors.HEADER, colors.ENDC)
    avg_online_accuracy = np.array(online_results).mean(axis=0)
    LOGGER.info("%sAverage online accuracy: %.3f%%%s",
                colors.BOLD, avg_online_accuracy.mean() * 100, colors.ENDC)
    LOGGER.info("%sOnline accuracy per trial:%s\n\t%s",
                colors.BOLD, colors.ENDC,
                 "\n\t".join("%i\t%.3f" % (i, acc_i * 100)
                             for i, acc_i in enumerate(avg_online_accuracy)))

    if args.gold_path:
        avg_ctx_accuracy = np.mean(ctx_results, axis=0)
        LOGGER.info("%sAverage offline accuracy with context: %.3f%%%s",
                    colors.BOLD, avg_ctx_accuracy.mean() * 100, colors.ENDC)
        LOGGER.info("%sAverage offline accuracy with context per trial:%s\n\t%s",
                    colors.BOLD, colors.ENDC,
                    "\n\t".join("%i\t%.3f" % (i, acc_i * 100)
                                for i, acc_i in enumerate(avg_ctx_accuracy)))

        avg_cf_accuracy = np.mean(cf_results, axis=0)
        LOGGER.info("%sAverage offline accuracy without context: %.3f%%%s",
                    colors.BOLD, avg_cf_accuracy.mean() * 100, colors.ENDC)
        LOGGER.info("%sAverage offline accuracy without context per trial:%s\n\t%s",
                    colors.BOLD, colors.ENDC,
                    "\n\t".join("%i\t%.3f" % (i, acc_i * 100)
                                for i, acc_i in enumerate(avg_cf_accuracy)))


SPEAKER_MODELS = {
    "sequence":  lambda env, listener, args: \
            WindowedSequenceSpeakerModel(env,
                                         word_embeddings=listener.word_embeddings,
                                         lf_embeddings=listener.lf_embeddings,
                                         embedding_dim=args.embedding_dim,
                                         max_timesteps=args.max_timesteps),
    "shallow": lambda env, listener, args: \
            ShallowSequenceSpeakerModel(env,
                                        lf_embeddings=listener.lf_embeddings,
                                        embedding_dim=args.embedding_dim,
                                        max_timesteps=args.max_timesteps),
    "ensemble": None # TODO set up
}

LISTENER_MODELS = {
    "window": lambda env, args: WindowedSequenceListenerModel(
        env, embedding_dim=args.embedding_dim,
        max_timesteps=args.max_timesteps)
}


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/tuna")
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--analyze_weights", default=False, action="store_true")

    p.add_argument("--corpus_path", required=True)
    p.add_argument("--corpus_selection", default=None)
    p.add_argument("--fn_selection", default="spatial_simple",
                   choices=FUNCTIONS.keys())
    p.add_argument("--atom_attribute", default="shape")
    p.add_argument("--gold_path",
                   help=("Path to JSON file containing gold listener "
                         "utterance -> LF interpretations"))

    p.add_argument("--speaker_model", default="shallow",
                   choices=SPEAKER_MODELS.keys())
    p.add_argument("--listener_model", default="window",
                   choices=LISTENER_MODELS.keys())
    p.add_argument("--bag_env", default=False, action="store_true")
    p.add_argument("--embedding_dim", type=int, default=4)

    p.add_argument("--max_timesteps", type=int, default=2)
    p.add_argument("--dream", default=False, action="store_true")
    p.add_argument("--num_listener_samples", type=int, default=5)
    p.add_argument("--max_rejections_after_trial", type=int, default=3)
    p.add_argument("--argmax_listener", action="store_true", default=False)

    p.add_argument("--num_runs", default=1, type=int,
                   help="Number of times to repeat entire training process")
    p.add_argument("--learning_method", default="xent", choices=["rl", "xent"])
    p.add_argument("--num_trials", default=100, type=int)
    p.add_argument("--num_test_trials", default=20, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--dream_samples", default=1, type=int)

    args = p.parse_args()
    pprint(vars(args))

    LOGGER.setLevel(logging.DEBUG)

    train(args)
