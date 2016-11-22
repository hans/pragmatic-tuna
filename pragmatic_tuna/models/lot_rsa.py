from argparse import ArgumentParser
from collections import Counter, defaultdict, namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import trange, tqdm

from pragmatic_tuna.environments import TUNAWithLoTEnv
from pragmatic_tuna.environments.spatial import *
from pragmatic_tuna.reinforce import reinforce_episodic_gradients


class NaiveGenerativeModel(object):

    """
    A very stupid generative utterance model $P(u | z)$ which is intended to
    map from bag-of-features $z$ representations to bag-of-words $u$
    representations. Optionally performs add-1 smoothing.
    """

    def __init__(self, smooth=True):
        self.smooth = smooth
        self.counter = defaultdict(lambda: Counter())

    def observe(self, z, u):
        z, u = z, tuple(u)
        self.counter[z][u] += 1

    def score(self, z, u):
        z, u = z, tuple(u)
        score = self.counter[z][u]
        if self.smooth:
            # Add-1 smoothing.
            score += 1
        return score


class ListenerModel(object):

    def __init__(self, env, scope="listener"):
        assert not env.bag
        self.env = env
        self._scope = tf.variable_scope(scope)
        self._build_graph()

    def _build_graph(self):
        """
        Build the core model graph.
        """
        n_outputs = len(self.env.lf_functions) * len(self.env.lf_atoms)

        with self._scope:
            self.items = tf.placeholder(tf.float32, shape=(None, self.env.attr_dim))
            self.utterance = tf.placeholder(tf.float32, shape=(self.env.vocab_size,))

            self.scores = layers.fully_connected(tf.expand_dims(self.utterance, 0),
                                                 n_outputs, tf.identity)
            self.probs = tf.squeeze(tf.nn.softmax(self.scores))

    def build_rl_gradients(self):
        if hasattr(self, "rl_action"):
            return (self.rl_action, self.rl_reward), (self.rl_gradients,)

        action = tf.placeholder(tf.int32, shape=(), name="action")
        reward = tf.placeholder(tf.float32, shape=(), name="reward")

        scores = [self.scores]
        actions = [action]
        rewards = reward
        gradients = reinforce_episodic_gradients(scores, actions, reward)

        self.rl_action = action
        self.rl_reward = reward
        self.rl_gradients = gradients

        return (action, reward), (gradients,)


def infer_trial(env, utterance, probs, generative_model, args):
    """
    Run RSA inference for a given trial.

    Args:
        env:
        utterance:
        probs: Literal listener model LF probabilities
        generative_model:
        args:

    Returns:
        lfs: LF IDs sampled from `probs`
        weights: Associated weight `p(z, u)` for each LF
    """

    # Sample multiple LFs from the predicted distribution.
    lfs = np.random.choice(len(probs), p=probs,
                           size=args.num_listener_samples)

    # Reweight using generative model.
    weights = []
    for lf in lfs:
        # Resolve referent.
        referent = env.resolve_lf_by_id(lf)
        if not referent:
            # Dereference failed.
            weights.append(-np.inf)
            continue
        referent = env._trial["domain"].index(referent[0])

        # Get p(z|r).
        g_lf_distr = env.get_generative_lf_probs(referent)
        # Sample from the distribution.
        g_lf = np.random.choice(len(g_lf_distr), p=g_lf_distr)

        # Record unnormalized score p(u, z)
        weight = generative_model.score(g_lf, utterance)
        weights.append(weight)

        # Debug logging.
        fn_id = lf // len(env.lf_atoms)
        atom_id = lf % len(env.lf_atoms)

        g_fn_id = g_lf // len(env.lf_atoms)
        g_atom_id = g_lf % len(env.lf_atoms)

        print("%s(%s) => %s => %s(%s) => %f" %
              (env.lf_function_from_id[fn_id][0], env.lf_atoms[atom_id],
               env._trial["domain"][referent]["attributes"][args.atom_attribute],
               env.lf_function_from_id[g_fn_id][0], env.lf_atoms[g_atom_id],
               weight))

    return lfs, weights


def run_trial(model, generative_model, train_op, env, sess, args):
    inputs = env.reset()

    partial_fetches = [model.probs, train_op]
    partial_feeds = [model.items, model.utterance,
                     model.rl_action, model.rl_reward]
    partial = sess.partial_run_setup(partial_fetches, partial_feeds)

    items, utterance = inputs
    prob_feeds = {model.items: items,
                  model.utterance: utterance}
    probs = sess.partial_run(partial, model.probs, prob_feeds)

    lfs, lf_weights = infer_trial(env, utterance, probs, generative_model, args)

    # TODO: Should generative model weights also receive REINFORCE gradients?

    # Now select action based on maximum generative score.
    lf_pred = max(zip(lfs, lf_weights), key=lambda el: el[1])[0]

    _, reward, _, _ = env.step(lf_pred)
    tqdm.write("%f\n" % reward)

    # Update recognition parameters.
    _ = sess.partial_run(partial, [train_op],
            {model.rl_action: lf_pred,
             model.rl_reward: reward})

    # Update generation parameters.
    if reward > 0:
        generative_model.observe(lf_pred, utterance)


def build_train_graph(model, env, args):
    model.build_rl_gradients()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.MomentumOptimizer(args.learning_rate, args.momentum)
    train_op = opt.apply_gradients(model.rl_gradients, global_step=global_step)

    # Make a dummy train_op that works with TF partial_run.
    with tf.control_dependencies([train_op]):
        train_op = tf.constant(0.0, name="dummy_train_op")

    return train_op, global_step


def analyze_weights(sess, env):
    descs = env.describe_features()
    weight_var = next(var for var in tf.trainable_variables()
                      if var.get_shape().as_list()[0] == len(descs))
    weight_var = sess.run(weight_var)

    weights = zip(descs, weight_var)
    weights = sorted(weights, key=lambda weight: -abs(weight[1]))

    print("%20s\t%10s\t%10s\t%s" % ("WORD", "KEY", "VALUE", "WEIGHT"))
    print("=" * 100)
    for (word, key, value), weight in weights[:500]:
        print("%20s\t%10s\t%10s\t%+.5f" % (word, key, value, weight))


def train(args):
    env = TUNAWithLoTEnv(args.corpus_path, corpus_selection=args.corpus_selection,
                         bag=args.bag_env, functions=FUNCTIONS[args.fn_selection],
                         atom_attribute=args.atom_attribute)
    model = ListenerModel(env)
    train_op, global_step = build_train_graph(model, env, args)
    generative_model = NaiveGenerativeModel()

    supervisor = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                                     summary_op=None)
    with supervisor.managed_session() as sess:
        for i in trange(args.num_trials):
            if supervisor.should_stop():
                break

            run_trial(model, generative_model, train_op, env, sess, args)

        if args.analyze_weights:
            analyze_weights(sess, env)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/tuna")
    p.add_argument("--analyze_weights", default=False, action="store_true")

    p.add_argument("--corpus_path", required=True)
    p.add_argument("--corpus_selection", default=None)
    p.add_argument("--fn_selection", default="spatial_simple",
                   choices=FUNCTIONS.keys())
    p.add_argument("--atom_attribute", default="shape")

    p.add_argument("--bag_env", default=False, action="store_true")
    p.add_argument("--item_repr_dim", type=int, default=64)
    p.add_argument("--utterance_repr_dim", type=int, default=64)

    p.add_argument("--num_listener_samples", type=int, default=5)

    p.add_argument("--num_trials", default=100, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)
    p.add_argument("--momentum", default=0.9, type=float)

    train(p.parse_args())
