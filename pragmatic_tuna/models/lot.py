from argparse import ArgumentParser
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import trange, tqdm

from pragmatic_tuna.environments import TUNAWithLoTEnv
from pragmatic_tuna.reinforce import reinforce_episodic_gradients


def above_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["y"] < candidate["attributes"]["y"]
def right_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["x"] < candidate["attributes"]["x"]
def left_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["x"] > candidate["attributes"]["x"]
def below_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["y"] > candidate["attributes"]["y"]

FUNCTIONS = {
    "spatial_simple": [
        ("above", above_fn)
    ],
    "spatial_complex": [
        ("above", above_fn),
        ("right", right_fn),
        ("left", left_fn),
        ("below", below_fn)
    ]
}


Model = namedtuple("Model", [# inputs for computation 1: action scores
                             "inputs",
                             # inputs for computation 2: REINFORCE gradient
                             "action", "reward",
                             # outputs for computation 1: scores
                             "scores", "probs",
                             # outputs for computation 2: gradients
                             "gradients"])
def build_model(env, item_repr_dim=50, utterance_repr_dim=50):
    n_outputs = len(env.lf_functions) + len(env.lf_atoms)
    if env.bag:
        input_shape = (env.domain_size, env.attr_dim * env.vocab_size)
        inputs = tf.placeholder(tf.float32, shape=input_shape)

        # TODO can't treat as a batch anymore.. !?!
        # don't know how this should be parameterized.
        scores = layers.fully_connected(inputs, n_outputs, tf.identity)
    else:
        items = tf.placeholder(tf.float32, shape=(None, env.attr_dim))
        utterance = tf.placeholder(tf.float32, shape=(env.vocab_size,))

        scores = layers.fully_connected(tf.expand_dims(utterance, 0),
                                        n_outputs, tf.identity)

    probs = tf.squeeze(tf.nn.softmax(scores))

    ###########

    action = tf.placeholder(tf.int32, shape=(), name="action")
    reward = tf.placeholder(tf.float32, shape=(), name="reward")

    scores = [scores]
    actions = [action]
    rewards = reward
    gradients = reinforce_episodic_gradients(scores, actions, reward)


    inputs = (inputs,) if env.bag else (items, utterance)
    return Model(inputs,
                 action, reward,
                 scores, probs,
                 gradients)


def run_trial(model, train_op, env, sess):
    inputs = env.reset()

    partial_fetches = [model.probs, train_op]
    partial_feeds = [model.action, model.reward]
    partial_feeds += list(model.inputs)
    partial = sess.partial_run_setup(partial_fetches, partial_feeds)

    if env.bag:
        prob_feeds = {model.inputs[0]: inputs}
    else:
        prob_feeds = {model.inputs[0]: inputs[0],
                      model.inputs[1]: inputs[1]}
    probs = sess.partial_run(partial, model.probs, prob_feeds)

    # Sample `fn(atom)` LF from the given distribution.
    choice = np.random.choice(len(probs), p=probs)

    _, reward, _, _ = env.step(choice)
    tqdm.write("%f\n" % reward)

    # Update parameters.
    _ = sess.partial_run(partial, [train_op],
            {model.action: choice,
             model.reward: reward})


def build_train_graph(model, env, args):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.MomentumOptimizer(args.learning_rate, args.momentum)
    train_op = opt.apply_gradients(model.gradients, global_step=global_step)

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
    model = build_model(env, args.item_repr_dim, args.utterance_repr_dim)
    train_op, global_step = build_train_graph(model, env, args)

    supervisor = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                                     summary_op=None)
    with supervisor.managed_session() as sess:
        for i in trange(args.num_trials):
            if supervisor.should_stop():
                break

            run_trial(model, train_op, env, sess)

        if args.analyze_weights:
            analyze_weights(sess, env)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/tuna")
    p.add_argument("--analyze_weights", default=False, action="store_true")

    p.add_argument("--corpus_path", required=True)
    p.add_argument("--corpus_selection", default="furniture")
    p.add_argument("--fn_selection", default="spatial_simple",
                   choices=FUNCTIONS.keys())
    p.add_argument("--atom_attribute", default="shape")

    p.add_argument("--bag_env", default=False, action="store_true")
    p.add_argument("--item_repr_dim", type=int, default=64)
    p.add_argument("--utterance_repr_dim", type=int, default=64)

    p.add_argument("--num_trials", default=100, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)
    p.add_argument("--momentum", default=0.9, type=float)

    train(p.parse_args())
