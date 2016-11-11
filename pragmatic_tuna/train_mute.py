from argparse import ArgumentParser
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import trange, tqdm

from pragmatic_tuna.environments import TUNAEnv
from pragmatic_tuna.reinforce import reinforce_episodic_gradients


Model = namedtuple("Model", [# inputs for computation 1: action scores
                             "items", "utterance",
                             # inputs for computation 2: REINFORCE gradient
                             "action", "reward",
                             # outputs for computation 1: scores
                             "scores", "probs",
                             # outputs for computation 2: gradients
                             "gradients"])
def build_model(env, item_repr_dim=50, utterance_repr_dim=50):
    items = tf.placeholder(tf.float32, shape=(env.domain_size, env.attr_dim),
                           name="items")
    utterance = tf.placeholder(tf.float32, shape=(env.vocab_size,),
                               name="utterance")

    # Pretend that the first axis of `items` is a batch axis; library functions
    # work for us then
    item_reprs = layers.fully_connected(items, item_repr_dim, tf.tanh)

    # `utterance` is a sparse input; this acts like an inefficient embedding
    # lookup
    utterance_repr = layers.fully_connected(tf.expand_dims(utterance, 0),
                                            utterance_repr_dim, tf.identity)
    # one preprocessing layer
    utterance_repr = layers.fully_connected(utterance_repr, item_repr_dim,
                                            tf.tanh)

    scores = tf.squeeze(tf.matmul(item_reprs, tf.transpose(utterance_repr)))
    probs = tf.nn.softmax(scores)

    ###########

    action = tf.placeholder(tf.int32, shape=(), name="action_taken")
    reward = tf.placeholder(tf.float32, shape=(), name="reward")

    # Reshape -- reinforce_episodic_gradients expects batches and multiple
    # timesteps.
    scores = [tf.expand_dims(scores, 0)]
    actions = [action]
    rewards = reward
    gradients = reinforce_episodic_gradients(scores, actions, reward)

    return Model(items, utterance,
                 action, reward,
                 scores, probs,
                 gradients)


def run_trial(model, train_op, env, sess):
    items, utterance = env.reset()

    partial = sess.partial_run_setup([model.probs, train_op],
                                     [model.items, model.utterance,
                                      model.action, model.reward])
    probs = sess.partial_run(partial, model.probs,
                             {model.items: items, model.utterance: utterance})

    # Sample action from probability distribution.
    choice = np.random.choice(env.domain_size, p=probs)
    _, reward, _, _ = env.step(choice)
    tqdm.write("%f" % reward)

    # Update parameters.
    _ = sess.partial_run(partial, [train_op],
                         {model.action: choice, model.reward: reward})


def build_train_graph(model, env, args):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.MomentumOptimizer(args.learning_rate, args.momentum)
    train_op = opt.apply_gradients(model.gradients, global_step=global_step)

    # Make a dummy train_op that works with TF partial_run.
    with tf.control_dependencies([train_op]):
        train_op = tf.constant(0.0, name="dummy_train_op")

    return train_op, global_step


def train(args):
    env = TUNAEnv(args.corpus_path, args.corpus_selection)
    model = build_model(env, args.item_repr_dim, args.utterance_repr_dim)
    train_op, global_step = build_train_graph(model, env, args)

    supervisor = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                                     summary_op=None)
    with supervisor.managed_session() as sess:
        for i in trange(args.num_trials):
            if supervisor.should_stop():
                break

            run_trial(model, train_op, env, sess)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--logdir", default="/tmp/tuna")

    p.add_argument("--corpus_path", required=True)
    p.add_argument("--corpus_selection", default="furniture")

    p.add_argument("--item_repr_dim", type=int, default=64)
    p.add_argument("--utterance_repr_dim", type=int, default=64)

    p.add_argument("--num_trials", default=100, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)
    p.add_argument("--momentum", default=0.9, type=float)

    train(p.parse_args())
