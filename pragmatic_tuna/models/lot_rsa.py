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
    A very stupid generative utterance model $p(u | z)$ which is intended to
    map from bag-of-features $z$ representations to bag-of-words $u$
    representations. Optionally performs add-1 smoothing.
    """

    smooth_val = 0.1

    def __init__(self, vocab_size, max_length, smooth=True):
        self.smooth = smooth
        self.counter = defaultdict(lambda: Counter())
        self.vocab_size = vocab_size
        self.max_length = max_length

    def observe(self, z, u):
        z, u = z, tuple(u)
        self.counter[z][u] += 1

    def score(self, z, u):
        """Retrieve unnormalized p(u|z)"""
        # TODO: weight on Z?
        z, u = z, tuple(u)
        score = self.counter[z][u]
        if self.smooth:
            # Add-1 smoothing.
            score += self.smooth_val
        return np.exp(score)

    def sample(self, z):
        """Sample from the distribution p(u|z)"""
        g_dict = self.counter[z]
        keys = list(g_dict.keys())
        values = np.array(list(g_dict.values()))

        if self.smooth:
            # Allow that we might sample one of the unseen u's.
            mass_seen = np.exp(values + self.smooth_val).sum()
            n_unseen = 2 ** self.max_length - len(g_dict)
            mass_unseen = np.exp(self.smooth_val) * (n_unseen)
            p_unseen = mass_unseen / (mass_unseen + mass_seen)

            if np.random.random() < p_unseen:
                print("Rejection!")
                # Rejection-sample a random unseen utterance.
                done = False
                while not done:
                    length = np.random.randint(1, self.max_length + 1)
                    idxs = np.random.randint(self.vocab_size, size=length)
                    utt = np.zeros(self.vocab_size)
                    utt[idxs] = 1
                    utt = tuple(utt)
                    done = utt not in keys
                return utt

        distr = np.exp(values - values.max())
        distr /= distr.sum()
        return keys[np.random.choice(len(keys), p=distr)]


class ListenerModel(object):

    """
    Parametric listener model $q_\\theta(z|u)$ which maps utterances to LF
    representations.
    """

    def __init__(self, env, scope="listener"):
        assert not env.bag
        self.env = env
        self._scope = tf.variable_scope(scope)
        self.feeds = []

        self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def build_rl_gradients(self):
        raise NotImplementedError

    def build_xent_gradients(self):
        raise NotImplementedError


class SimpleListenerModel(ListenerModel):

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

        self.feeds.extend([self.items, self.utterance])

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

        self.feeds.extend([self.rl_action, self.rl_reward])

        return (action, reward), (gradients,)

    def build_xent_gradients(self):
        """
        Assuming the client can determine some gold-standard LF for a given
        trial, we can simply train by cross-entropy (maximize log prob of the
        gold output).
        """
        gold_lf = tf.placeholder(tf.int32, shape=(), name="gold_lf")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                tf.squeeze(self.scores), gold_lf)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_lf = gold_lf
        self.xent_gradients = zip(gradients, params)

        self.feeds.extend([self.xent_gold_lf])

        return (gold_lf,), (gradients,)

    def sample(self, utterance_bag, words):
        sess = tf.get_default_session()
        probs = sess.run(self.probs, {self.utterance: utterance_bag})
        return np.random.choice(len(probs), p=probs)

    def observe(self, obs, lf_pred, reward, gold_lf, train_op):
        if hasattr(self, "rl_action"):
            train_feeds = {self.utterance: obs[1],
                           self.rl_action: lf_pred,
                           self.rl_reward: reward}
        elif hasattr(self, "xent_gold_lf"):
            if gold_lf is None:
                # TODO log?
                return
            train_feeds = {self.utterance: obs[1],
                           self.xent_gold_lf: gold_lf}
        else:
            raise RuntimeError("no gradients defined")

        sess = tf.get_default_session()
        sess.run(train_op, train_feeds)


class WindowedSequenceListenerModel(ListenerModel):

    """
    Parametric listener model $q_\\theta(z|u) which maps utterances (sequences)
    to LF representations (factored as sequences).

    This model takes a window of embedding inputs and outputs a sequence of LF
    tokens.
    """

    def __init__(self, env, scope="listener", max_timesteps=4, embedding_dim=10):
        super(SequenceListenerModel, env, scope=scope)
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

    def _build_graph(self):
        with self._scope:
            # TODO: padding representation?
            self.words = tf.placeholder(tf.int32, shape=(self.max_timesteps,),
                                        name="word_%i" % t)

            emb_shape = (env.vocab_size, self.embedding_dim)
            word_embeddings = tf.get_variable("word_embeddings", shape=emb_shape)

            word_window = tf.nn.embedding_lookup(word_embeddings, self.words)
            word_window = tf.flatten(word_window)

            # Create embeddings for LF tokens + 1 null/stop token
            n_lf_embeddings = len(self.env.lf_functions) + len(self.env.lf_atoms) + 1
            lf_emb_shape = (n_lf_embeddings, self.embedding_dim)
            lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape)
            null_embedding = tf.gather(lf_embeddings, 0)

            # Now run a teeny LF decoder.
            outputs, samples = [], []
            output_dim = n_lf_embeddings
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=i > 0):
                    input_t = tf.pack([prev_sample, word_window])
                    output_t = layers.fully_connected(input_t, output_dim,
                                                      tf.identity)

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.multinomial(output_t)
                    prev_sample = tf.nn.embedding_lookup(lf_embeddings, sample_t)

                    outputs.append(output_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.samples = samples

    def build_xent_gradients(self):
        gold_lf_tokens = [tf.zeros((), dtype=tf.int32, name="gold_lf_%i" % t)
                          for t in range(self.max_timesteps)]
        gold_lf_length = tf.placeholder(tf.int32, shape=(), name="gold_lf_length")
        losses = [tf.to_float(t < gold_lf_length) *
                  tf.nn.sparse_softmax_cross_entropy_with_logits(
                          tf.squeeze(output_t), gold_lf_t)
                  for t, (output_t, gold_lf_t)
                  in enumerate(zip(self.outputs, gold_lf_tokens))]
        loss = tf.add_n(losses) / gold_lf_length

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_lf_tokens = gold_lf_tokens
        self.xent_gold_lf_length = gold_lf_length
        self.xent_gradients = zip(gradients, params)

        self.feeds.extend([self.xent_gold_lf_tokens, self.xent_gold_lf_length])

        return ((self.xent_gold_lf_tokens, self.xent_gold_lf_length),
                (self.xent_gradients,))


def infer_trial(env, obs, listener_model, speaker_model, args):
    """
    Run RSA inference for a given trial.

    Args:
        env:
        obs: Observation from environment
        listener_model:
        speaker_model:
        args:

    Returns:
        lfs: LF IDs sampled from `listener_model`
        weights: Associated weight `p(z, u)` for each LF
    """

    items, utterance_bag, words = obs

    # Sample N LFs from the predicted distribution, accepting only when they
    # resolve to a referent in the scene.
    lfs, weights = [], []
    while len(weights) < args.num_listener_samples:
        lf = listener_model.sample(utterance_bag, words)

        # Resolve referent.
        referent = env.resolve_lf_by_id(lf)
        if not referent:
            # Dereference failed. No object matched.
            continue
        referent = env._trial["domain"].index(referent[0])

        # Get p(z|r).
        g_lf_distr = env.get_generative_lf_probs(referent)
        # Sample from the distribution.
        g_lf = np.random.choice(len(g_lf_distr), p=g_lf_distr)

        # Record unnormalized score p(u, z)
        weight = speaker_model.score(g_lf, utterance_bag)

        lfs.append(lf)
        weights.append(weight)

        # Debug logging.
        print("LF %20s  =>  Referent %10s  =>  Gen LF %20s  =>  %f" %
              (env.describe_lf_by_id(lf),
               env._trial["domain"][referent]["attributes"][args.atom_attribute],
               env.describe_lf_by_id(g_lf),
               weight))

    return lfs, weights


def run_listener_trial(listener_model, speaker_model, listener_train_op,
                       env, sess, args):
    """
    Run single recognition trial.

    1. Predict a referent give an utterance.
    2. Update listener model weights.
    3. Update speaker model weights.
    """
    env.configure(dreaming=False)
    obs = env.reset()

    lfs, lf_weights = infer_trial(env, obs, listener_model, speaker_model, args)
    lfs = sorted(zip(lfs, lf_weights), key=lambda el: el[1], reverse=True)

    # Now select action based on maximum generative score.
    lf_pred = lfs[0][0]
    _, reward, _, _ = env.step(lf_pred)
    tqdm.write("%f" % reward)

    # Find the highest-scoring LF that dereferences to the correct referent.
    gold_lf, gold_lf_pos = None, -1
    for i, (lf_i, weight_i) in enumerate(lfs):
        resolved = env.resolve_lf_by_id(lf_i)
        if resolved and resolved[0]["target"]:
            gold_lf = lf_i
            gold_lf_pos = i
            break
    if gold_lf is not None:
        print("gold", env.describe_lf_by_id(gold_lf), gold_lf_pos)

    # Update listener parameters.
    listener_model.observe(obs, lf_pred, reward, gold_lf, listener_train_op)

    # Update speaker parameters.
    speaker_model.observe(gold_lf, obs[1])


def run_dream_trial(model, generative_model, env, sess, args):
    """
    Run a single dream trial.
    """
    env.configure(dreaming=True)
    inputs = env.reset()

    referent_idx = [i for i, referent in enumerate(env._trial["domain"])
                    if referent["target"]][0]
    g_lf_distr = env.get_generative_lf_probs(referent_idx)

    success = False
    i = 0
    while not success:
        print("Dream trial %i" % i)

        # Sample an LF from p(z|r).
        g_lf = np.random.choice(len(g_lf_distr), p=g_lf_distr)

        # Sample utterances from p(u|z).
        g_ut = generative_model.sample(g_lf)
        words = [env.vocab[idx] for idx, count in enumerate(g_ut)
                if count > 0]

        # Run listener model q(z|u).
        probs = sess.run(model.probs, {model.utterance: g_ut})
        # Literally dereference and see if we get the expected referent.
        # TODO: run multiple particles through this whole process!
        l_lf = np.random.choice(len(probs), p=probs)
        l_referent = env.resolve_lf_by_id(l_lf)
        if l_referent:
            l_referent_id = env._trial["domain"].index(l_referent[0])
            success = l_referent_id == referent_idx

        print(
"""    Referent:\t\t%s
    z ~ p(z|r):\t\t%s
    u ~ p(u|z):\t\t%s
    z' ~ q(z|u):\t%s
    Deref:\t\t%s""" %
              (env._trial["domain"][referent_idx],
               env.describe_lf_by_id(g_lf),
               " ".join([env.vocab[idx] for idx, count in enumerate(g_ut) if count]),
               env.describe_lf_by_id(l_lf),
               l_referent))

        i += 1


def build_train_graph(model, env, args):
    if args.learning_method == "rl":
        model.build_rl_gradients()
        gradients = model.rl_gradients
    elif args.learning_method == "xent":
        model.build_xent_gradients()
        gradients = model.xent_gradients
    else:
        raise NotImplementedError("undefined learning method " + args.learning_method)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.MomentumOptimizer(args.learning_rate, args.momentum)
    train_op = opt.apply_gradients(gradients, global_step=global_step)

    # Make a dummy train_op that works with TF partial_run.
    with tf.control_dependencies([train_op]):
        train_op = tf.constant(0.0, name="dummy_train_op")

    return train_op, global_step


def train(args):
    env = TUNAWithLoTEnv(args.corpus_path, corpus_selection=args.corpus_selection,
                         bag=args.bag_env, functions=FUNCTIONS[args.fn_selection],
                         atom_attribute=args.atom_attribute)
    model = SimpleListenerModel(env)
    train_op, global_step = build_train_graph(model, env, args)
    generative_model = NaiveGenerativeModel(env.vocab_size, 3) # TODO fixed

    supervisor = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                                     summary_op=None)
    with supervisor.managed_session() as sess:
        with sess.as_default():
            for i in trange(args.num_trials):
                if supervisor.should_stop():
                    break

                tqdm.write("\n===========\nLISTENER TRIAL\n===========")
                run_listener_trial(model, generative_model, train_op, env, sess, args)

                if args.dream:
                    tqdm.write("\n===========\nDREAM TRIAL\n============")
                    run_dream_trial(model, generative_model, env, sess, args)


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

    p.add_argument("--dream", default=False, action="store_true")
    p.add_argument("--num_listener_samples", type=int, default=5)

    p.add_argument("--learning_method", default="rl", choices=["rl", "xent"])
    p.add_argument("--num_trials", default=100, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)
    p.add_argument("--momentum", default=0.9, type=float)

    train(p.parse_args())
