"""
Defines discriminative listener models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers

from pragmatic_tuna.reinforce import reinforce_episodic_gradients
from pragmatic_tuna.util import orthogonal_initializer


EMBEDDING_INITIALIZER = orthogonal_initializer()
LF_EMBEDDING_INITIALIZER = orthogonal_initializer()


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
        self.train_op = None

        self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def build_rl_gradients(self):
        raise NotImplementedError

    def build_xent_gradients(self):
        raise NotImplementedError

    def sample(self, words, temperature=None, argmax=False, evaluating=False):
        """
        Returns:
            lf: LF token ID sequence
            p_lf: float scalar p(lf)
        """
        raise NotImplementedError

    def observe(self, obs, lf_pred, reward, gold_lf):
        raise NotImplementedError

    def score_batch(self, words, lfs):
        """
        Evaluate p(z|u) for a batch of word sequence inputs and sequence LF
        outputs.
        """
        raise NotImplementedError

    def reset(self):
        pass


class EnsembledListenerModel(ListenerModel):

    def __init__(self, models):
        self.models = models

    def build_xent_gradients(self):
        gradients = []
        for model in self.models:
            model.build_xent_gradients()
            gradients.extend(model.xent_gradients)

        self.xent_gradients = gradients

    def sample(self, words, **kwargs):
        # TODO if argmax, should sample from all models
        model = self.models[np.random.choice(len(self.models))]
        return model.sample(words, **kwargs)

    def observe(self, obs, lf_pred, reward, gold_lf):
        raise NotImplementedError

    def reset(self):
        for model in self.models:
            model.reset()


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

    def _list_to_id(self, id_list):
        """
        Convert an LF token ID list to an action ID.
        """
        fn_tok_id, atom_tok_id = id_list

        # Convert ID sequence back to our hacky space.
        fn_name = self.env.lf_vocab[fn_tok_id]
        fn_id = self.env.lf_functions.index(fn_name)
        atom = self.env.lf_vocab[atom_tok_id]
        atom_id = self.env.lf_atoms.index(atom)

        action_id = fn_id * len(self.env.lf_atoms) + atom_id
        return action_id

    def _id_to_list(self, idx):
        """
        Convert an action ID to an LF token ID list.
        """
        fn_id = idx // len(self.env.lf_atoms)
        atom_id = idx % len(self.env.lf_atoms)
        fn_name = self.env.lf_functions[fn_id]
        atom_name = self.env.lf_atoms[atom_id]
        token_ids = [self.env.lf_token_to_id[fn_name],
                     self.env.lf_token_to_id[atom_name]]
        return token_ids

    def sample(self, utterance_bag, words):
        raise NotImplementedError("does not implement new observation formatting. Manually convert to bag-of-words.")
        sess = tf.get_default_session()
        probs = sess.run(self.probs, {self.utterance: utterance_bag})
        lf = np.random.choice(len(probs), p=probs)

        # Jump through some hoops to make sure we sample a valid fn(atom) LF
        return self._id_to_list(lf)

    def observe(self, obs, lf_pred, reward, gold_lf):
        raise NotImplementedError("does not implement new observation formatting. Manually convert to bag-of-words.")
        lf_pred = self._list_to_id(lf_pred)
        if gold_lf is not None:
            gold_lf = self._list_to_id(gold_lf)

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
        sess.run(self.train_op, train_feeds)


class WindowedSequenceListenerModel(ListenerModel):

    """
    Parametric listener model $q_\\theta(z|u) which maps utterances (sequences)
    to LF representations (factored as sequences).

    This model takes a window of embedding inputs and outputs a sequence of LF
    tokens.
    """

    def __init__(self, env, scope="listener", max_timesteps=2, embedding_dim=10):
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim
        super(WindowedSequenceListenerModel, self).__init__(env, scope=scope)

    def _build_graph(self):
        with self._scope:
            self.temperature = tf.constant(1.0, name="sampling_temperature")

            # TODO: padding representation?
            self.words = tf.placeholder(tf.int32, shape=(None, self.max_timesteps,),
                                        name="words")

            emb_shape = (self.env.vocab_size, self.embedding_dim)
            word_embeddings = tf.get_variable(
                    "word_embeddings", shape=emb_shape, initializer=EMBEDDING_INITIALIZER)

            word_window = tf.nn.embedding_lookup(word_embeddings, self.words)
            word_window = tf.reshape(word_window, (-1, self.embedding_dim * self.max_timesteps))

            # Create embeddings for LF tokens
            lf_emb_shape = (len(self.env.lf_vocab), self.embedding_dim)
            lf_embeddings = tf.get_variable(
                    "lf_embeddings", shape=lf_emb_shape,
                    initializer=LF_EMBEDDING_INITIALIZER)
            batch_size = tf.shape(self.words)[0]
            null_embedding = tf.tile(
                    tf.expand_dims(tf.gather(lf_embeddings, self.env.lf_unk_id), 0),
                    (batch_size, 1))

            # Weight matrices mapping input -> ~p(fn), input -> ~p(atom)
            input_dim = self.embedding_dim + self.embedding_dim * self.max_timesteps
            W_fn = tf.get_variable("W_fn", shape=(input_dim, len(self.env.lf_functions)))
            W_atom = tf.get_variable("W_atom", shape=(input_dim, len(self.env.lf_atoms)))

            # Now run a teeny LF decoder.
            outputs, probs, samples = [], [], []
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(1, [prev_sample, word_window])

                    # Force-sample a syntactically valid LF.
                    # i.e. alternating fn,atom,fn,atom,...
                    #
                    # TODO: This is coupled with the ordering of the LF tokens
                    # in the env definition. That could be bad.
                    if t % 2 == 0:
                        fn_logits = tf.matmul(input_t, W_fn) / self.temperature
                        atom_logits = tf.zeros((batch_size, len(self.env.lf_atoms)))
                        sample_t = tf.multinomial(fn_logits, num_samples=1)
                        fn_probs = tf.nn.softmax(fn_logits)
                        atom_probs = atom_logits
                    else:
                        fn_logits = tf.zeros((batch_size, len(self.env.lf_functions)))
                        atom_logits = tf.matmul(input_t, W_atom) / self.temperature
                        sample_t = tf.multinomial(atom_logits, num_samples=1)
                        fn_probs = fn_logits
                        atom_probs = tf.nn.softmax(atom_logits)

                    output_t = tf.concat(1, (fn_logits, atom_logits),
                                         name="output_%i" % t)
                    probs_t = tf.concat(1, (fn_probs, atom_probs),
                                        name="probs_%i" % t)

                    sample_t = tf.squeeze(sample_t, [1])
                    if t % 2 == 1:
                        # Shift index to match standard vocabulary.
                        sample_t = tf.add(sample_t, len(self.env.lf_functions),
                                          name="sample_%i" % t)

                    # Hack shape.
                    prev_sample = tf.nn.embedding_lookup(lf_embeddings, sample_t)

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.word_embeddings = word_embeddings
            self.lf_embeddings = lf_embeddings

            self.outputs = outputs
            self.probs = probs
            self.samples = samples

    def build_xent_gradients(self):
        gold_lf_tokens = [tf.placeholder(tf.int32, shape=(None,),
                                         name="gold_lf_%i" % t)
                          for t in range(self.max_timesteps)]
        gold_lf_length = tf.placeholder(tf.int32, shape=(None,),
                                        name="gold_lf_length")

        losses = []
        for t in range(self.max_timesteps):
            xent_t = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    self.outputs[t], gold_lf_tokens[t])
            mask = tf.to_float(t < gold_lf_length)

            # How many examples are still active?
            num_valid = tf.reduce_sum(mask)
            # Calculate mean xent over examples.
            mean_xent = tf.reduce_sum(mask * xent_t) / num_valid
            losses.append(mean_xent)

        loss = tf.add_n(losses) / float(len(losses))

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_lf_tokens = gold_lf_tokens
        self.xent_gold_lf_length = gold_lf_length
        self.xent_gradients = zip(gradients, params)

        self.feeds.extend([self.xent_gold_lf_tokens, self.xent_gold_lf_length])

        return ((self.xent_gold_lf_tokens, self.xent_gold_lf_length),
                (self.xent_gradients,))

    def _get_word_idxs(self, words):
        # Look up word indices. TODO: padding with something other than UNK..?
        word_idxs = [self.env.word2idx[word] for word in words]
        assert len(word_idxs) <= self.max_timesteps
        word_idxs += [self.env.word_unk_id] * (self.max_timesteps - len(word_idxs))
        return word_idxs

    def _pad_lf_idxs(self, lf_idxs):
        # # Look up LF indices. TODO: padding with something other than UNK..?
        # lf_idxs = [self.env.lf_token_to_id[lf_tok] for lf_tok in lf]
        assert len(lf_idxs) <= self.max_timesteps
        missing_conjuncts = (self.max_timesteps - len(lf_idxs)) / 2
        assert int(missing_conjuncts) == missing_conjuncts
        lf_idxs += [self.env.lf_eos_id, self.env.lf_unk_id] * int(missing_conjuncts)
        return lf_idxs

    def sample(self, words, temperature=1.0, argmax=False, evaluating=False):
        ret_lfs, total_probs = self.sample_batch([words])
        return ret_lfs[0], total_probs[0]

    def sample_batch(self, words, temperature=1.0, argmax=False, evaluating=False):
        # TODO handle argmax, evaluating
        batch_size = len(words)

        sess = tf.get_default_session()
        feed = {self.words: [self._get_word_idxs(words_i)
                             for words_i in words],
                self.temperature: temperature}

        rets = sess.run(self.samples + self.probs, feed)

        # Unpack.
        sample_toks = rets[:len(self.samples)]
        probs = rets[len(self.samples):]

        # Calculate sequence probability as batch.
        done = np.zeros(batch_size)
        total_probs = np.ones(batch_size)
        batch_range = np.arange(batch_size)
        ret_lfs = [[] for _ in range(batch_size)]
        for t, (samples_t, probs_t) in enumerate(zip(sample_toks, probs)):
            total_probs *= probs_t[batch_range, samples_t]
            done = np.logical_or(done, samples_t == self.env.lf_eos_id)
            for i, sample_t_i in enumerate(samples_t):
                if not done[i]:
                    ret_lfs[i].append(sample_t_i)

        return ret_lfs, total_probs

    def observe(self, obs, lf_pred, reward, gold_lf):
        if gold_lf is None:
            return

        # Pad gold output LF.
        real_length = min(self.max_timesteps, len(gold_lf) + 1) # train to output a single stop token
        gold_lf = self._pad_lf_idxs(gold_lf)

        word_idxs = self._get_word_idxs(obs[1])
        feed = {self.words: [word_idxs],
                self.xent_gold_lf_length: [real_length]}
        feed.update({self.xent_gold_lf_tokens[t]: [gold_lf[t]]
                     for t in range(self.max_timesteps)})
        feed.update({self.samples[t]: [lf_t]
                     for t, lf_t in enumerate(gold_lf)})

        sess = tf.get_default_session()
        sess.run(self.train_op, feed)

    def score_batch(self, words, lfs):
        words = [self._get_word_idxs(words_i) for words_i in words]

        lfs = [self._pad_lf_idxs(lf_i) for lf_i in lfs]
        # Transpose to num_timesteps * batch_size
        lfs = np.array(lfs).T

        feed = {self.words: np.array(words)}
        feed.update({self.samples[t]: lfs_t
                     for t, lfs_t in enumerate(lfs)})

        sess = tf.get_default_session()
        probs = sess.run(self.probs, feed)

        # Index the sampled word for each example at each timestep
        batch_size = len(words)
        probs = [probs_t[np.arange(batch_size), lf_sample_t]
                 for probs_t, lf_sample_t in zip(probs, lfs)]

        return np.prod(probs, axis=0)
