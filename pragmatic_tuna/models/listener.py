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

    def sample(self, words, temperature=None,
               context_free=False, argmax=False, evaluating=False):
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


class EnsembledSkipGramListenerModel(EnsembledListenerModel):

    def __init__(self, env, n):
        self.env = env
        models = [SkipGramListenerModel(env, scope="listener%i" % i)
                  for i in range(n)]
        super(EnsembledSkipGramListenerModel, self).__init__(models)

    def observe(self, obs, lf_pred, reward, gold_lf):
        if gold_lf is None:
            return

        referent = self.env.resolve_lf(gold_lf)[0]
        for model in self.models:
            model._populate_cache(obs[1], context_free=True)

        # Build gold vector
        model = self.models[0]
        gold_lfs = np.zeros((len(model.lf_cache), 1))
        for i, lf in enumerate(model.lf_cache):
            lf = model.to_lot_lf(lf)
            if lf == gold_lf:
                gold_lfs[i] = 1.0
        gold_lfs /= np.sum(gold_lfs)

        train_feeds = {}
        for model in self.models:
            train_feeds.update({model.feats: model.feat_matrix,
                                model.gold_lfs: gold_lfs})

        sess = tf.get_default_session()
        sess.run(self.train_op, train_feeds)


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

    def sample(self, words, temperature=1.0, argmax=False,
               context_free=False, evaluating=False):
        ret_lfs, total_probs = self.sample_batch([words])
        return ret_lfs[0], total_probs[0]

    def sample_batch(self, words, temperature=1.0, argmax=False,
                     context_free=False, evaluating=False):
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

        # Pad LF with stop tokens
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


class SkipGramListenerModel(ListenerModel):
    """
        MaxEnt model p(z|u) with skip-gram features of utterance
        and logical form.
    """


    def __init__(self, env, scope="listener"):
        self.vocab_size = len(env.word2idx.keys())
        self.lf_vocab_size = len(env.lf_vocab)
        self.word_feat_count = self.vocab_size * (self.vocab_size + 1)
        self.lf_feat_count = self.lf_vocab_size * (self.lf_vocab_size + 1)

        self.feat_count = self.word_feat_count * self.lf_feat_count

        self.l1_reg = 0.0

        self.reset()
        super(SkipGramListenerModel, self).__init__(env, scope=scope)


        self.feed_cache = []

    def _build_graph(self):
        with self._scope:
            self.weights = tf.get_variable("weights", shape=(self.feat_count, 1),
                                             initializer=tf.constant_initializer(0))
            self.feats = tf.placeholder(tf.float32, shape=(None, self.feat_count))


            self.scores = tf.squeeze(tf.matmul(self.feats, self.weights), [1])
            self.probs = tf.nn.softmax(self.scores)



    """
        Convert internal LF of form left(dog,cat) to LoT expression
        id(cat) AND left(dog)

        Assumes at most 2 conjuncts.
    """
    def to_lot_lf(self, lf):
        #case 1: id(x)
        if len(lf) == 1:
            return (self.env.lf_token_to_id["id"], lf[0])
        #case 2: fn(x)
        elif len(lf) == 2:
            return lf
        #case 3:
        elif len(lf) == 3:
            return (self.env.lf_token_to_id["id"],
                    lf[2],
                    lf[0],
                    lf[1])

    """
        Convert LoT expression of the form id(cat) AND left(dog) to
        internal LF of form left(dog,cat)

        Assumes at most 2 conjuncts and that one of them has the form id(x).
    """

    def from_lot_lf(self, lot):

        id_idx = self.env.lf_token_to_id["id"]

        if len(lot) == 2:
            if lot[0] == id_idx:
                return lot[1:]
            else:
                return lot
        elif len(lot) == 4:
            if id_idx not in lot:
                print("%sfrom_lot_lf: Invalid LF.%s" % (colors.FAIL, colors.ENDC))
                return ()
            else:
                if lot[0] == id_idx:
                    #id(x) and fn(y)
                    return (lot[2], lot[3], lot[1])
                else:
                    #fn(x) and id(y)
                    return (lot[0], lot[1], lot[3])

        else:
            print("%sfrom_lot_lf: Invalid LF.%s" % (colors.FAIL, colors.ENDC))
            return ()

    def build_xent_gradients(self):
        """
        Assuming the client can determine some gold-standard LF for a given
        trial, we can simply train by cross-entropy (maximize log prob of the
        gold output).
        """

        self.gold_lfs = tf.placeholder(tf.float32, shape=(None))

        loss = tf.nn.softmax_cross_entropy_with_logits(tf.squeeze(self.scores), tf.squeeze(self.gold_lfs))
        loss += self.l1_reg * tf.reduce_sum(tf.abs(self.weights))

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gradients = zip(gradients, params)


        return (self.gold_lfs, None), (gradients,)


    def featurize_words(self, words):

        word_idxs = [self.env.word2idx[word] for word in words]

        word_idxs.append(self.env.word2idx[self.env.EOS])

        #unigrams
        skip_gram_idxs = word_idxs
        #bigrams and bi-skip-grams
        for i in range(len(words)-1):
            #bigram (add-1, 0=no word, i.e., unigram)
            idx = word_idxs[i] + self.vocab_size * (word_idxs[i+1]+1)
            skip_gram_idxs.append(idx)

        #trigrams and tri-skip-grams:
        #for i in range(len(words)-2):
            #trigram (add-1, 0=no word
            #idx = word_idxs[i] + self.vocab_size * (word_idxs[i+1]+1) + self.vocab_size * (self.vocab_size + 1) * (word_idxs[i+2]+1)
            #skip_gram_idxs.append(idx)
            #skip-gram (word * word)
            #idx = word_idxs[i] + self.vocab_size * (self.vocab_size + 1) * (word_idxs[i+2]+1)
            #skip_gram_idxs.append(idx)




        #turn into one-hot vectors
        word_feats = np.zeros((self.word_feat_count,))
        word_feats[skip_gram_idxs] = 1
        word_feats = word_feats.reshape((1, self.word_feat_count))

        return word_feats

    def featurize_lf(self, lf):
        lf_idxs = []
        lf_idxs.extend(lf)

        if len(lf) > 1:
            idx = lf_idxs[0] + self.lf_vocab_size * (lf_idxs[1]+1)
            lf_idxs.append(idx)

        #if len(lf) > 2:
            #trigram (add-1, 0=no token)
            #idx = lf_idxs[0] + self.lf_vocab_size * (lf_idxs[1]+1) + self.lf_vocab_size * (self.lf_vocab_size + 1) * (lf_idxs[2]+1)
            #lf_idxs.append(idx)

        #turn into one-hot vectors
        lf_feats = np.zeros(self.lf_feat_count)
        lf_feats[lf_idxs] = 1
        return lf_feats


    def _populate_cache(self, words, context_free=False):
        id_idx = self.env.lf_token_to_id["id"]

        # HACK: pre-allocate lf_feats cache using known size
        num_lfs = (len(self.env.lf_atoms) * len(self.env.lf_functions)) ** 2
        all_lf_feats = np.empty((num_lfs, self.lf_feat_count))

        #TODO: tune this number?
        i = 0
        seen = set()
        for lf_pref in self.env.enumerate_lfs(includeOnlyPossibleReferents=not context_free):
            for lf in self.env.enumerate_lfs(includeOnlyPossibleReferents=not context_free, lf_prefix=lf_pref):
                valid = len(lf) < 3 or id_idx in lf
                if not valid:
                    continue

                lf = self.from_lot_lf(lf)
                if lf in seen:
                    continue
                seen.add(lf)

                self.lf_cache.append(lf)
                all_lf_feats[i] = self.featurize_lf(lf)
                i += 1

        self.lf_feats = all_lf_feats[:i]

        word_feats = self.featurize_words(words)
        #take cross product
        self.feat_matrix = np.zeros((len(self.lf_cache), self.feat_count))
        for i in range(len(self.lf_cache)):
            lf_feats = self.lf_feats[i].reshape(self.lf_feat_count, 1)

            self.feat_matrix[i] = np.dot(lf_feats, word_feats).reshape((self.feat_count,))

            #print(feats)

        sess = tf.get_default_session()
        feed = {self.feats: self.feat_matrix}

        self.probs_cache = sess.run(self.probs, feed)

    #TODO: iteratively sample, i.e., start with single predicate and then only consider LFs with that predicate
    def sample(self, words, temperature=None,
               context_free=False, argmax=False, evaluating=False):
        if len(self.lf_cache) < 1:
            self._populate_cache(words, context_free=context_free)

        if argmax:
            idx = np.argmax(self.probs_cache)
        else:
            idx = np.random.choice(len(self.probs_cache), p=self.probs_cache)

        if evaluating:
            scores = [(self.probs_cache[i], self.env.describe_lf(self.to_lot_lf(lf)))
                      for i, lf in enumerate(self.lf_cache)]
            scores = sorted(scores, key=lambda pair: pair[0], reverse=True)
            print("\n".join("LF %30s  =>  (%.3g)" % (pair[1], pair[0])
                            for pair in scores))

        sampled_lf = self.to_lot_lf(self.lf_cache[idx])

        #print("####")
        #print(self.lf_cache[idx])
        #print(sampled_lf)
        #self.reset()
        return sampled_lf, self.probs_cache[idx]

    def reset(self):
        self.lf_cache = []
        self.lf_feats = None
        self.probs_cache = None


    def observe(self, obs, lf_pred, reward, gold_lf, batch=False):
        if gold_lf is None:
            return

        #print(gold_lf)

        referent = self.env.resolve_lf(gold_lf)[0]

        #word_feats = self.featurize_words(obs[2])
        #take cross product


        self._populate_cache(obs[1], context_free=True)


        #lf =  self.from_lot_lf(gold_lf)
        #lf_feats = self.featurize_lf(lf).reshape((self.lf_feat_count, 1))
        #feats = np.dot(lf_feats, word_feats).reshape((1,self.feat_count))

        #go through all LFs, check if they



        gold_lfs = np.zeros((len(self.lf_cache),1))
        for i, lf in enumerate(self.lf_cache):
            lf = self.to_lot_lf(lf)
            if lf == gold_lf:
                gold_lfs[i] = 1.0

            #matches = self.env.resolve_lf(lf)
            #if matches and len(matches) == 1 and matches[0] == referent:
            #    gold_lfs[i] = 1.0

        gold_lfs /= np.sum(gold_lfs)


        train_feeds = {self.feats: self.feat_matrix,
                       self.gold_lfs: gold_lfs}

        if batch:
            self.feed_cache.append(train_feeds)

        sess = tf.get_default_session()
        sess.run(self.train_op, train_feeds)

    def batch_observe(self):
        batch_size = len(self.feed_cache)
        lf_size = len(self.feed_cache[0][self.gold_lfs])
        feats = np.zeros((batch_size*lf_size, self.feat_count))

        gold_lfs = np.zeros((batch_size*lf_size,1))


        for i in range(batch_size):
            j = lf_size * i
            feats[j:j+lf_size] = self.feed_cache[i][self.feats]
            gold_lfs[j:j+lf_size] = self.feed_cache[i][self.gold_lfs]

        gold_lfs /= np.sum(gold_lfs)

        self.feed_cache = []

        train_feeds = {self.feats: feats,
                       self.gold_lfs: gold_lfs}


        sess = tf.get_default_session()
        for i in range(100):
            sess.run(self.train_op, train_feeds)
            print("Batch update: %d" % (i))

