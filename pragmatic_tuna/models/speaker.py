"""
Defines generative speaker models.
"""

from collections import Counter, defaultdict
from itertools import permutations

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers


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

    def observe(self, obs, gold_lf):
        if gold_lf is None:
            return

        u = obs[1]
        z = gold_lf

        z, u = tuple(z), tuple(u)
        self.counter[z][u] += 1

    def score(self, z, u, u_seq):
        """Retrieve unnormalized p(u|z)"""
        # TODO: weight on Z?
        z, u = tuple(z), tuple(u)
        score = self.counter[z][u]
        if self.smooth:
            # Add-1 smoothing.
            score += self.smooth_val
        return np.exp(score)

    def sample(self, z):
        """Sample from the distribution p(u|z)"""
        g_dict = self.counter[tuple(z)]
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


class DiscreteGenerativeModel(object):

    """
    A generative model that maps atoms and functions of a logical form
    z to and words of an utterance u and scores the fluency of the utterance
    using a bigram language model.
    """

    smooth_val = 0.1
    unk_prob = 0.01
    #if set to 0, use +1 smoothing of bigrams instead of backoff LM
    backoff_factor = 0
    distortion_prob = 0.5

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, env, max_timesteps=4, smooth=True):
        self.smooth = smooth
        self.vocab_size = env.vocab_size
        self.max_conjuncts = max_timesteps / 2
        self.env = env

        self.counter = defaultdict(lambda: Counter())
        self.bigramcounter = defaultdict(lambda: Counter())
        self.unigramcounter = Counter()

    def observe(self, obs, gold_lf):
        if gold_lf is None:
            return

        u = obs[2]
        z = gold_lf

        for lf_token in z:
            self.counter[lf_token].update(u)

        words = []
        words.extend(u)
        #words.append(self.END_TOKEN)

        prev_word = self.START_TOKEN
        for word in words:
            self.bigramcounter[prev_word][word] +=1
            self.unigramcounter[word] +=1
            prev_word = word

    def _score_word_atom(self, word, atom):
        score = self.counter[atom][word]
        denom = sum(self.counter[atom].values())
        if self.smooth:
            score += 1
            denom += len(self.env.vocab)
        return float(score) / denom

    def _score_words_atom(self, words, atom):
        """Return p(word | atom) for a collection `words`"""
        counter = self.counter[atom]

        score_delta = 1 if self.smooth else 0
        scores = [counter[word] + score_delta for word in words]

        denom = sum(counter.values())
        if self.smooth:
            denom += len(self.env.vocab)

        return np.array(scores, dtype=np.float32) / denom

    def _score_bigram(self, w1, w2):
        score = self.bigramcounter[w1][w2]
        denom = sum(self.bigramcounter[w1].values())
        if self.smooth and self.backoff_factor == 0:
          score += 1
          denom += len(self.env.vocab)
        if score < 1 :
            return 0
        return np.log(float(score) / denom)


    def _score_unigram(self, w):
        score = self.unigramcounter[w]
        if score < 1:
          prob = self.unk_prob
        else:
          prob = float(score) / sum(self.unigramcounter.values())

        return np.log(prob)


    def _score_sequence(self, u):
        prev_word = self.START_TOKEN

        words = []
        words.extend(u)
        #words.append(self.END_TOKEN)
        prob = 0
        for word in u:
            p_bigram = self._score_bigram(prev_word, word)
            p_bigram = ((p_bigram + np.log(1-self.backoff_factor))
                            if p_bigram < 0
                            else self._score_unigram(word) + np.log(self.backoff_factor))
            prob += p_bigram
            prev_word = word

        return prob

    def score(self, z, u_bag, u):
        # Limit utterance lengths to LF length.
        if len(u) != len(z):
            return -np.Inf
        #compute translation probability p(u|z)
        words = u
        alignments = permutations(range(len(z)))

        p_trans = 0
        for a in alignments:
            n_distortions = sum(abs(a[i] - i) for i in range(len(a)))
            p = self.distortion_prob ** n_distortions

            pairs = []
            for i, w in enumerate(words):
                p *= self._score_word_atom(w, z[a[i]])
                pairs.append((w, self.env.lf_vocab[z[a[i]]]))
            p_trans += p

        p_trans = np.log(p_trans)

        #compute fluency probability, i.e., lm probability
        p_seq  = self._score_sequence(u)

        return 0.1 * p_seq + 0.9 * p_trans

    def sample_with_alignment(self, z, alignment):
        unigram_denom = max(sum(self.unigramcounter.values()), 1.0)
        unigram_probs = np.array(list(self.unigramcounter.values())) / unigram_denom
        keys = list(self.unigramcounter.keys())

        prev_word = self.START_TOKEN

        u = []
        ps = []
        i = 0
        for i in range(len(z)):

            bigram_counts = np.array([self.bigramcounter[prev_word][w]
                                        for w in keys])
            bigram_denom = max(sum(bigram_counts), 1.0)
            bigram_probs = bigram_counts / bigram_denom

            cond_probs = self._score_words_atom(keys, z[alignment[i]])

            interpolated = bigram_probs * (1 - self.backoff_factor) + unigram_probs * self.backoff_factor
            distr = interpolated * cond_probs
            distr = distr / np.sum(distr)

            idx = np.random.choice(len(keys), p=distr)
            word = keys[idx]
            u.append(word)
            prev_word = word
            ps.append(distr[idx])

        p = np.exp(np.sum(np.log(ps)))
        return " ".join(u), p

    def sample(self, z):
        alignments = permutations(range(len(z)))
        utterances = []
        distr = []
        for a in alignments:
            u, p = self.sample_with_alignment(z, a)
            utterances.append(u)
            distr.append(p)

        distr = np.array(distr) / np.sum(distr)
        idx = np.random.choice(len(utterances), p=distr)
        return utterances[idx]


class WindowedSequenceSpeakerModel(object):

    """
    Windowed sequence speaker/decoder model that mirrors
    `WindowedSequenceListenerModel`.
    """
    # could probably be unified with WindowedSequenceListenerModel if there is
    # sufficient motivation.

    def __init__(self, env, scope="speaker", max_timesteps=4,
                 word_embeddings=None, lf_embeddings=None, embedding_dim=10):
        self.env = env
        self._scope_name = scope
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim

        self.train_op = None

        self._build_embeddings(word_embeddings, lf_embeddings)
        self._build_graph()

    def _build_embeddings(self, word_embeddings, lf_embeddings):
        with tf.variable_scope(self._scope_name):
            emb_shape = (self.env.vocab_size, self.embedding_dim)
            if word_embeddings is None:
                word_embeddings = tf.get_variable("word_embeddings", emb_shape)
            assert tuple(word_embeddings.get_shape().as_list()) == emb_shape

            lf_emb_shape = (len(self.env.lf_vocab), self.embedding_dim)
            if lf_embeddings is None:
                lf_embeddings = tf.get_variable("lf_embeddings", shape=lf_emb_shape)
            assert tuple(lf_embeddings.get_shape().as_list()) == lf_emb_shape

            self.word_embeddings = word_embeddings
            self.lf_embeddings = lf_embeddings

    def _build_graph(self):
        with tf.variable_scope(self._scope_name):
            self.lf_toks = tf.placeholder(tf.int32, shape=(self.max_timesteps,),
                                          name="lf_toks")

            lf_window = tf.nn.embedding_lookup(self.lf_embeddings, self.lf_toks)
            lf_window = tf.reshape(lf_window, (-1,))

            null_embedding = tf.gather(self.word_embeddings, self.env.word_unk_id)

            # Now run a teeny utterance decoder.
            outputs, probs, samples = [], [], []
            output_dim = self.env.vocab_size
            prev_sample = null_embedding
            for t in range(self.max_timesteps):
                with tf.variable_scope("recurrence", reuse=t > 0):
                    input_t = tf.concat(0, [prev_sample, lf_window])
                    output_t = layers.fully_connected(tf.expand_dims(input_t, 0),
                                                      output_dim, tf.identity)
                    probs_t = tf.squeeze(tf.nn.softmax(output_t))

                    # Sample an LF token and provide as feature to next timestep.
                    sample_t = tf.squeeze(tf.multinomial(output_t, num_samples=1))
                    prev_sample = tf.nn.embedding_lookup(self.word_embeddings, sample_t)

                    # TODO support stop token here?

                    outputs.append(output_t)
                    probs.append(probs_t)
                    samples.append(sample_t)

            self.outputs = outputs
            self.probs = probs
            self.samples = samples

    def build_xent_gradients(self):
        gold_words = [tf.zeros((), dtype=tf.int32, name="gold_word_%i" % t)
                      for t in range(self.max_timesteps)]
        gold_length = tf.placeholder(tf.int32, shape=(), name="gold_length")
        losses = [tf.to_float(t < gold_length) *
                  tf.nn.sparse_softmax_cross_entropy_with_logits(
                          tf.squeeze(output_t), gold_word_t)
                  for t, (output_t, gold_word_t)
                  in enumerate(zip(self.outputs, gold_words))]
        loss = tf.add_n(losses) / tf.to_float(gold_length)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        self.xent_gold_words = gold_words
        self.xent_gold_length = gold_length
        self.xent_gradients = zip(gradients, params)

        return ((self.xent_gold_words, self.xent_gold_length),
                (self.xent_gradients,))

    def _pad_lf_idxs(self, z):
        assert len(z) <= self.max_timesteps
        z_new = z + [self.env.lf_unk_id] * (self.max_timesteps - len(z))
        return z_new

    def sample(self, z):
        sess = tf.get_default_session()
        feed = {self.lf_toks: self._pad_lf_idxs(z)}

        sample = sess.run(self.samples, feed)
        try:
            stop_idx = sample.index(self.env.word_eos_id)
            sample = sample[:stop_idx]
        except ValueError:
            # No stop token. No trimming necessary. Pass.
            pass

        return " ".join(self.env.vocab[idx] for idx in sample)

    def score(self, z, u_bag, u):
        sess = tf.get_default_session()

        z = self._pad_lf_idxs(z)
        words = [self.env.word2idx[word] for word in u]

        feed = {self.lf_toks: z}
        feed.update({self.samples[t]: word for t, word in enumerate(words)})

        probs = sess.run(self.probs[:len(words)], feed)
        probs = [probs_t[word_t] for probs_t, word_t in zip(probs, words)]
        return np.log(np.prod(probs))

    def observe(self, obs, gold_lf):
        if gold_lf is None:
            return

        z = self._pad_lf_idxs(gold_lf)

        words = [self.env.word2idx[word] for word in obs[2]]
        real_length = min(len(words) + 1, self.max_timesteps) # train to output a single EOS token
        # Add a EOS token to words
        if len(words) < self.max_timesteps:
            words.append(self.env.word_eos_id)

        sess = tf.get_default_session()
        feed = {self.lf_toks: z, self.xent_gold_length: real_length}
        feed.update({self.xent_gold_words[t]: word_t
                     for t, word_t in enumerate(words)})
        sess.run(self.train_op, feed)

