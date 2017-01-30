import numpy as np
import tensorflow as tf


class colors:
    """Defines shell escapes for colored text output."""

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def orthogonal_initializer(scale=1.1):
    """
    Orthogonal matrix initializer, stolen from Keras.

    # References
        Saxe et al., http://arxiv.org/abs/1312.6120
    """
    def _orthogonal_initializer(shape, dtype=np.float32, **kwargs):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # Pick the one with the correct shape.
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
    return _orthogonal_initializer
