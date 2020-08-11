"""
Test the custom tensorflow utilities
"""
import numpy as np
import tensorflow as tf
from phygnn.tf_utilities import tf_isin, tf_log10


def test_tf_isin():
    """Test the tensorflow isin utility method"""
    a = np.arange(10)
    b = [4, 7, 8]
    tfa = tf.convert_to_tensor(a)
    c = np.isin(a, b)
    tfc = tf_isin(tfa, b)
    assert all(c == tfc.numpy())


def test_tf_log10():
    """Test the tensorflow log base 10 utility method"""
    a = np.arange(1, 10)
    tfa = tf.convert_to_tensor(a, dtype=np.float32)
    b = np.log10(a)
    tfb = tf_log10(tfa)
    assert np.allclose(b, tfb.numpy())
