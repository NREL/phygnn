# -*- coding: utf-8 -*-
"""
Tensorflow utilities
"""
import tensorflow as tf


def tf_isin(a, b):
    """Check whether a is in b"""
    assert isinstance(b, (list, tuple)), 'Second arg must be a list or tuple!'
    x = [tf.equal(a, i) for i in b]
    out = tf.reduce_any(tf.stack(x), axis=0)
    assert out.shape == a.shape
    return out


def tf_log10(x):
    """Compute log base 10 of a tensor x"""
    num = tf.math.log(x)
    den = tf.math.log(tf.constant(10, dtype=num.dtype))
    return num / den
