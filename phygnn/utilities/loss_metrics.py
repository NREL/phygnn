# -*- coding: utf-8 -*-
"""
Loss metrics for PhyGNN
"""
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy as tf_bx


def binary_crossentropy(*args):
    """Binary crossentropy loss from keras"""
    loss = tf_bx(*args)
    return tf.reduce_mean(loss)


def mae(y_true, y_predicted):
    """Calculate the Mean Absolute Error (MAE)"""
    err = y_predicted - y_true
    err = tf.boolean_mask(err, ~tf.math.is_nan(err))
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.reduce_mean(tf.abs(err))
    return err


def mse(y_true, y_predicted):
    """Calculate the Mean Square Error (MSE)"""
    err = y_predicted - y_true
    err = tf.boolean_mask(err, ~tf.math.is_nan(err))
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.reduce_mean(tf.square(err))
    return err


def mbe(y_true, y_predicted):
    """Calculate the (absolute) Mean Bias Error (MBE).

    Note that this is actually abs(mbe) so that the NN doesnt predict a
    large negative mbe.
    """
    err = y_predicted - y_true
    err = tf.boolean_mask(err, ~tf.math.is_nan(err))
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.abs(tf.reduce_mean(err))
    return err


def relative_mae(y_true, y_predicted):
    """Calculate the relative Mean Absolute Error (MAE)"""
    err = mae(y_predicted, y_true)
    err /= tf.abs(tf.reduce_mean(y_true))
    return err


def relative_mse(y_true, y_predicted):
    """Calculate the relative Mean Squared Error (MSE)"""
    err = mse(y_predicted, y_true)
    err /= tf.abs(tf.reduce_mean(y_true))
    return err


def relative_mbe(y_true, y_predicted):
    """Calculate the relative Mean Bias Error (MBE)

    Note that this is actually abs(relative_mbe) so that the NN doesnt predict
    a large negative mbe.
    """
    err = mbe(y_predicted, y_true)
    err /= tf.abs(tf.reduce_mean(y_true))
    return err


METRICS = {'mae': mae,
           'mbe': mbe,
           'mse': mse,
           'relative_mae': relative_mae,
           'relative_mbe': relative_mbe,
           'relative_mse': relative_mse,
           'binary_crossentropy': binary_crossentropy,
           }
