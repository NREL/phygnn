# -*- coding: utf-8 -*-
"""
Loss metrics for PhyGNN
"""
import tensorflow as tf


def mae(y_predicted, y_true):
    """Calculate the Mean Absolute Error (MAE)"""
    err = y_predicted - y_true
    err = tf.boolean_mask(err, ~tf.math.is_nan(err))
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.reduce_mean(tf.abs(err))
    return err


def mse(y_predicted, y_true):
    """Calculate the Mean Square Error (MSE)"""
    err = y_predicted - y_true
    err = tf.boolean_mask(err, ~tf.math.is_nan(err))
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.reduce_mean(tf.square(err))
    return err


def mbe(y_predicted, y_true):
    """Calculate the (absolute) Mean Bias Error (MBE)"""
    err = y_predicted - y_true
    err = tf.boolean_mask(err, ~tf.math.is_nan(err))
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.abs(tf.reduce_mean(err))
    return err


def relative_mae(y_predicted, y_true):
    """Calculate the relative Mean Absolute Error (MAE)"""
    err = mae(y_predicted, y_true)
    err /= tf.reduce_mean(y_true)
    return err


def relative_mse(y_predicted, y_true):
    """Calculate the relative Mean Squared Error (MSE)"""
    err = mse(y_predicted, y_true)
    err /= tf.reduce_mean(y_true)
    return err


def relative_mbe(y_predicted, y_true):
    """Calculate the relative Mean Bias Error (MBE)"""
    err = mbe(y_predicted, y_true)
    err /= tf.reduce_mean(y_true)
    return err


METRICS = {'mae': mae,
           'mbe': mbe,
           'mse': mse,
           'relative_mae': relative_mae,
           'relative_mbe': relative_mbe,
           'relative_mse': relative_mse,
           }
