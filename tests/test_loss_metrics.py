"""
Tests for basic phygnn functionality and execution.
"""
# pylint: disable=W0613
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, InputLayer

from phygnn import PhysicsGuidedNeuralNetwork

N = 100
A = np.linspace(-1, 1, N)
B = np.linspace(-1, 1, N)
A, B = np.meshgrid(A, B)
A = np.expand_dims(A.flatten(), axis=1)
B = np.expand_dims(B.flatten(), axis=1)

Y = np.sqrt(A ** 2 + B ** 2)
X = np.hstack((A, B))
P = X.copy()
Y_NOISE = Y * (1 + (np.random.random(Y.shape) - 0.5) * 0.5) + 0.1


HIDDEN_LAYERS = [{'units': 64, 'activation': 'relu', 'name': 'relu1'},
                 {'units': 64, 'activation': 'relu', 'name': 'relu2'},
                 ]


def p_fun_pythag(model, y_true, y_predicted, p):  # noqa: ARG001
    """Example function for loss calculation using physical relationships.

    Parameters
    ----------
    model : PhysicsGuidedNeuralNetwork
        Instance of the phygnn model at the current point in training.
    y_true : np.ndarray
        Known y values that were given to the PhyGNN fit method.
    y_predicted : tf.Tensor
        Predicted y values in a 2D tensor based on x values in this batch.
    p : np.ndarray
        Supplemental physical feature data that can be used to calculate a
        y_physical value to compare against y_predicted. The rows in this
        array have been carried through the batching process alongside y_true
        and the features used to create y_predicted and so can be used 1-to-1
        with the rows in y_predicted and y_true.

    Returns
    -------
    p_loss : tf.Tensor
        A 0D tensor physical loss value.
    """

    p = tf.convert_to_tensor(p, dtype=tf.float32)
    y_physical = tf.sqrt(p[:, 0]**2 + p[:, 1]**2)
    y_physical = tf.expand_dims(y_physical, 1)

    p_loss = tf.math.reduce_mean(tf.math.abs(y_predicted - y_physical))

    return p_loss


@pytest.mark.parametrize('metric_name', ('mae', 'mse', 'mbe', 'relative_mae',
                                         'relative_mse', 'relative_mbe'))
def test_loss_metric(metric_name):
    """Test the operation of the PGNN with weighting pfun."""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                       metric=metric_name,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(0.0, 1.0),
                                       n_features=2, n_labels=1)
    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    test_mae = np.mean(np.abs(model.predict(X) - Y))

    assert len(model.layers) == 6
    assert len(model.weights) == 6
    assert len(model.history) == 20
    assert isinstance(model.layers[0], InputLayer)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Activation)
    assert isinstance(model.layers[3], Dense)
    assert isinstance(model.layers[4], Activation)
    assert isinstance(model.layers[5], Dense)
    assert model.history.validation_loss.values[-1] < 0.05
    assert test_mae < 0.05
