"""
Tests for basic phygnn functionality and execution.
"""
# pylint: disable=W0613
import os
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (InputLayer, Dense, Dropout, Activation,
                                     BatchNormalization)
from phygnn import PhysicsGuidedNeuralNetwork, TESTDATADIR


FPATH = os.path.join(TESTDATADIR, '_temp_model.pkl')

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


def p_fun_pythag(y_predicted, y_true, p):
    """Example function for loss calculation using physical relationships.

    Parameters
    ----------
    y_predicted : tf.Tensor
        Predicted y values in a 2D tensor based on x values in this batch.
    y_true : np.ndarray
        Known y values that were given to the PhyGNN fit method.
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


def p_fun_bad(y_predicted, y_true, p):
    """This is an example of a poorly formulated p_fun() that uses
    numpy operations."""

    y_physical = p[:, 0]**2 + p[:, 1]**2
    p_loss = np.mean(np.abs(y_predicted.numpy() - y_physical))
    p_loss = tf.convert_to_tensor(p_loss, dtype=tf.float32)

    return p_loss


def test_nn():
    """Test the basic NN operation of the PGNN without weighting pfun."""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(1.0, 0.0),
                                       input_dims=2, output_dims=1,
                                       feature_names=['a', 'b'],
                                       output_names=['c'])
    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    test_mae = np.mean(np.abs(model.predict(X) - Y))

    assert len(model.layers) == 6
    assert len(model.weights) == 6
    assert len(model.history) == 20
    assert model.history.validation_loss.values[-1] < 0.15
    assert test_mae < 0.15


def test_phygnn():
    """Test the operation of the PGNN with weighting pfun."""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(0.0, 1.0),
                                       input_dims=2, output_dims=1)
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
    assert model.history.validation_loss.values[-1] < 0.015
    assert test_mae < 0.015


def test_df_input():
    """Test the operation of the PGNN with labeled input dataframes."""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(0.0, 1.0),
                                       input_dims=2, output_dims=1)
    x_df = pd.DataFrame(X, columns=('a', 'b'))
    y_df = pd.DataFrame(Y_NOISE, columns=('c',))
    p_df = pd.DataFrame(P, columns=('a', 'b'))
    model.fit(x_df, y_df, p_df, n_batch=1, n_epoch=2)

    assert model.feature_names == ['a', 'b']
    assert model.output_names == ['c']

    x_df_bad = pd.DataFrame(X, columns=('x1', 'x2'))
    y_df_bad = pd.DataFrame(Y_NOISE, columns=('y',))

    try:
        model.fit(x_df_bad, y_df_bad, p_df, n_batch=1, n_epoch=2)
    except AssertionError as e:
        assert "Cannot work with input x columns: ['x1', 'x2']" in str(e)

    try:
        model.fit(x_df, y_df_bad, p_df, n_batch=1, n_epoch=2)
    except AssertionError as e:
        assert "Cannot work with input y columns: ['y']" in str(e)


def test_kernel_regularization():
    """Test the kernel regularization of phygnn."""
    base = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                      hidden_layers=HIDDEN_LAYERS,
                                      loss_weights=(1.0, 0.0),
                                      input_dims=2, output_dims=1)

    model_l1 = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                          hidden_layers=HIDDEN_LAYERS,
                                          loss_weights=(1.0, 0.0),
                                          input_dims=2, output_dims=1,
                                          kernel_reg_rate=0.01,
                                          kernel_reg_power=1)

    model_l2 = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                          hidden_layers=HIDDEN_LAYERS,
                                          loss_weights=(1.0, 0.0),
                                          input_dims=2, output_dims=1,
                                          kernel_reg_rate=0.01,
                                          kernel_reg_power=2)

    base.seed(0)
    base.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l1.seed(0)
    model_l1.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l2.seed(0)
    model_l2.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)

    assert base.kernel_reg_term > model_l1.kernel_reg_term
    assert model_l1.kernel_reg_term > model_l2.kernel_reg_term

    assert np.abs(base.kernel_reg_term - 498) < 5
    assert np.abs(model_l1.kernel_reg_term - 84) < 5
    assert np.abs(model_l2.kernel_reg_term - 17) < 5


def test_bias_regularization():
    """Test the bias regularization of phygnn."""
    base = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                      hidden_layers=HIDDEN_LAYERS,
                                      loss_weights=(1.0, 0.0),
                                      input_dims=2, output_dims=1)

    model_l1 = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                          hidden_layers=HIDDEN_LAYERS,
                                          loss_weights=(1.0, 0.0),
                                          input_dims=2, output_dims=1,
                                          bias_reg_rate=0.01,
                                          bias_reg_power=1)

    model_l2 = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                          hidden_layers=HIDDEN_LAYERS,
                                          loss_weights=(1.0, 0.0),
                                          input_dims=2, output_dims=1,
                                          bias_reg_rate=0.01,
                                          bias_reg_power=2)

    base.seed(0)
    base.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l1.seed(0)
    model_l1.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)
    model_l2.seed(0)
    model_l2.fit(X, Y_NOISE, P, n_batch=1, n_epoch=20)

    assert base.bias_reg_term > model_l1.bias_reg_term
    assert model_l1.bias_reg_term > model_l2.bias_reg_term
    assert np.abs(base.bias_reg_term - 5) < 5
    assert np.abs(model_l1.bias_reg_term - 4) < 5
    assert np.abs(model_l2.bias_reg_term - 1) < 5


def test_save_load():
    """Test the save/load operations of PGNN"""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(0.0, 1.0),
                                       input_dims=2, output_dims=1,
                                       feature_names=['a', 'b'],
                                       output_names=['c'])

    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)
    y_pred = model.predict(X)

    model.save(FPATH)
    loaded = PhysicsGuidedNeuralNetwork.load(FPATH)
    y_pred_loaded = loaded.predict(X)
    assert np.allclose(y_pred, y_pred_loaded)
    assert loaded.feature_names == ['a', 'b']
    assert loaded.output_names == ['c']
    os.remove(FPATH)


def test_bad_pfun():
    """Test the preflight check with a non-differentiable p_fun"""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_bad,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(0.0, 1.0),
                                       input_dims=2, output_dims=1)
    with pytest.raises(RuntimeError) as e:
        model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)

    assert 'not differentiable' in str(e.value)


def test_dropouts():
    """Test the dropout rate kwargs for adding dropout layers."""
    HIDDEN_LAYERS = [
        {'units': 64}, {'activation': 'relu'}, {'dropout': 0.1},
        {'units': 64, 'activation': 'relu', 'name': 'relu2', 'dropout': 0.1}]
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(0.0, 1.0),
                                       input_dims=2, output_dims=1)

    assert len(model.layers) == 8, "dropout layers did not get added!"
    assert isinstance(model.layers[0], InputLayer)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Activation)
    assert isinstance(model.layers[3], Dropout)
    assert isinstance(model.layers[4], Dense)
    assert isinstance(model.layers[5], Activation)
    assert isinstance(model.layers[6], Dropout)

    model.fit(X, Y_NOISE, P, n_batch=4, n_epoch=20)
    y_pred = model.predict(X)

    model.save(FPATH)
    loaded = PhysicsGuidedNeuralNetwork.load(FPATH)
    y_pred_loaded = loaded.predict(X)
    assert np.allclose(y_pred, y_pred_loaded)
    assert len(model.layers) == len(loaded.layers)
    os.remove(FPATH)


def test_batch_norm():
    """Test the addition of BatchNormalization layers"""
    HIDDEN_LAYERS = [{'units': 64},
                     {'batch_normalization': {'axis': 1}},
                     {'activation': 'relu'},
                     {'units': 64, 'activation': 'relu',
                      'batch_normalization': {'axis': 1}},
                     ]
    model = PhysicsGuidedNeuralNetwork(p_fun=p_fun_pythag,
                                       hidden_layers=HIDDEN_LAYERS,
                                       loss_weights=(0.0, 1.0),
                                       input_dims=2, output_dims=1)

    assert len(model.layers) == 8, "Batch norm layers did not get added!"
    assert isinstance(model.layers[0], InputLayer)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], BatchNormalization)
    assert isinstance(model.layers[3], Activation)
    assert isinstance(model.layers[4], Dense)
    assert isinstance(model.layers[5], BatchNormalization)
    assert isinstance(model.layers[6], Activation)

    model.fit(X, Y_NOISE, P, n_batch=1, n_epoch=10)
    y_pred = model.predict(X)

    model.save(FPATH)
    loaded = PhysicsGuidedNeuralNetwork.load(FPATH)
    y_pred_loaded = loaded.predict(X)

    assert np.allclose(y_pred, y_pred_loaded)
    assert len(model.layers) == len(loaded.layers)

    os.remove(FPATH)


def test_validation_split_shuffle():
    """Test the validation split operation with shuffling"""
    out = PhysicsGuidedNeuralNetwork._get_val_split(X, Y, P, shuffle=True,
                                                    validation_split=0.3)
    x, y, p, x_val, y_val, p_val = out

    assert (x_val == p_val).all()
    assert (x == p).all()

    assert id(x) != id(X)
    assert x.shape[1] == x.shape[1]
    assert len(x) == int(0.7 * len(X))
    assert len(x_val) == int(0.3 * len(X))

    assert id(y) != id(Y)
    assert y.shape[1] == y.shape[1]
    assert len(y) == int(0.7 * len(Y))
    assert len(y_val) == int(0.3 * len(Y))

    assert id(p) != id(P)
    assert p.shape[1] == p.shape[1]
    assert len(p) == int(0.7 * len(P))
    assert len(p_val) == int(0.3 * len(P))

    for i in range(len(x_val)):
        row = x_val[i, :]
        assert ~np.any(np.all((row == x), axis=1))

    for i in range(len(p_val)):
        row = p_val[i, :]
        assert ~np.any(np.all((row == p), axis=1))


def test_validation_split_no_shuffle():
    """Test the validation split operation without shuffling"""
    out = PhysicsGuidedNeuralNetwork._get_val_split(X, Y, P, shuffle=False,
                                                    validation_split=0.3)
    x, y, p, x_val, y_val, p_val = out
    assert (x_val == p_val).all()
    assert (x == p).all()
    assert all(np.sqrt(x[:, 0]**2 + x[:, 1]**2).reshape((len(x), 1)) == y)
    assert (x_val == X[0:len(x_val)]).all()
    assert (y_val == Y[0:len(y_val)]).all()
    assert (p_val == P[0:len(p_val)]).all()


def test_batching_shuffle():
    """Test the batching operation with shuffling"""
    x_batches, y_batches, p_batches = PhysicsGuidedNeuralNetwork._make_batches(
        X, Y, P, n_batch=4, shuffle=True)

    assert len(x_batches) == 4
    assert len(y_batches) == 4
    assert len(p_batches) == 4

    assert ~(x_batches[0] == X[0:len(x_batches[0]), :]).all()
    assert ~(y_batches[0] == Y[0:len(y_batches[0]), :]).all()
    assert ~(p_batches[0] == P[0:len(p_batches[0]), :]).all()

    for i, x_b in enumerate(x_batches):
        assert (x_b == p_batches[i]).all()
        truth = np.sqrt(x_b[:, 0]**2 + x_b[:, 1]**2).reshape((len(x_b), 1))
        y_check = y_batches[i]
        assert np.allclose(truth, y_check)


def test_batching_no_shuffle():
    """Test the batching operation without shuffling"""
    x_batches, y_batches, p_batches = PhysicsGuidedNeuralNetwork._make_batches(
        X, Y, P, n_batch=6, shuffle=False)

    assert len(x_batches) == 6
    assert len(y_batches) == 6
    assert len(p_batches) == 6

    assert (x_batches[0] == X[0:len(x_batches[0]), :]).all()
    assert (y_batches[0] == Y[0:len(y_batches[0]), :]).all()
    assert (p_batches[0] == P[0:len(p_batches[0]), :]).all()

    assert (x_batches[-1] == X[-(len(x_batches[0]) - 1):, :]).all()
    assert (y_batches[-1] == Y[-(len(y_batches[0]) - 1):, :]).all()
    assert (p_batches[-1] == P[-(len(p_batches[0]) - 1):, :]).all()

    for i, x_b in enumerate(x_batches):
        assert (x_b == p_batches[i]).all()
        truth = np.sqrt(x_b[:, 0]**2 + x_b[:, 1]**2).reshape((len(x_b), 1))
        y_check = y_batches[i]
        assert np.allclose(truth, y_check)
