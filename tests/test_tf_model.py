"""
Tests for basic phygnn functionality and execution.
"""
# pylint: disable=W0613
import numpy as np
import pandas as pd
import pytest

from phygnn.model_interfaces.tf_model import TfModel


N = 100
A = np.linspace(-1, 1, N)
B = np.linspace(-1, 1, N)
A, B = np.meshgrid(A, B)
A = np.expand_dims(A.flatten(), axis=1)
B = np.expand_dims(B.flatten(), axis=1)

Y = np.sqrt(A ** 2 + B ** 2)
X = np.hstack((A, B))
features = pd.DataFrame(X, columns=['a', 'b'])

Y_NOISE = Y * (1 + (np.random.random(Y.shape) - 0.5) * 0.5) + 0.1
labels = pd.DataFrame(Y_NOISE, columns=['c'])


@pytest.mark.parametrize(
    'hidden_layers',
    [None,
     [{'units': 64, 'name': 'relu1'},
      {'units': 64, 'name': 'relu2'}]])
def test_nn(hidden_layers):
    """Test TfModel """
    model = TfModel.train(features, labels, hidden_layers=hidden_layers,
                          epochs=10, fit_kwargs={"batch_size": 4},
                          early_stop=False)

    test_mae = np.mean(np.abs(model[X].values.ravel() - Y))

    n_layers = len(hidden_layers) + 1 if hidden_layers is not None else 1
    loss = 4
    assert len(model.layers) == n_layers
    assert len(model.weights) == n_layers * 2
    assert len(model.history) == 10
    assert model.history['val_loss'].values[-1] < loss
    assert test_mae < loss


@pytest.mark.parametrize(
    ('normalize', 'loss'),
    [(True, 0.35),
     (False, 0.035),
     ((True, False), 0.035),
     ((False, True), 0.35)])
def test_normalize(normalize, loss):
    """Test TfModel """
    hidden_layers = [{'units': 64, 'activation': 'relu', 'name': 'relu1'},
                     {'units': 64, 'activation': 'relu', 'name': 'relu2'}]
    model = TfModel.train(features, labels,
                          normalize=normalize,
                          hidden_layers=hidden_layers,
                          epochs=10, fit_kwargs={"batch_size": 4},
                          early_stop=False)

    test_mae = np.mean(np.abs(model[X].values.ravel() - Y))
    assert model.history['val_loss'].values[-1] < loss
    assert test_mae < loss


def test_complex_nn():
    """Test complex TfModel """
    hidden_layers = [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                     {'units': 64},
                     {'batch_normalization': {'axis': -1}},
                     {'activation': 'relu'},
                     {'dropout': 0.01}]
    model = TfModel.train(features, labels, hidden_layers=hidden_layers,
                          epochs=10, fit_kwargs={"batch_size": 4},
                          early_stop=False)

    test_mae = np.mean(np.abs(model[X].values.ravel() - Y))

    loss = 0.55
    assert len(model.layers) == 8
    assert len(model.weights) == 10
    assert model.history['val_loss'].values[-1] < loss
    assert test_mae < loss
