"""
Tests for basic tensorflow model functionality and execution.
"""
# pylint: disable=W0613
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from phygnn.model_interfaces.tf_model import TfModel
from phygnn.utilities import TF2

TfModel.seed(0)

mae_key = 'val_mae' if TF2 else 'val_mean_absolute_error'

N = 100
A = np.linspace(-1, 1, N)
B = np.linspace(-1, 1, N)
A, B = np.meshgrid(A, B)
A = np.expand_dims(A.flatten(), axis=1)
B = np.expand_dims(B.flatten(), axis=1)

Y = np.sqrt(A ** 2 + B ** 2)
X = np.hstack((A, B))
FEATURES = pd.DataFrame(X, columns=['a', 'b'])
LABELS = pd.DataFrame(Y, columns=['c'])


@pytest.mark.parametrize(
    ('hidden_layers', 'loss'),
    [(None, 0.6),
     ([{'units': 64, 'activation': 'relu', 'name': 'relu1'},
       {'units': 64, 'activation': 'relu', 'name': 'relu2'}], 0.03)])
def test_nn(hidden_layers, loss):
    """Test TfModel"""
    model = TfModel.build_trained(FEATURES.copy(), LABELS.copy(),
                                  hidden_layers=hidden_layers,
                                  epochs=10,
                                  fit_kwargs={"batch_size": 16},
                                  early_stop=False)

    n_l = len(hidden_layers) * 2 + 1 if hidden_layers is not None else 1
    n_w = (len(hidden_layers) + 1) * 2 if hidden_layers is not None else 2
    assert len(model.layers) == n_l
    assert len(model.weights) == n_w
    assert len(model.history) == 10

    test_mae = np.mean(np.abs(model[X].values - Y))
    assert model.history[mae_key].values[-1] < loss
    assert test_mae < loss


@pytest.mark.parametrize(
    ('normalize', 'loss'),
    [(True, 0.09),
     (False, 0.015),
     ((True, False), 0.01),
     ((False, True), 0.09)])
def test_normalize(normalize, loss):
    """Test TfModel"""
    hidden_layers = [{'units': 64, 'activation': 'relu', 'name': 'relu1'},
                     {'units': 64, 'activation': 'relu', 'name': 'relu2'}]
    model = TfModel.build_trained(FEATURES.copy(), LABELS.copy(),
                                  normalize=normalize,
                                  hidden_layers=hidden_layers,
                                  epochs=10, fit_kwargs={"batch_size": 16},
                                  early_stop=False)

    test_mae = np.mean(np.abs(model[X].values - Y))
    assert model.history[mae_key].values[-1] < loss
    assert test_mae < loss


def test_normalize_build_separate():
    """Annoying case of building and training separately with numpy array
    input."""
    hidden_layers = [{'units': 64, 'activation': 'relu', 'name': 'relu1'},
                     {'units': 64, 'activation': 'relu', 'name': 'relu2'}]
    model = TfModel.build(list(FEATURES.columns.values),
                          list(LABELS.columns.values),
                          normalize=(True, True),
                          hidden_layers=hidden_layers)
    model.train_model(FEATURES.values.copy(), LABELS.values.copy(),
                      epochs=10, fit_kwargs={"batch_size": 16},
                      early_stop=False)
    y = model.predict(FEATURES.values.copy())
    mse = np.mean((y.values - LABELS.values)**2)
    mbe = np.abs(np.mean(y.values - LABELS.values))
    assert mse < 3e-5
    assert mbe < 3e-3
    assert 'c' in model._norm_params


def test_complex_nn():
    """Test complex TfModel"""
    hidden_layers = [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                     {'units': 64},
                     {'batch_normalization': {'axis': -1}},
                     {'activation': 'relu'},
                     {'dropout': 0.01}]
    model = TfModel.build_trained(FEATURES.copy(), LABELS.copy(),
                                  hidden_layers=hidden_layers,
                                  epochs=10, fit_kwargs={"batch_size": 16},
                                  early_stop=False)

    assert len(model.layers) == 8
    assert len(model.weights) == 10

    test_mae = np.mean(np.abs(model[X].values - Y))
    loss = 0.15
    assert model.history[mae_key].values[-1] < loss
    assert test_mae < loss


def test_dropout():
    """Test a model trained with dropout vs. no dropout and make sure the
    predictions are different."""
    hidden_layers_1 = [{'units': 64, 'activation': 'relu'},
                       {'units': 64}, {'activation': 'relu'}]
    hidden_layers_2 = [{'units': 64, 'activation': 'relu', 'dropout': 0.05},
                       {'units': 64}, {'activation': 'relu'},
                       {'dropout': 0.05}]
    TfModel.seed()
    model_1 = TfModel.build_trained(FEATURES.copy(), LABELS.copy(),
                                    hidden_layers=hidden_layers_1,
                                    epochs=10, fit_kwargs={"batch_size": 16},
                                    early_stop=False)
    TfModel.seed()
    model_2 = TfModel.build_trained(FEATURES.copy(), LABELS.copy(),
                                    hidden_layers=hidden_layers_2,
                                    epochs=10, fit_kwargs={"batch_size": 16},
                                    early_stop=False)

    out1 = model_1.history[mae_key].values[-5:]
    out2 = model_2.history[mae_key].values[-5:]
    assert (out2 > out1).all()


def test_save_load():
    """Test the save/load operations of TfModel"""
    with tempfile.TemporaryDirectory() as td:
        model_fpath = os.path.join(td, 'test_model/')
        hidden_layers = [
            {'units': 64, 'activation': 'relu', 'name': 'relu1'},
            {'units': 64, 'activation': 'relu', 'name': 'relu2'},
        ]
        model = TfModel.build_trained(FEATURES.copy(), LABELS.copy(),
                                      hidden_layers=hidden_layers,
                                      epochs=10, fit_kwargs={"batch_size": 16},
                                      early_stop=False,
                                      save_path=model_fpath)
        y_pred = model[X]

        loaded = TfModel.load(model_fpath)
        y_pred_loaded = loaded[X]
        np.allclose(y_pred.values, y_pred_loaded.values)
        assert loaded.feature_names == ['a', 'b']
        assert loaded.label_names == ['c']

        with open(os.path.join(model_fpath, 'model.json')) as f:
            params = json.load(f)

        assert 'version_record' in params


def test_OHE():
    """
    Test one-hot encoding
    """
    ohe_features = FEATURES.copy()
    categories = list('def')
    ohe_features['categorical'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}

    hidden_layers = [{'units': 64, 'activation': 'relu', 'name': 'relu1'},
                     {'units': 64, 'activation': 'relu', 'name': 'relu2'}]
    model = TfModel.build_trained(ohe_features, LABELS,
                                  one_hot_categories=one_hot_categories,
                                  hidden_layers=hidden_layers,
                                  epochs=10, fit_kwargs={"batch_size": 16},
                                  early_stop=False)

    assert all(np.isin(categories, model.feature_names))
    assert not any(np.isin(categories, model.input_feature_names))
    assert 'categorical' not in model.feature_names
    assert 'categorical' in model.input_feature_names

    x = ohe_features.values
    out = model.predict(x)
    assert 'c' in out


def test_bad_categories():
    """
    Test OHE checks
    """
    hidden_layers = [{'units': 64, 'activation': 'relu', 'name': 'relu1'},
                     {'units': 64, 'activation': 'relu', 'name': 'relu2'}]

    one_hot_categories = {'categorical': list('abc')}
    feature_names = [*FEATURES.columns.tolist(), 'categorical']
    label_names = 'c'
    with pytest.raises(RuntimeError):
        TfModel.build(feature_names, label_names,
                      one_hot_categories=one_hot_categories,
                      hidden_layers=hidden_layers)

    one_hot_categories = {'categorical': list('cdf')}
    feature_names = [*FEATURES.columns.tolist(), 'categorical']
    label_names = 'c'
    with pytest.raises(RuntimeError):
        TfModel.build(feature_names, label_names,
                      one_hot_categories=one_hot_categories,
                      hidden_layers=hidden_layers)

    one_hot_categories = {'categorical': list('def')}
    feature_names = [*FEATURES.columns.tolist(), 'categories']
    label_names = 'c'
    with pytest.raises(RuntimeError):
        TfModel.build(feature_names, label_names,
                      one_hot_categories=one_hot_categories,
                      hidden_layers=hidden_layers)

    one_hot_categories = {'cat1': list('def'), 'cat2': list('fgh')}
    feature_names = [*FEATURES.columns.tolist(), 'cat1', 'cat2']
    label_names = 'c'
    with pytest.raises(RuntimeError):
        TfModel.build(feature_names, label_names,
                      one_hot_categories=one_hot_categories,
                      hidden_layers=hidden_layers)

    ohe_features = FEATURES.copy()
    categories = list('def')
    ohe_features['categorical'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}

    model = TfModel.build_trained(ohe_features, LABELS,
                                  one_hot_categories=one_hot_categories,
                                  hidden_layers=hidden_layers,
                                  epochs=10, fit_kwargs={"batch_size": 16},
                                  early_stop=False)

    with pytest.raises(RuntimeError):
        x = ohe_features.values[:, 1:]
        model.predict(x)


def _test_conv_models(hidden_layers, features, labels):
    """Basic evaluation of 1/2/3D convolutional model."""

    feature_names = [f'F{i}' for i in range(features.shape[-1])]
    label_names = [f'L{i}' for i in range(labels.shape[-1])]

    dummy = TfModel.build(feature_names, label_names,
                          hidden_layers=hidden_layers,
                          input_layer=False, output_layer=False)

    model = TfModel.build_trained(features, labels,
                                  hidden_layers=hidden_layers,
                                  normalize=(True, True),
                                  input_layer=False, output_layer=False,
                                  epochs=50, fit_kwargs={"batch_size": 16},
                                  early_stop=False)

    # make sure raw data was not normalized
    assert np.allclose(features.mean(), 100, atol=1)
    assert np.allclose(labels.mean(), 50, atol=1)

    # check that numpy array channels are given basic names
    assert all(name in model.feature_names for name in feature_names)
    assert all(name in model.label_names for name in label_names)
    assert all(name in model.means for name in feature_names)
    assert all(name in model.means for name in label_names)

    with tempfile.TemporaryDirectory() as td:
        model.save_model(os.path.join(td, 'model'))

        loaded = TfModel.load(os.path.join(td, 'model'))

    dummy_out = dummy.predict(features)
    model_out = model.predict(features)
    loaded_out = loaded.predict(features)

    assert not np.allclose(dummy_out, model_out, atol=1)
    assert np.allclose(model_out, loaded_out)


def test_1D_conv():
    """Test a 1D convolutional model with data shape
    (n_obs, n_time, n_features)"""

    hidden_layers = [
        {"class": "FlexiblePadding", "paddings": [[0, 0], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {'class': 'Conv1D', 'filters': 16, 'kernel_size': 3,
         'padding': 'valid', 'activation': None},
        {"class": "LeakyReLU", "alpha": 0.1},
        {"class": "Cropping1D", "cropping": 2},

        {"class": "FlexiblePadding", "paddings": [[0, 0], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {'class': 'Conv1D', 'filters': 4, 'kernel_size': 3,
         'padding': 'valid', 'activation': None},
        {"class": "Cropping1D", "cropping": 2},
    ]

    features = np.random.normal(100, 10, (8, 24, 2))
    labels = np.random.normal(50, 10, (8, 24, 4))

    _test_conv_models(hidden_layers, features, labels)


def test_2D_conv():
    """Test a 2D convolutional model with data shape
    (n_obs, n_spatial_1, n_spatial_2, n_features)"""

    hidden_layers = [
        {"class": "FlexiblePadding",
         "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {'class': 'Conv2D', 'filters': 16, 'kernel_size': 3,
         'padding': 'valid', 'activation': None},
        {"class": "LeakyReLU", "alpha": 0.1},
        {"class": "Cropping2D", "cropping": 2},

        {"class": "FlexiblePadding",
         "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {'class': 'Conv2D', 'filters': 4, 'kernel_size': 3,
         'padding': 'valid', 'activation': None},
        {"class": "Cropping2D", "cropping": 2},
    ]

    features = np.random.normal(100, 10, (8, 10, 10, 2))
    labels = np.random.normal(50, 10, (8, 10, 10, 4))

    _test_conv_models(hidden_layers, features, labels)


def test_3D_conv():
    """Test a 3D convolutional model with data shape
    (n_obs, n_spatial_1, n_spatial_2, n_temporal, n_features)"""

    hidden_layers = [
        {"class": "FlexiblePadding",
         "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {'class': 'Conv3D', 'filters': 16, 'kernel_size': 3,
         'padding': 'valid', 'activation': None},
        {"class": "LeakyReLU", "alpha": 0.1},
        {"class": "Cropping3D", "cropping": 2},

        {"class": "FlexiblePadding",
         "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {'class': 'Conv3D', 'filters': 4, 'kernel_size': 3,
         'padding': 'valid', 'activation': None},
        {"class": "Cropping3D", "cropping": 2},
    ]

    features = np.random.normal(100, 10, (8, 10, 10, 6, 2))
    labels = np.random.normal(50, 10, (8, 10, 10, 6, 4))

    _test_conv_models(hidden_layers, features, labels)
