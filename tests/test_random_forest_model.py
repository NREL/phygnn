"""
Tests for basic phygnn functionality and execution.
"""
# pylint: disable=W0613
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from phygnn import TESTDATADIR
from phygnn.model_interfaces.random_forest_model import RandomForestModel

FPATH = os.path.join(TESTDATADIR, '_temp_model', '_temp_model.json')

N = 100
A = np.linspace(-1, 1, N)
B = np.linspace(-1, 1, N)
A, B = np.meshgrid(A, B)
A = np.expand_dims(A.flatten(), axis=1)
B = np.expand_dims(B.flatten(), axis=1)

Y = np.sqrt(A ** 2 + B ** 2)
X = np.hstack((A, B))
FEATURES = pd.DataFrame(X, columns=['a', 'b'])

Y_NOISE = Y * (1 + (np.random.random(Y.shape) - 0.5) * 0.5) + 0.1
LABELS = pd.DataFrame(Y_NOISE, columns=['c'])


def test_random_forest():
    """Test the RandomForestModel"""
    model = RandomForestModel.build_trained(FEATURES.copy(), LABELS.copy())

    test_mae = np.mean(np.abs(model[X].values.ravel() - Y))
    assert test_mae < 0.4


def test_save_load():
    """Test the save/load operations of RandomForestModel"""
    model = RandomForestModel.build_trained(FEATURES.copy(), LABELS.copy(),
                                            save_path=FPATH)
    y_pred = model[X]

    loaded = RandomForestModel.load(FPATH)
    loaded.train_model(FEATURES.copy(), LABELS.copy())
    y_pred_loaded = loaded[X]
    np.allclose(y_pred.values, y_pred_loaded.values)
    assert loaded.feature_names == ['a', 'b']
    assert loaded.label_names == ['c']
    shutil.rmtree(os.path.dirname(FPATH))


def test_OHE():
    """
    Test one-hot encoding
    """
    ohe_features = FEATURES.copy()
    categories = list('def')
    ohe_features['categorical'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}

    model = RandomForestModel.build_trained(
        ohe_features, LABELS,
        one_hot_categories=one_hot_categories)

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
    ohe_features = FEATURES.copy()
    categories = list('abc')
    ohe_features['categorical'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}
    with pytest.raises(RuntimeError):
        RandomForestModel.build_trained(
            ohe_features, LABELS,
            one_hot_categories=one_hot_categories)

    ohe_features = FEATURES.copy()
    categories = list('cdf')
    ohe_features['categorical'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}
    with pytest.raises(RuntimeError):
        RandomForestModel.build_trained(
            ohe_features, LABELS,
            one_hot_categories=one_hot_categories)

    ohe_features = FEATURES.copy()
    categories = list('def')
    ohe_features['categories'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}
    with pytest.raises(RuntimeError):
        RandomForestModel.build_trained(
            ohe_features, LABELS,
            one_hot_categories=one_hot_categories)

    ohe_features = FEATURES.copy()
    categories = list('def')
    ohe_features['categorical'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}
    model = RandomForestModel.build_trained(
        ohe_features, LABELS,
        one_hot_categories=one_hot_categories)

    with pytest.raises(RuntimeError):
        x = ohe_features.values[:, 1:]
        model.predict(x)
