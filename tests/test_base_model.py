"""
Tests for basic tensorflow model functionality and execution.
"""
# pylint: disable=W0613
import numpy as np
import pandas as pd
import pytest

from phygnn.model_interfaces.base_model import ModelBase
from phygnn.utilities.pre_processing import PreProcess

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


def test_norm_df():
    """Test ModelBase Normalization on a dataframe"""
    model = ModelBase(None, feature_names=FEATURES.columns,
                      label_names=LABELS.columns, normalize=True)

    baseline, means, stdevs = PreProcess.normalize(FEATURES.copy())
    test = model.parse_features(FEATURES.copy())
    assert np.allclose(baseline, test)
    assert np.allclose(means, model.feature_means)
    assert np.allclose(stdevs, model.feature_stdevs)

    baseline, means, stdevs = PreProcess.normalize(LABELS.copy())
    test = model.parse_labels(LABELS.copy())
    np.allclose(baseline, test)
    assert np.allclose(means, model.label_means)
    assert np.allclose(stdevs, model.label_stdevs)


def test_norm_arr():
    """Test ModelBase Normalization on a dataframe"""
    features = FEATURES.values
    feature_names = FEATURES.columns.tolist()
    labels = LABELS.values
    label_names = LABELS.columns.tolist()
    model = ModelBase(None, feature_names=feature_names,
                      label_names=label_names, normalize=True)

    baseline, means, stdevs = PreProcess.normalize(features.copy())
    test = model.parse_features(features.copy(), names=feature_names)
    assert np.allclose(baseline, test)
    assert np.allclose(means, model.feature_means)
    assert np.allclose(stdevs, model.feature_stdevs)

    baseline, means, stdevs = PreProcess.normalize(labels.copy())
    test = model.parse_labels(labels.copy(), names=label_names)
    assert np.allclose(baseline, test)
    assert np.allclose(means, model.label_means)
    assert np.allclose(stdevs, model.label_stdevs)


def test_OHE():
    """
    Test one-hot encoding
    """
    ohe_features = FEATURES.copy()
    categories = list('def')
    ohe_features['categorical'] = np.random.choice(categories, len(FEATURES))
    one_hot_categories = {'categorical': categories}

    model = ModelBase(None, feature_names=ohe_features.columns,
                      label_names=LABELS.columns, normalize=True,
                      one_hot_categories=one_hot_categories)

    baseline, means, stdevs = \
        PreProcess.normalize(FEATURES.values.astype('float32'))
    test = model.parse_features(ohe_features)

    assert np.allclose(baseline, test[:, :2])
    assert np.allclose(means,
                       np.array(model.feature_means, dtype='float32')[:2])
    assert np.allclose(stdevs,
                       np.array(model.feature_stdevs, dtype='float32')[:2])
    for c in categories:
        assert model.get_mean(c) is None
        assert model.get_stdev(c) is None

    assert all(np.isin(categories, model.feature_names))
    assert not any(np.isin(categories, model.input_feature_names))
    assert 'categorical' not in model.feature_names
    assert 'categorical' in model.input_feature_names


def test_bad_categories():
    """
    Test OHE checks
    """
    one_hot_categories = {'cat1': list('def'), 'cat2': list('fgh')}
    feature_names = [*FEATURES.columns.tolist(), 'cat1', 'cat2']
    label_names = 'c'
    with pytest.raises(RuntimeError):
        ModelBase(None, feature_names=feature_names,
                  label_names=label_names, normalize=True,
                  one_hot_categories=one_hot_categories)
