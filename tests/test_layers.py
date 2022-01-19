"""
Test the custom tensorflow utilities
"""
import copy
import numpy as np
import pytest
import tensorflow as tf

from phygnn.layers.custom_layers import SkipConnection, SpatioTemporalExpansion
from phygnn.layers.handlers import Layers, HiddenLayers


@pytest.mark.parametrize(
    'hidden_layers',
    [None,
     [{'units': 64, 'name': 'relu1'},
      {'units': 64, 'name': 'relu2'}]])
def test_layers(hidden_layers):
    """Test Layers handler"""
    n_features = 1
    n_labels = 1
    layers = Layers(n_features, n_labels=n_labels, hidden_layers=hidden_layers)
    n_layers = len(hidden_layers) + 2 if hidden_layers is not None else 2
    assert len(layers) == n_layers


def test_dropouts():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [{'units': 64, 'name': 'relu1', 'dropout': 0.1},
                     {'units': 64, 'name': 'relu2', 'dropout': 0.1}]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 4


def test_activate():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [{'units': 64, 'activation': 'relu', 'name': 'relu1'},
                     {'units': 64, 'activation': 'relu', 'name': 'relu2'}]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 4


def test_batch_norm():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [
        {'units': 64, 'name': 'relu1', 'batch_normalization': {'axis': -1}},
        {'units': 64, 'name': 'relu2', 'batch_normalization': {'axis': -1}}]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 4


def test_complex_layers():
    """Test the dropout rate kwargs for adding dropout layers."""
    hidden_layers = [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                     {'units': 64},
                     {'batch_normalization': {'axis': -1}},
                     {'activation': 'relu'},
                     {'dropout': 0.01}]
    layers = HiddenLayers(hidden_layers)

    assert len(layers) == 7


def test_repeat_layers():
    """Test repeat argument to duplicate layers"""
    hidden_layers = [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                     {'n': 3, 'repeat': [{'units': 64},
                                         {'activation': 'relu'},
                                         {'dropout': 0.01}]},
                     ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers) == 12

    hidden_layers = [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                     {'n': 3, 'repeat': {'units': 64}},
                     ]
    layers = HiddenLayers(hidden_layers)
    assert len(layers) == 6

    hidden_layers = [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                     {'repeat': {'units': 64}},
                     ]
    with pytest.raises(KeyError):
        layers = HiddenLayers(hidden_layers)


def test_skip_connection():
    """Test a functional skip connection"""
    hidden_layers = [
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'class': 'SkipConnection', 'name': 'a'},
        {'units': 64, 'activation': 'relu', 'dropout': 0.01},
        {'class': 'SkipConnection', 'name': 'a'}]
    layers = HiddenLayers(hidden_layers)
    assert len(layers.layers) == 8

    skip_layers = [x for x in layers.layers if isinstance(x, SkipConnection)]
    assert len(skip_layers) == 2
    assert id(skip_layers[0]) == id(skip_layers[1])

    x = np.ones((5, 3))
    cache = None
    x_input = None

    for i, layer in enumerate(layers):
        if i == 3:  # skip start
            cache = copy.deepcopy(x)
        elif i == 7:  # skip end
            x_input = copy.deepcopy(x)

        x = layer(x)

        if i == 3:  # skip start
            assert layer._cache is not None
        elif i == 7:  # skip end
            assert layer._cache is None
            assert tf.reduce_all(x == tf.add(x_input, cache))


@pytest.mark.parametrize(
    ('t_mult', 's_mult'),
    ((1, 1),
     (2, 1),
     (1, 2),
     (2, 2),
     (3, 2),
     (5, 3)))
def test_st_expansion(t_mult, s_mult):
    """Test the spatiotemporal expansion layer."""
    layer = SpatioTemporalExpansion(spatial_mult=s_mult, temporal_mult=t_mult)
    n_filters = 2 * s_mult**2
    x = np.ones((123, 10, 10, 24, n_filters))
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == s_mult * x.shape[1]
    assert y.shape[2] == s_mult * x.shape[2]
    assert y.shape[3] == t_mult * x.shape[3]
    assert y.shape[4] == x.shape[4] / (s_mult**2)


def test_st_expansion_bad():
    """Test an illegal spatial expansion request."""
    layer = SpatioTemporalExpansion(spatial_mult=2, temporal_mult=2)
    x = np.ones((123, 10, 10, 24, 3))
    with pytest.raises(RuntimeError):
        _ = layer(x)


@pytest.mark.parametrize(
    ('hidden_layers'),
    (([{'class': 'FlexiblePadding', 'paddings': [[1, 1], [2, 2]],
        'mode': 'REFLECT'}]),
     ([{'class': 'FlexiblePadding', 'paddings': [[1, 1], [2, 2]],
        'mode': 'CONSTANT'}]),
     ([{'class': 'FlexiblePadding', 'paddings': [[1, 1], [2, 2]],
        'mode': 'SYMMETRIC'}])))
def test_flexible_padding(hidden_layers):
    """Test flexible padding routine"""
    layer = HiddenLayers(hidden_layers).layers[0]
    t = tf.constant([[1, 2, 3],
                     [4, 5, 6]])
    if layer.mode == 'CONSTANT':
        t_check = tf.constant([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 2, 3, 0, 0],
                               [0, 0, 4, 5, 6, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]])
    elif layer.mode == 'REFLECT':
        t_check = tf.constant([[6, 5, 4, 5, 6, 5, 4],
                               [3, 2, 1, 2, 3, 2, 1],
                               [6, 5, 4, 5, 6, 5, 4],
                               [3, 2, 1, 2, 3, 2, 1]])
    elif layer.mode == 'SYMMETRIC':
        t_check = tf.constant([[2, 1, 1, 2, 3, 3, 2],
                               [2, 1, 1, 2, 3, 3, 2],
                               [5, 4, 4, 5, 6, 6, 5],
                               [5, 4, 4, 5, 6, 6, 5]])
    tf.assert_equal(layer(t), t_check)
