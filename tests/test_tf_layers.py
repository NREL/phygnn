"""
Test the custom tensorflow utilities
"""
import pytest

from phygnn.layers.layers import Layers, HiddenLayers


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
