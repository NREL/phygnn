"""
Tests for basic phygnn functionality and execution.
"""
import numpy as np
import pandas as pd

from phygnn import PhysicsGuidedNeuralNetwork

x1 = np.arange(500) - 250
x2 = np.arange(500) - 250
x1, x2 = np.meshgrid(x1, x2)
x1 = x1.flatten()
x2 = x2.flatten()
x3 = x1 * x2
features = pd.DataFrame({'x1': x1, 'x2': x2})

y = ((x1 * x2) > 0).astype(bool).astype(float)
labels = pd.DataFrame({'y': y})

hidden_layers = [{'units': 16},
                 {'activation': 'relu'},
                 {'units': 16},
                 {'activation': 'relu'},
                 ]
output_layer = [{'units': 1},
                {'activation': 'sigmoid'},
                ]


def test_classification():
    """Test the phygnn model as a classifier without the pfun"""
    PhysicsGuidedNeuralNetwork.seed(0)
    model = PhysicsGuidedNeuralNetwork(p_fun=None,
                                       hidden_layers=hidden_layers,
                                       output_layer=output_layer,
                                       loss_weights=(1.0, 0.0),
                                       metric='binary_crossentropy',
                                       learning_rate=0.05,
                                       n_features=2, n_labels=1)
    model.fit(features, labels, features, n_batch=1, n_epoch=50)

    y_pred = model.predict(features)
    accuracy = 100 * (np.round(y_pred) == labels.values).sum() / len(labels)
    assert accuracy > 0.99
