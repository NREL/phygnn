"""
Tests for basic phygnn functionality and execution.
"""
# pylint: disable=W0613
import numpy as np
import pandas as pd

from phygnn.model_interfaces.random_forest_model import RandomForestModel


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


def test_random_forest():
    """Test the RandomForestModel """
    model = RandomForestModel.train(features, labels)

    test_mae = np.mean(np.abs(model[X].values.ravel() - Y))
    assert test_mae < 0.4
