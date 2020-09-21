"""
Tests for basic phygnn functionality and execution.
"""
# pylint: disable=W0613
import numpy as np
import os
import pandas as pd
import shutil

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
features = pd.DataFrame(X, columns=['a', 'b'])

Y_NOISE = Y * (1 + (np.random.random(Y.shape) - 0.5) * 0.5) + 0.1
labels = pd.DataFrame(Y_NOISE, columns=['c'])


def test_random_forest():
    """Test the RandomForestModel """
    model = RandomForestModel.build_trained(features, labels)

    test_mae = np.mean(np.abs(model[X].values.ravel() - Y))
    assert test_mae < 0.4


def test_save_load():
    """Test the save/load operations of RandomForestModel"""
    model = RandomForestModel.build_trained(features, labels,
                                            save_path=FPATH)
    y_pred = model[X]

    loaded = RandomForestModel.load(FPATH)
    loaded.train_model(features, labels)
    y_pred_loaded = loaded[X]
    np.allclose(y_pred.values, y_pred_loaded.values)
    assert loaded.feature_names == ['a', 'b']
    assert loaded.label_names == ['c']
    shutil.rmtree(os.path.dirname(FPATH))
