# -*- coding: utf-8 -*-
"""Physics Guided Neural Network python library."""
import os
from .model_interfaces import PhygnnModel, RandomForestModel, TfModel
from .phygnn import PhysicsGuidedNeuralNetwork, p_fun_dummy
from .utilities import Layers, HiddenLayers, PreProcess, tf_isin, tf_log10
from phygnn.version import __version__

PHYGNNDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(PHYGNNDIR), 'tests', 'data')
