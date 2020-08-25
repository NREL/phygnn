# -*- coding: utf-8 -*-
"""Physics Guided Neural Network python library."""
import os
from .model_interfaces import TfModel
from .phygnn import PhysicsGuidedNeuralNetwork
from .utilities import tf_isin, tf_log10

PHYGNNDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(PHYGNNDIR), 'tests', 'data')
