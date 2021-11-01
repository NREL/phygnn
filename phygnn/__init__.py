# -*- coding: utf-8 -*-
"""Physics Guided Neural Network python library."""
import os
from .model_interfaces import PhygnnModel, RandomForestModel, TfModel
from .base import CustomNetwork, GradientUtils
from .phygnn import PhysicsGuidedNeuralNetwork
from .layers import Layers, HiddenLayers
from .utilities import PreProcess, tf_isin, tf_log10
from phygnn.version import __version__

__author__ = """Grant Buster"""
__email__ = "grant.buster@nrel.gov"

PHYGNNDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(PHYGNNDIR), 'tests', 'data')
