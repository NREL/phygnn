# -*- coding: utf-8 -*-
"""Utilities"""
from .pre_processing import PreProcess
from .tf_layers import Layers, HiddenLayers
from .tf_utilities import tf_isin, tf_log10
import tensorflow as tf
from packaging import version

TF_VERSION = version.parse(tf.__version__)
TF2 = TF_VERSION >= version.parse('2.0.0')
