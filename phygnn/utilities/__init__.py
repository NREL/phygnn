# -*- coding: utf-8 -*-
"""Utilities"""
import h5py
import tensorflow as tf
from packaging import version

from .pre_processing import PreProcess
from .tf_utilities import tf_isin, tf_log10

TF_VERSION = version.parse(tf.__version__)
TF2 = TF_VERSION >= version.parse('2.0.0')

H5PY_VERSION = version.parse(h5py.__version__)
H5PY3 = H5PY_VERSION >= version.parse('3.0.0')
