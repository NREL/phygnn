# -*- coding: utf-8 -*-
"""Utilities"""
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import rex
import h5py
from packaging import version

from .pre_processing import PreProcess
from .tf_utilities import tf_isin, tf_log10
from phygnn.version import __version__

TF_VERSION = version.parse(tf.__version__)
TF2 = TF_VERSION >= version.parse('2.0.0')

H5PY_VERSION = version.parse(h5py.__version__)
H5PY3 = H5PY_VERSION >= version.parse('3.0.0')

VERSION_RECORD = {'phygnn': __version__,
                  'tensorflow': tf.__version__,
                  'sklearn': sklearn.__version__,
                  'pandas': pd.__version__,
                  'numpy': np.__version__,
                  'nrel-rex': rex.__version__,
                  'python': sys.version,
                  }
