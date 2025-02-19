# -*- coding: utf-8 -*-
"""Utilities"""
import sys

import h5py
import numpy as np
import pandas as pd
import rex
import sklearn
import tensorflow as tf
from packaging import version

from phygnn.version import __version__

from .pre_processing import PreProcess
from .tf_utilities import tf_isin, tf_log10

TF_VERSION = version.parse(tf.__version__)
TF2 = version.parse('2.0.0') <= TF_VERSION

H5PY_VERSION = version.parse(h5py.__version__)
H5PY3 = version.parse('3.0.0') <= H5PY_VERSION

VERSION_RECORD = {'phygnn': __version__,
                  'tensorflow': tf.__version__,
                  'sklearn': sklearn.__version__,
                  'pandas': pd.__version__,
                  'numpy': np.__version__,
                  'nrel-rex': rex.__version__,
                  'python': sys.version,
                  }
