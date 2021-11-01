# -*- coding: utf-8 -*-
"""Custom tf layers."""
import logging
import tensorflow as tf


logger = logging.getLogger(__name__)


class SpatioTemporalExpansion(tf.keras.layers.Layer):
    """Class to expand the spatiotemporal dimensions of tensors with shape:
    (n_observations, n_spatial_0, n_spatial_1, n_temporal, n_features)
    """

    def __init__(self, spatial_mult=1, temporal_mult=1,
                 temporal_method='nearest'):
        """
        Parameters
        ----------
        spatial_multiplier : int
            Number of times to multiply the spatial dimensions. For example,
            if the input layer has shape (123, 5, 5, 24, 2) with multiplier=2
            the output shape will be (123, 10, 10, 24, 2).
        temporal_multiplier : int
            Number of times to multiply the temporal dimension. For example,
            if the input layer has shape (123, 5, 5, 24, 2) with multiplier=2
            the output shape will be (123, 5, 5, 48, 2).
        temporal_method : str
            Interpolation method for tf.image.resize().
        """
        super().__init__()
        self._spatial_mult = int(spatial_mult)
        self._temporal_mult = int(temporal_mult)
        self._temporal_meth = temporal_method
        self._n_spatial_1 = None
        self._n_temporal = None
        self._temp_expand_shape = None

    @staticmethod
    def _check_shape(input_shape):
        """Assert that the shape of the input tensor is the expected 5D
        spatiotemporal shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        msg = ('Input to SpatioTemporalExpansion must be 5D with dimensions: '
               '(n_observations, n_spatial_0, n_spatial_1, n_temporal, '
               'n_features), but received shape: {}'.format(input_shape))
        assert len(input_shape) == 5, msg

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self._check_shape(input_shape)

        # desired final shape of the 2nd and 3rd axes for temporal expansion
        self._n_spatial_1 = input_shape[2]
        self._n_temporal = input_shape[3]
        self._temp_expand_shape = tf.stack([
            self._n_spatial_1, self._n_temporal * self._temporal_mult])

    def _temporal_expand(self, x):
        """Expand the temporal dimension (axis=3) of a 5D tensor"""
        out = []
        for x_unstack in tf.unstack(x, axis=1):
            out.append(tf.image.resize(x_unstack, self._temp_expand_shape,
                       method=self._temporal_meth))

        return tf.stack(out, axis=1)

    def _spatial_expand(self, x):
        """Expand the two spatial dimensions (axis=1,2) of a 5D tensor"""
        out = []
        for x_unstack in tf.unstack(x, axis=3):
            out.append(tf.nn.depth_to_space(x_unstack, self._spatial_mult))

        return tf.stack(out, axis=3)

    def call(self, x):
        """Call the custom SpatioTemporalExpansion layer

        Parameters
        ----------
        x : tf.Tensor
            5D spatiotemporal tensor.

        Returns
        -------
        x : tf.Tensor
            5D spatiotemporal tensor with axes 1,2 expanded (if spatial_mult>1)
            and axes 3 expanded (if temporal_mult>1).
        """
        self._check_shape(x.shape)

        if self._temporal_mult > 1:
            x = self._temporal_expand(x)

        if self._spatial_mult > 1:
            x = self._spatial_expand(x)

        return x


class SkipConnection(tf.keras.layers.Layer):
    """Custom layer to implement a skip connection. This layer should be
    initialized and referenced in a layer list by the same name as both the
    skip start and skip end.
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            Unique string identifier of the skip connection. The skip endpoint
            should have the same name.
        """
        super().__init__()
        self._name = name
        self._cache = None

    def call(self, x):
        """Call the custom SkipConnection layer

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor. If this is the skip start, the input will be cached
            and returned without manipulation. If this is the skip endpoint,
            the output will be the input x added to the tensor cached at the
            skip start.
        """
        if self._cache is None:
            self._cache = x
            return x
        else:
            if x.shape != self._cache.shape:
                msg = ('The endpoint input tensor for SkipConnection "{}" had '
                       'shape {} but the cached data from the start of the '
                       'skip connection had shape {}.'
                       .format(self._name, x.shape, self._cache.shape))
                logger.error(msg)
                raise RuntimeError(msg)

            out = tf.add(x, self._cache)
            self._cache = None
            return out
