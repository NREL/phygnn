# -*- coding: utf-8 -*-
"""Custom tf layers."""
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class FlexiblePadding(tf.keras.layers.Layer):
    """Class to perform padding on tensors
    """
    def __init__(self, paddings, mode='REFLECT'):
        """
        Parameters
        ----------
        paddings : int array
            Integer array with shape [n,2] where n is the
            rank of the tensor and elements give the number
            of leading and trailing pads
        mode : str
            tf.pad() padding mode. Can be REFLECT, CONSTANT,
            or SYMMETRIC
        """
        super().__init__()
        self.paddings = tf.constant(paddings)
        self.rank = len(paddings)
        self.mode = mode

    def compute_output_shape(self, input_shape):
        """computes output shape after padding

        Parameters
        ----------
        input_shape : tuple
            shape of input tensor

        Returns
        -------
        output_shape : tf.TensorShape
            shape of padded tensor
        """
        output_shape = [0] * self.rank
        for d in range(self.rank):
            output_shape[d] = sum(self.paddings[d]) + input_shape[d]
        return tf.TensorShape(output_shape)

    def call(self, x):
        """calls the padding routine

        Parameters
        ----------
        x : tf.Tensor
            tensor on which to perform padding

        Returns
        -------
        x : tf.Tensor
            padded tensor with shape given
            by compute_output_shape

        """
        return tf.pad(x, self.paddings,
                      mode=self.mode)


class ExpandDims(tf.keras.layers.Layer):
    """Layer to add an extra dimension to a tensor."""

    def __init__(self, axis=3):
        """
        Parameters
        ----------
        axis : int
            Target axis at which to expand the shape of the input. Default is
            axis 3 based on creating a new temporal axis of the default
            spatiotemporal shape of: (n_observations, n_spatial_0, n_spatial_1,
            n_temporal, n_features)
        """
        super().__init__()
        self._axis = axis

    def call(self, x):
        """calls the expand dims operation

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with an extra dimension based on the init axes arg
        """
        return tf.expand_dims(x, axis=self._axis)


class TileTensor(tf.keras.layers.Layer):
    """Layer to tile (repeat) data across a given axis."""

    def __init__(self, multiples):
        """
        Parameters
        ----------
        multiples : list
            This is a list with the same length as number of dimensions in the
            input tensor. Each entry in the list determines how many times to
            tile each axis in the tensor.
        """
        super().__init__()
        self._mult = tf.constant(multiples, tf.int32)

    def call(self, x):
        """calls the tile operation

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with the specified axes tiled into larger shapes
            based on the multiples initialization argument.
        """
        return tf.tile(x, self._mult)


class FlattenAxis(tf.keras.layers.Layer):
    """Layer to flatten an axis from a 5D spatiotemporal Tensor into axis-0
    observations."""

    def __init__(self, axis=3):
        """
        Parameters
        ----------
        axis : int
            Target axis that holds the dimension to be flattened into the
            axis-0 dimension. Default is axis 3 based on flatteneing the
            temporal axis of the default spatiotemporal shape of:
            (n_observations, n_spatial_0, n_spatial_1, n_temporal, n_features)
        """
        super().__init__()
        self._axis = axis

    @staticmethod
    def _check_shape(input_shape):
        """Assert that the shape of the input tensor is the expected 5D
        spatiotemporal shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        msg = ('Input to FlattenAxis must be 5D with dimensions: '
               '(n_observations, n_spatial_0, n_spatial_1, n_temporal, '
               'n_features), but received shape: {}'.format(input_shape))
        assert len(input_shape) == 5, msg

    def call(self, x):
        """calls the flatten axis operation

        Parameters
        ----------
        x : tf.Tensor
            5D spatiotemporal tensor with dimensions:
            (n_observations, n_spatial_0, n_spatial_1, n_temporal, n_features)

        Returns
        -------
        x : tf.Tensor
            4D spatiotemporal tensor with target axis flattened into axis 0
        """
        self._check_shape(x.shape)
        return tf.concat(tf.unstack(x, axis=self._axis), axis=0)


class SpatialExpansion(tf.keras.layers.Layer):
    """Class to expand the spatial dimensions of tensors with shape:
    (n_observations, n_spatial_0, n_spatial_1, n_features)
    """

    def __init__(self, spatial_mult=1):
        """
        Parameters
        ----------
        spatial_multiplier : int
            Number of times to multiply the spatial dimensions. Note that the
            spatial expansion is an un-packing of the feature dimension. For
            example, if the input layer has shape (123, 5, 5, 16) with
            multiplier=2 the output shape will be (123, 10, 10, 4). The
            input feature dimension must be divisible by the spatial multiplier
            squared.
        """
        super().__init__()
        self._spatial_mult = int(spatial_mult)

    @staticmethod
    def _check_shape(input_shape):
        """Assert that the shape of the input tensor is the expected 4D
        spatiotemporal shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        msg = ('Input to SpatialExpansion must be 4D with dimensions: '
               '(n_observations, n_spatial_0, n_spatial_1, n_features), '
               'but received shape: {}'.format(input_shape))
        assert len(input_shape) == 4, msg

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self._check_shape(input_shape)

    def _spatial_expand(self, x):
        """Expand the two spatial dimensions (axis=1,2) of a 4D tensor using
        data from the last axes"""
        check_shape = x.shape[-1] % self._spatial_mult**2
        if check_shape != 0:
            msg = ('Spatial expansion of factor {} is being attempted on '
                   'input tensor of shape {}, but the last dimension of the '
                   'input tensor ({}) must be divisible by the spatial '
                   'factor squared ({}).'
                   .format(self._spatial_mult, x.shape, x.shape[-1],
                           self._spatial_mult**2))
            logger.error(msg)
            raise RuntimeError(msg)

        return tf.nn.depth_to_space(x, self._spatial_mult)

    def call(self, x):
        """Call the custom SpatialExpansion layer

        Parameters
        ----------
        x : tf.Tensor
            4D spatial tensor
            (n_observations, n_spatial_0, n_spatial_1, n_features)

        Returns
        -------
        x : tf.Tensor
            4D spatiotemporal tensor with axes 1,2 expanded (if spatial_mult>1)
        """
        self._check_shape(x.shape)

        if self._spatial_mult > 1:
            x = self._spatial_expand(x)

        return x


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
            Number of times to multiply the spatial dimensions. Note that the
            spatial expansion is an un-packing of the feature dimension. For
            example, if the input layer has shape (123, 5, 5, 24, 16) with
            multiplier=2 the output shape will be (123, 10, 10, 24, 4). The
            input feature dimension must be divisible by the spatial multiplier
            squared.
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

    def _temporal_expand(self, x):
        """Expand the temporal dimension (axis=3) of a 5D tensor"""
        temp_expand_shape = tf.stack(
            [x.shape[2], x.shape[3] * self._temporal_mult])
        out = []
        for x_unstack in tf.unstack(x, axis=1):
            out.append(tf.image.resize(x_unstack, temp_expand_shape,
                       method=self._temporal_meth))

        return tf.stack(out, axis=1)

    def _spatial_expand(self, x):
        """Expand the two spatial dimensions (axis=1,2) of a 5D tensor using
        data from the last axes"""
        check_shape = x.shape[-1] % self._spatial_mult**2
        if check_shape != 0:
            msg = ('Spatial expansion of factor {} is being attempted on '
                   'input tensor of shape {}, but the last dimension of the '
                   'input tensor ({}) must be divisible by the spatial '
                   'factor squared ({}).'
                   .format(self._spatial_mult, x.shape, x.shape[-1],
                           self._spatial_mult**2))
            logger.error(msg)
            raise RuntimeError(msg)

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
