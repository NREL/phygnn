# -*- coding: utf-8 -*-
"""Custom tf layers."""
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class FlexiblePadding(tf.keras.layers.Layer):
    """Class to perform padding on tensors
    """

    def __init__(self, paddings, mode='REFLECT', option='tf'):
        """
        Parameters
        ----------
        paddings : int array
            Integer array with shape [n,2] where n is the
            rank of the tensor and elements give the number
            of leading and trailing pads
        mode : str
            tf.pad() / np.pad() padding mode. Can be REFLECT, CONSTANT,
            or SYMMETRIC
        option : str
            Option for TensorFlow padding ("tf") or numpy ("np"). Default is tf
            for tensorflow training. We have observed silent failures of
            tf.pad() with larger array sizes, so "np" might be preferable at
            inference time on large chunks, but it is much slower when it has
            to convert tensors to numpy arrays.
        """
        super().__init__()
        self.paddings = tf.constant(paddings)
        self.rank = len(paddings)
        self.mode = mode.lower()
        self.option = option.lower()

        if self.option == 'tf':
            self._pad_fun = tf.pad
        elif self.option == 'np':
            self._pad_fun = np.pad
        else:
            msg = ('FlexiblePadding option must be "tf" or "np" but '
                   f'received: {self.option}')
            logger.error(msg)
            raise KeyError(msg)

    def compute_output_shape(self, input_shape):
        """Computes output shape after padding

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
        """Calls the padding routine

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
        return self._pad_fun(x, self.paddings, mode=self.mode)


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
        """Calls the expand dims operation

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


class TileLayer(tf.keras.layers.Layer):
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
        """Calls the tile operation

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


class GaussianAveragePooling2D(tf.keras.layers.Layer):
    """Custom layer to implement tensorflow average pooling layer but with a
    gaussian kernel. This is basically a gaussian smoothing layer with a fixed
    convolution window that limits the area of effect"""

    def __init__(self, pool_size, strides=None, padding='valid', sigma=1,
                 trainable=True, **kwargs):
        """
        Parameters
        ----------
        pool_size: integer
            Pooling window size. This sets the number of pixels in each
            dimension that will be averaged into an output pixel. Only one
            integer is specified, the same window length will be used for both
            dimensions. For example, if ``pool_size=2`` and ``strides=2`` then
            the output dimension will be half of the input.
        strides: Integer, tuple of 2 integers, or None.
            Strides values. If None, it will default to `pool_size`.
        padding: One of `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the
            same height/width dimension as the input.
        sigma : float
            Sigma parameter for gaussian distribution
        trainable : bool
            Flag for whether sigma is trainable weight or not.
        kwargs : dict
            Extra kwargs for tf.keras.layers.Layer
        """

        super().__init__(**kwargs)
        assert isinstance(pool_size, int), 'pool_size must be int!'
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.upper()
        self.trainable = trainable
        self.sigma = sigma

    # pylint: disable=unused-argument
    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Initializes the trainable sigma variable

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        if not any(self.weights):
            init = tf.keras.initializers.Constant(value=self.sigma)
            self.sigma = self.add_weight("sigma", shape=[1],
                                         trainable=self.trainable,
                                         dtype=tf.float32,
                                         initializer=init)

    def make_kernel(self):
        """Creates 2D gaussian kernel with side length `self.pool_size` and a
        sigma of `sigma`

        Returns
        -------
        kernel : np.ndarray
            2D kernel with shape (self.pool_size, self.pool_size)
        """
        ax = tf.linspace(-(self.pool_size - 1) / 2.,
                         (self.pool_size - 1) / 2.,
                         self.pool_size)
        gauss = tf.math.exp(-0.5 * tf.math.square(ax)
                            / tf.math.square(self.sigma))
        kernel = tf.expand_dims(gauss, 0) * tf.expand_dims(gauss, -1)
        kernel = kernel / tf.math.reduce_sum(kernel)
        kernel = tf.expand_dims(kernel, -1)
        kernel = tf.expand_dims(kernel, -1)
        return kernel

    def get_config(self):
        """Implementation of get_config method from tf.keras.layers.Layer for
        saving/loading as part of keras sequential model.

        Returns
        -------
        config : dict
        """
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            'trainable': self.trainable,
            'sigma': float(self.sigma),
        })
        return config

    def call(self, x):
        """Operates on x with the specified function

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor operated on by the specified function
        """

        kernel = self.make_kernel()

        out = []
        for idf in range(x.shape[-1]):
            fslice = slice(idf, idf + 1)
            iout = tf.nn.convolution(x[..., fslice], kernel,
                                     strides=self.strides,
                                     padding=self.padding)
            out.append(iout)
        out = tf.concat(out, -1, name='concat')
        return out


class GaussianNoiseAxis(tf.keras.layers.Layer):
    """Layer to apply random noise along a given axis."""

    def __init__(self, axis, mean=1, stddev=0.1):
        """
        Parameters
        ----------
        axis : int | list | tuple
            Axes to apply random noise across. All other axes will have the
            same noise. For example, for a 5D spatiotemporal tensor with
            axis=(1, 2, 3) (both spatial axes and the temporal axis), this
            layer will apply a single random number to every unique index of
            axis=(1, 2, 3).
        mean : float
            The mean of the normal distribution.
        stddev : float
            The standard deviation of the normal distribution.
        """

        super().__init__()
        self.rank = None
        self._axis = axis if isinstance(axis, (tuple, list)) else [axis]
        self._mean = tf.constant(mean, dtype=tf.dtypes.float32)
        self._stddev = tf.constant(stddev, dtype=tf.dtypes.float32)

    def _get_rand_shape(self, x):
        """Get shape of random noise along the specified axes."""
        shape = np.ones(len(x.shape), dtype=np.int32)
        for ax in self._axis:
            shape[ax] = x.shape[ax]
        return tf.constant(shape, dtype=tf.dtypes.int32)

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Sets the shape of the random noise along the specified axis

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self.rank = len(input_shape)

    def call(self, x):
        """Calls the tile operation

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with noise applied to the requested axis.
        """

        rand_tensor = tf.random.normal(self._get_rand_shape(x),
                                       mean=self._mean,
                                       stddev=self._stddev,
                                       dtype=tf.dtypes.float32)
        return x + rand_tensor


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
        """Calls the flatten axis operation

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
        spatial_mult : int
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
                 temporal_method='nearest', t_roll=0):
        """
        Parameters
        ----------
        spatial_mult : int
            Number of times to multiply the spatial dimensions. Note that the
            spatial expansion is an un-packing of the feature dimension. For
            example, if the input layer has shape (123, 5, 5, 24, 16) with
            multiplier=2 the output shape will be (123, 10, 10, 24, 4). The
            input feature dimension must be divisible by the spatial multiplier
            squared.
        temporal_mult : int
            Number of times to multiply the temporal dimension. For example,
            if the input layer has shape (123, 5, 5, 24, 2) with multiplier=2
            the output shape will be (123, 5, 5, 48, 2).
        temporal_method : str
            Interpolation method for tf.image.resize(). Can also be
            "depth_to_time" for an operation similar to tf.nn.depth_to_space
            where the feature axis is unpacked into the temporal axis.
        t_roll : int
            Option to roll the temporal axis after expanding. When using
            temporal_method="depth_to_time", the default (t_roll=0) will add
            temporal steps after the input steps such that if input temporal
            shape is 3 and the temporal_mult is 24x, the output will have the
            index-0 timesteps at idt=0,24,48 but if t_roll=12, the output will
            have the original timesteps at idt=12,36,60. This is no longer
            recommended, as a positive roll will move the features of timestep
            -1 from the end of the series to the beginning.
        """

        super().__init__()
        self._spatial_mult = int(spatial_mult)
        self._temporal_mult = int(temporal_mult)
        self._temporal_meth = temporal_method
        self._t_roll = t_roll

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

        if self._temporal_meth == 'depth_to_time':
            check_shape = x.shape[-1] % self._temporal_mult
            if check_shape != 0:
                msg = ('Temporal expansion of factor {} is being attempted on '
                       'input tensor of shape {}, but the last dimension of '
                       'the input tensor ({}) must be divisible by the '
                       'temporal factor ({}).'
                       .format(self._temporal_mult, x.shape, x.shape[-1],
                               self._temporal_mult))
                logger.error(msg)
                raise RuntimeError(msg)

            shape = (x.shape[0], x.shape[1], x.shape[2],
                     x.shape[3] * self._temporal_mult,
                     x.shape[4] // self._temporal_mult)
            out = tf.reshape(x, shape)
            out = tf.roll(out, self._t_roll, axis=3)

        else:
            temp_expand_shape = tf.stack([x.shape[2],
                                          x.shape[3] * self._temporal_mult])
            out = []
            for x_unstack in tf.unstack(x, axis=1):
                out.append(tf.image.resize(x_unstack, temp_expand_shape,
                           method=self._temporal_meth))
            out = tf.stack(out, axis=1)

        return out

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
        super().__init__(name=name)
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
        try:
            out = tf.add(x, self._cache)
        except Exception as e:
            msg = ('Could not add SkipConnection "{}" data cache of '
                   'shape {} to input of shape {}.'
                   .format(self._name, self._cache.shape, x.shape))
            logger.error(msg)
            raise RuntimeError(msg) from e
        else:
            self._cache = None
            return out


class SqueezeAndExcitation(tf.keras.layers.Layer):
    """Custom layer for squeeze and excitation block for convolutional networks

    Note that this is only set up to take a channels-last conv output

    References
    ----------
    1. Hu, Jie, et al. Squeeze-and-Excitation Networks. arXiv:1709.01507,
       arXiv, 16 May 2019, http://arxiv.org/abs/1709.01507.
    2. Pröve, Paul-Louis. “Squeeze-and-Excitation Networks.” Medium, 18 Oct.
       2017,
    https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
    """

    def __init__(self, ratio=16):
        """
        Parameters
        ----------
        ratio : int
            Number of convolutional channels/filters divided by the number of
            dense connections in the SE block.
        """

        super().__init__()
        self._ratio = ratio
        self._n_channels = None
        self._dense_units = None
        self._hidden_layers = None

    def build(self, input_shape):
        """Build the SqueezeAndExcitation layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """

        self._n_channels = input_shape[-1]
        self._dense_units = int(np.ceil(self._n_channels / self._ratio))

        if len(input_shape) == 4:
            pool_layer = tf.keras.layers.GlobalAveragePooling2D()
        elif len(input_shape) == 5:
            pool_layer = tf.keras.layers.GlobalAveragePooling3D()
        else:
            msg = ('SqueezeAndExcitation layer can only accept 4D or 5D data '
                   'for image or video input but received input shape: {}'
                   .format(input_shape))
            logger.error(msg)
            raise RuntimeError(msg)

        self._hidden_layers = [
            pool_layer,
            tf.keras.layers.Dense(self._dense_units, activation='relu'),
            tf.keras.layers.Dense(self._n_channels, activation='sigmoid'),
            tf.keras.layers.Multiply()]

    def call(self, x):
        """Call the custom SqueezeAndExcitation layer

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is the squeeze-and-excitation weights
            multiplied by the original input tensor x
        """

        t_in = x
        for layer in self._hidden_layers[:-1]:
            x = layer(x)

        # multiply layer
        x = self._hidden_layers[-1]([t_in, x])

        return x


class FNO(tf.keras.layers.Layer):
    """Custom layer for fourier neural operator block

    Note that this is only set up to take a channels-last input

    References
    ----------
    1. FourCastNet: A Global Data-driven High-resolution Weather Model using
    Adaptive Fourier Neural Operators. http://arxiv.org/abs/2202.11214
    2. Adaptive Fourier Neural Operators: Efficient Token Mixers for
    Transformers. http://arxiv.org/abs/2111.13587
    """

    def __init__(self, filters, sparsity_threshold=0.5, activation='relu'):
        """
        Parameters
        ----------
        filters : int
            Number of dense connections in the FNO block.
        sparsity_threshold : float
            Parameter to control sparsity and shrinkage in the softshrink
            activation function following the MLP layers.
        activation : str
            Activation function used in MLP layers.
        """

        super().__init__()
        self._filters = filters
        self._fft_layer = None
        self._ifft_layer = None
        self._mlp_layers = None
        self._activation = activation
        self._n_channels = None
        self._perms_in = None
        self._perms_out = None
        self._lambd = sparsity_threshold

    def _softshrink(self, x):
        """Softshrink activation function

        https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
        """
        values_below_lower = tf.where(x < -self._lambd, x + self._lambd, 0)
        values_above_upper = tf.where(self._lambd < x, x - self._lambd, 0)
        return values_below_lower + values_above_upper

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x = tf.transpose(x, perm=self._perms_in)
        x = self._fft_layer(tf.cast(x, tf.complex64))
        x = tf.transpose(x, perm=self._perms_out)
        return x

    def _ifft(self, x):
        """Apply needed transpositions and ifft operation."""
        x = tf.transpose(x, perm=self._perms_in)
        x = self._ifft_layer(tf.cast(x, tf.complex64))
        x = tf.transpose(x, perm=self._perms_out)
        return x

    def build(self, input_shape):
        """Build the FNO layer based on an input shape

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input tensor
        """
        self._n_channels = input_shape[-1]
        dims = list(range(len(input_shape)))
        self._perms_in = [dims[-1], *dims[:-1]]
        self._perms_out = [*dims[1:], dims[0]]

        if len(input_shape) == 4:
            self._fft_layer = tf.signal.fft2d
            self._ifft_layer = tf.signal.ifft2d
        elif len(input_shape) == 5:
            self._fft_layer = tf.signal.fft3d
            self._ifft_layer = tf.signal.ifft3d
        else:
            msg = ('FNO layer can only accept 4D or 5D data '
                   'for image or video input but received input shape: {}'
                   .format(input_shape))
            logger.error(msg)
            raise RuntimeError(msg)

        self._mlp_layers = [
            tf.keras.layers.Dense(self._filters, activation=self._activation),
            tf.keras.layers.Dense(self._n_channels)]

    def _mlp_block(self, x):
        """Run mlp layers on input"""
        for layer in self._mlp_layers:
            x = layer(x)
        return x

    def call(self, x):
        """Call the custom FourierNeuralOperator layer

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        x : tf.Tensor
            Output tensor, this is the FNO weights added to the original input
            tensor.
        """
        t_in = x
        x = self._fft(x)
        x = self._mlp_block(x)
        x = self._softshrink(x)
        x = self._ifft(x)
        x = tf.cast(x, dtype=t_in.dtype)

        return x + t_in


class Sup3rAdder(tf.keras.layers.Layer):
    """Layer to add high-resolution data to a sup3r model in the middle of a
    super resolution forward pass."""

    def __init__(self, name=None):
        """
        Parameters
        ----------
        name : str | None
            Unique str identifier of the adder layer. Usually the name of the
            hi-resolution feature used in the addition.
        """
        super().__init__(name=name)

    @staticmethod
    def call(x, hi_res_adder):
        """Adds hi-resolution data to the input tensor x in the middle of a
        sup3r resolution network.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        hi_res_adder : tf.Tensor | np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) that can be added to x.

        Returns
        -------
        x : tf.Tensor
            Output tensor with the hi_res_adder added to x.
        """
        return x + hi_res_adder


class Sup3rConcatMasked(tf.keras.layers.Layer):
    """Layer to fix certain values for a sup3r model in the middle of a
    super resolution forward pass. This is used to condition models on sparse
    observation data. This uses the first channel of the input tensor as a
    background for the provided fixed values and then concatenates with the
    input tensor."""

    def __init__(self, name=None):
        """
        Parameters
        ----------
        name : str | None
            Unique str identifier of the fixer layer. Usually the name of the
            hi-resolution feature used in the fixing.
        """
        super().__init__(name=name)

    @staticmethod
    def call(x, hi_res_feature):
        """Combine the first channel of x and the non-nan data in
        hi_res_feature and concatenates with x.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        hi_feature : tf.Tensor | np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), 1) that can be used to fix values of x.

        Returns
        -------
        x : tf.Tensor
            Output tensor with the hi_res_fixer used to fix values of x.
        """
        mask = tf.math.is_nan(hi_res_feature[..., 0])
        fixed = tf.where(mask, x[..., 0], hi_res_feature[..., 0])
        return tf.concat((x, tf.expand_dims(fixed, axis=-1)), axis=-1)


class Sup3rConcat(tf.keras.layers.Layer):
    """Layer to concatenate a high-resolution feature to a sup3r model in the
    middle of a super resolution forward pass."""

    def __init__(self, name=None):
        """
        Parameters
        ----------
        name : str | None
            Unique str identifier for the concat layer. Usually the name of the
            hi-resolution feature used in the concatenation.
        """
        super().__init__(name=name)

    @staticmethod
    def call(x, hi_res_feature):
        """Concatenates a hi-resolution feature to the input tensor x in the
        middle of a sup3r resolution network.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        hi_res_feature : tf.Tensor | np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) that can be concatenated to x.

        Returns
        -------
        x : tf.Tensor
            Output tensor with the hi_res_feature added to x.
        """
        return tf.concat((x, hi_res_feature), axis=-1)


class FunctionalLayer(tf.keras.layers.Layer):
    """Custom layer to implement the tensorflow layer functions (e.g., add,
    subtract, multiply, maximum, and minimum) with a constant value. These
    cannot be implemented in phygnn as normal layers because they need to
    operate on two tensors of equal shape."""

    def __init__(self, name, value):
        """
        Parameters
        ----------
        name : str
            Name of the tensorflow layer function to be implemented, options
            are (all lower-case): add, subtract, multiply, maximum, and minimum
        value : float
            Constant value to use in the function operation
        """

        options = ('add', 'subtract', 'multiply', 'maximum', 'minimum')
        msg = (f'FunctionalLayer input `name` must be one of "{options}" '
               f'but received "{name}"')
        assert name in options, msg

        super().__init__(name=name)
        self.value = value
        self.fun = getattr(tf.keras.layers, self.name)

    def call(self, x):
        """Operates on x with the specified function

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor operated on by the specified function
        """
        const = tf.constant(value=self.value, shape=x.shape, dtype=x.dtype)
        return self.fun((x, const))


class SigLin(tf.keras.layers.Layer):
    """Sigmoid linear unit. This can be used to set a soft minimum on a range.

    y = 1/(1+exp(-x)) where x<0.5
    y = x + 0.5 where x>=0.5
    """

    @staticmethod
    def call(x):
        """Operates on x with SigLin

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        x : tf.Tensor
            Output tensor with same shape as input x operated on by SigLin
        """

        return tf.math.maximum(tf.math.sigmoid(x), x + 0.5)


class LogTransform(tf.keras.layers.Layer):
    """Log transform or inverse transform of data

    ``y = log(x + adder) * scalar`` or
    ``y = exp(x / scalar) - adder`` for the inverse
    """

    def __init__(self, name=None, adder=0, scalar=1, inverse=False, idf=None):
        """
        Parameters
        ----------
        name : str | None
            Name of the tensorflow layer
        adder : float
            Adder term for ``y = log(x + adder) * scalar``
        scalar : float
            Scalar term for ``y = log(x + adder) * scalar``
        inverse : bool
            Option to perform the inverse operation e.g.
            ``y = exp(x / scalar) - adder``
        idf : int | list | None
            One or more feature channel indices to perform log transform on.
            None will perform transform on all feature channels.
        """

        super().__init__(name=name)
        self.adder = adder
        self.scalar = scalar
        self.inverse = inverse
        self.rank = None
        self.idf = [idf] if isinstance(idf, int) else idf

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self.rank = len(input_shape)

    def _logt(self, x):
        if not self.inverse:
            return tf.math.log(x + self.adder) * self.scalar
        return tf.math.exp(x / self.scalar) - self.adder

    def call(self, x):
        """Operates on x with (inverse) log transform

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        y : tf.Tensor
            Log-transformed x tensor
        """

        if self.idf is None:
            return self._logt(x)
        out = []
        for idf in range(x.shape[-1]):
            if idf in self.idf:
                out.append(self._logt(x[..., idf:idf + 1]))
            else:
                out.append(x[..., idf:idf + 1])

        out = tf.concat(out, -1, name='concat')
        return out


class UnitConversion(tf.keras.layers.Layer):
    """Layer to convert units per feature channel using the linear transform:
    ``y = x * scalar + adder``

    Be sure to check how this will interact with normalization factors.
    """

    def __init__(self, name=None, adder=0, scalar=1):
        """
        Parameters
        ----------
        name : str | None
            Name of the tensorflow layer
        adder : float | list
            Adder term for ``y = x * scalar + adder``. If this is a float, the
            same value will be used for all feature channels. If this is a
            list, each value will be used for the corresponding feature channel
            and the length must match the number of feature channels
        scalar : float | list
            Scalar term for ``y = x * scalar + adder``. If this is a float, the
            same value will be used for all feature channels. If this is a
            list, each value will be used for the corresponding feature channel
            and the length must match the number of feature channels
        """

        super().__init__(name=name)
        self.adder = adder
        self.scalar = scalar
        self.rank = None

    def build(self, input_shape):
        """Custom implementation of the tf layer build method.

        Parameters
        ----------
        input_shape : tuple
            Shape tuple of the input
        """
        self.rank = len(input_shape)
        nfeat = input_shape[-1]

        dtypes = (int, np.int64, np.int32, float, np.float32, np.float64)

        if isinstance(self.adder, dtypes):
            self.adder = np.ones(nfeat) * self.adder
            self.adder = tf.convert_to_tensor(self.adder, dtype=tf.float32)
        else:
            msg = (f'UnitConversion layer `adder` array has length '
                   f'{len(self.adder)} but input shape has last dimension '
                   f'as {input_shape[-1]}')
            assert len(self.adder) == input_shape[-1], msg

        if isinstance(self.scalar, dtypes):
            self.scalar = np.ones(nfeat) * self.scalar
            self.scalar = tf.convert_to_tensor(self.scalar, dtype=tf.float32)
        else:
            msg = (f'UnitConversion layer `scalar` array has length '
                   f'{len(self.scalar)} but input shape has last dimension '
                   f'as {input_shape[-1]}')
            assert len(self.scalar) == input_shape[-1], msg

    def call(self, x):
        """Convert units

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        y : tf.Tensor
            Unit-converted x tensor
        """

        if self.rank is None:
            self.build(x.shape)

        out = []
        for idf, (adder, scalar) in enumerate(zip(self.adder, self.scalar)):
            out.append(x[..., idf:idf + 1] * scalar + adder)

        out = tf.concat(out, -1, name='concat')

        return out
