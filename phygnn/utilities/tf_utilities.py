# -*- coding: utf-8 -*-
"""
Tensorflow utilities
"""

import tensorflow as tf


def tf_isin(a, b):
    """Check whether a is in b"""
    assert isinstance(b, (list, tuple)), 'Second arg must be a list or tuple!'
    x = [tf.equal(a, i) for i in b]
    out = tf.reduce_any(tf.stack(x), axis=0)
    assert out.shape == a.shape
    return out


def tf_log10(x):
    """Compute log base 10 of a tensor x"""
    num = tf.math.log(x)
    den = tf.math.log(tf.constant(10, dtype=num.dtype))
    return num / den


def _mean_fill(x, mask):
    """Fill NaN values in the input tensor with the mean of the non-NaN.
    If all values are NaN, fill with 0. This is assumed to be either a 3D
    or 4D tensor, with the trailing feature channel removed.

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (n_obs, spatial_1, spatial_2) or
        (n_obs, spatial_1, spatial_2, n_temporal)
    mask : tf.Tensor
        Boolean mask of the non-NaN locations in the original x

    Returns
    -------
    x : tf.Tensor
        Input tensor with NaN values filled with the mean of the non-NaN
        values or 0 if all values are NaN.
    mask : tf.Tensor
        Mask of the input tensor where True is not NaN and False is NaN.
    """
    not_nan_float = tf.cast(mask, x.dtype)

    hi_res_zeroed = tf.where(mask, x, tf.zeros_like(x))

    count = tf.reduce_sum(not_nan_float)
    total = tf.reduce_sum(hi_res_zeroed)
    mean = tf.math.divide_no_nan(total, count)

    return tf.where(mask, x, mean)


def mean_fill(x):
    """Fill NaN values in the input tensor with the mean of the non-NaN.
    If all values are NaN, fill with 0. This is assumed to be either a 4D
    or 5D tensor, with the trailing feature channel included.

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (n_obs, spatial_1, spatial_2, n_features) or
        (n_obs, spatial_1, spatial_2, n_temporal, n_features)

    Returns
    -------
    x_filled : tf.Tensor
        Input tensor with NaN values filled with the mean of the non-NaN
        values or 0 if all values are NaN.
    mask : tf.Tensor
        Mask of the input tensor where 1 is not NaN and 0 is NaN.
    """
    mask = tf.math.logical_not(tf.math.is_nan(x))
    hr_feat = [_mean_fill(x[..., i], mask[..., i]) for i in range(x.shape[-1])]
    hr_feat = tf.stack(hr_feat, axis=-1)
    return hr_feat, tf.cast(mask, tf.float32)


def idw_flat_fill(x, coords, mask):
    """IDW fill for flattened tensors without depth dimension. Assumes input
    shape is (N), where N is the number of points (H * W).

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (N)
    coords: tf.Tensor
        Coordinates of the points in the image, shape (N, 2) where N is the
        number of points (H * W)
    mask : tf.Tensor
        Mask of the input tensor where True is not NaN and False is NaN.

    Returns
    -------
    x_filled: tf.Tensor
        Flattened tensor with NaNs filled
    """
    nan_mask = tf.math.logical_not(mask)
    valid_coords = tf.boolean_mask(coords, mask)
    nan_coords = tf.boolean_mask(coords, nan_mask)
    valid_vals = tf.boolean_mask(x, mask)

    diffs = tf.expand_dims(nan_coords, 1) - tf.expand_dims(valid_coords, 0)
    dists = tf.norm(diffs, axis=-1)

    weights = 1 / dists
    weights /= tf.reduce_sum(weights, axis=1, keepdims=True)

    vals = tf.reduce_sum(weights * tf.expand_dims(valid_vals, 0), axis=1)

    filled = tf.where(mask, x, tf.zeros_like(x))
    return tf.tensor_scatter_nd_update(filled, tf.where(nan_mask), vals)


def idw_batch_fill(x, coords, mask):
    """IDW fill for tensors without batch dimension. Assumes input shape is:
    - (H, W) or (H, W, D)

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (H, W) or (H, W, D)
    coords: tf.Tensor
        Coordinates of the points in the image, shape (N, 2) where N is the
        number of points (H * W)
    mask : tf.Tensor
        Mask of the input tensor where True is not NaN and False is NaN.

    Returns
    -------
    x_filled: tf.Tensor
        Tensor with NaNs filled
    """
    rank = len(x.shape)
    x = tf.expand_dims(x, -1) if rank == 2 else x
    H, W, D = x.shape[0], x.shape[1], x.shape[2]

    x_flat = tf.reshape(x, [H * W, D])
    is_valid = tf.reshape(mask, [H * W, D])
    out = [
        idw_flat_fill(x_flat[:, d], coords, is_valid[:, d]) for d in range(D)
    ]
    out = tf.stack(out, axis=-1)
    out = tf.reshape(out, [H, W, D])
    return tf.squeeze(out, axis=-1) if rank == 2 else tf.cast(out, tf.float32)


def _idw_fill(x, coords, mask):
    """IDW fill for tensors without channel dimension. Assumes input shape is
    (n_obs, spatial_1, spatial_2) or (n_obs, spatial_1, spatial_2, n_temporal)

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (n_obs, spatial_1, spatial_2) or
        (n_obs, spatial_1, spatial_2, n_temporal)
    coords: tf.Tensor
        Coordinates of the points in the image, shape (N, 2) where N is the
        number of points (spatial_1 * spatial_2)
    mask : tf.Tensor
        Mask of the input tensor where True is not NaN and False is NaN.

    Returns
    -------
    x_filled: tf.Tensor
        Tensor with NaNs filled
    """
    rank = len(x.shape)
    x = tf.expand_dims(x, -1) if rank == 3 else x
    res = [idw_batch_fill(x[b], coords, mask[b]) for b in range(x.shape[0])]
    out = tf.stack(res, axis=0)
    return tf.squeeze(out, axis=-1) if rank == 3 else tf.cast(out, tf.float32)


def idw_fill(x):
    """IDW fill for tensors with channel dimension. Assumes input shape
    is (n_obs, spatial_1, spatial_2, n_features) or (n_obs, spatial_1,
    spatial_2, n_temporal, n_features)

    Parameters
    ----------
    x : tf.Tensor
        Tensor with NaNs and shape (n_obs, spatial_1, spatial_2, n_features) or
        (n_obs, spatial_1, spatial_2, n_temporal, n_features)

    Returns
    -------
    x_filled: tf.Tensor
        Tensor with NaNs filled
    mask : tf.Tensor
        Mask of the input tensor where 1 is not NaN and 0 is NaN.
    """
    assert len(x.shape) in [4, 5], 'Input tensor must be 4D or 5D'
    x = tf.cast(x, tf.float32)
    H, W = x.shape[1], x.shape[2]
    N = H * W
    mask = tf.math.logical_not(tf.math.is_nan(x))
    coords = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
    coords = tf.stack(coords, axis=-1)
    coords = tf.cast(tf.reshape(coords, [N, 2]), tf.float32)
    filled_list = [
        _idw_fill(x[..., c], coords, mask[..., c]) for c in range(x.shape[-1])
    ]
    filled = tf.stack(filled_list, axis=-1)
    return tf.cast(filled, tf.float32), tf.cast(mask, tf.float32)
