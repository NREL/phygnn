"""
Test the custom tensorflow utilities
"""

import numpy as np
import pytest
import tensorflow as tf

from phygnn.utilities.tf_utilities import idw_fill, tf_isin, tf_log10


def test_tf_isin():
    """Test the tensorflow isin utility method"""
    a = np.arange(10)
    b = [4, 7, 8]
    tfa = tf.convert_to_tensor(a)
    c = np.isin(a, b)
    tfc = tf_isin(tfa, b)
    assert all(c == tfc.numpy())


def test_tf_log10():
    """Test the tensorflow log base 10 utility method"""
    a = np.arange(1, 10)
    tfa = tf.convert_to_tensor(a, dtype=np.float32)
    b = np.log10(a)
    tfb = tf_log10(tfa)
    assert np.allclose(b, tfb.numpy())


@pytest.mark.parametrize('low_mem', [True, False])
def test_idw(low_mem):
    """Test the IDW interpolation fill method"""
    x = np.full([2, 30, 30, 10, 3], np.nan)
    mask = np.random.uniform(0, 1, size=x.shape[0:3]) < 0.99
    const = 2.5
    x_input = tf.convert_to_tensor(x, dtype=tf.float64)

    # all nans results in all zeros
    x_out, _ = idw_fill(x_input, low_mem=low_mem)
    assert np.allclose(x_out.numpy(), 0)

    # all nans with a single value results in all const
    x[:, 15, 15, :, :] = const
    x_input = tf.convert_to_tensor(x, dtype=tf.float64)
    x_out, _ = idw_fill(x_input, low_mem=low_mem)
    assert np.allclose(x_out.numpy(), const)

    x = np.random.uniform(const, size=x.shape)
    x[mask] = np.nan
    x_input = tf.convert_to_tensor(x, dtype=tf.float64)
    x_out, _ = idw_fill(x_input, low_mem=low_mem)
    assert np.allclose(x_out.numpy()[~mask], x[~mask], atol=1e-6)
    assert not np.any(np.isnan(x_out.numpy()[mask]))

    assert np.allclose(
        np.mean(x_out.numpy()), np.mean(x[~mask]), atol=0.1
    )
    assert np.allclose(np.min(x_out.numpy()), np.min(x[~mask]), atol=0.1)
    assert np.allclose(np.max(x_out.numpy()), np.max(x[~mask]), atol=0.1)

    # manual idw check
    # TensorFlow fill
    x_input = tf.convert_to_tensor(x, dtype=tf.float64)
    x_out, _ = idw_fill(x_input, low_mem=low_mem)

    # numpy fill
    B, H, W, D, C = x.shape
    coords = np.array(
        [[i, j] for i in range(H) for j in range(W)], dtype=np.float64
    )
    coords_flat = coords.reshape(H * W, 2)

    x_out_np = np.empty_like(x)
    for b in range(B):
        for d in range(D):
            for c in range(C):
                x_slice = x[b, :, :, d, c]
                x_flat = x_slice.reshape(-1)
                mask_flat = ~np.isnan(x_flat)
                nan_mask_local = ~mask_flat
                if np.all(mask_flat):
                    filled = x_flat
                else:
                    valid_coords = coords_flat[mask_flat]
                    valid_vals = x_flat[mask_flat]
                    nan_coords = coords_flat[nan_mask_local]
                    nan_indices = np.where(nan_mask_local)[0]

                    diffs = nan_coords[:, None, :] - valid_coords[None, :, :]
                    dists = np.linalg.norm(diffs, axis=-1)
                    weights = 1.0 / dists
                    weights /= np.sum(weights, axis=1, keepdims=True)

                    vals = np.sum(weights * valid_vals[None, :], axis=1)
                    filled = x_flat.copy()
                    filled[nan_indices] = vals

                x_out_np[b, :, :, d, c] = filled.reshape(H, W)

    assert np.allclose(x_out.numpy(), x_out_np, atol=1e-3)
