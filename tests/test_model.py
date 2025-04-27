"""Test module for the ctnerf model implementation.

This module contains unit tests for verifying the functionality
of the model components in the ctnerf package.
"""

import jax
import jax.numpy as jnp
import pytest

from ctnerf import model


@pytest.mark.parametrize("batch_size", [None, 128])
def test_pos_enc(batch_size: int | None) -> None:
    L = 20  # noqa: N806
    if batch_size is not None:
        vmapped = jax.jit(
            jax.vmap(model._positional_encoding, in_axes=(0, None)),
            static_argnums=(1,),
        )
        pos_enc = vmapped(jnp.array([[1, 0, -1]] * batch_size), L)
        assert pos_enc.shape == (batch_size, 3 * 2 * L)
        assert jnp.max(jnp.abs(pos_enc)) <= jnp.pi
        assert jnp.all(pos_enc[:, 3 * L :] ** 2 + pos_enc[:, : 3 * L] ** 2 == 1)  # trig 1
        assert not jnp.any(jnp.isnan(pos_enc))
        assert jnp.all(jnp.isfinite(pos_enc))
    else:
        pos_enc = jax.jit(model._positional_encoding, static_argnums=(1,))(jnp.array([1, 0, -1]), L)
        assert pos_enc.shape == (3 * 2 * L,)
        assert jnp.max(jnp.abs(pos_enc)) <= jnp.pi
        assert jnp.all(pos_enc[3 * L :] ** 2 + pos_enc[: 3 * L] ** 2 == 1)  # trig 1
        assert not jnp.any(jnp.isnan(pos_enc))
        assert jnp.all(jnp.isfinite(pos_enc))


@pytest.mark.parametrize("dims", [(1, 128), (128, 1)])
def test_linear_init(dims: tuple[int, int]) -> None:
    keys = jax.random.split(jax.random.key(42), 2)
    layer = model._init__linear_layer(keys, *dims)
    w, b = layer["w"], layer["b"]
    assert w.shape == (dims[1], dims[0])
    assert b.shape == (dims[1],)
    assert jnp.max(jnp.abs(w)) <= jnp.sqrt(1 / dims[0])
    assert jnp.max(jnp.abs(b)) <= jnp.sqrt(1 / dims[0])
    assert not jnp.any(jnp.isnan(w))
    assert jnp.all(jnp.isfinite(w))
    assert not jnp.any(jnp.isnan(b))
    assert jnp.all(jnp.isfinite(b))


@pytest.mark.parametrize("n_layers", [5, 10])
@pytest.mark.parametrize("layer_dim", [128, 256])
@pytest.mark.parametrize("L", [10, 20])
def test_model_init(n_layers: int, layer_dim: int, L: int) -> None:  # noqa: N803
    key = jax.random.key(42)
    pre_concat, post_concat = model.init_params(key, n_layers, layer_dim, L)
    for layer in pre_concat + post_concat:
        w, b = layer["w"], layer["b"]
        assert isinstance(w, jax.Array)
        assert isinstance(b, jax.Array)
        assert jnp.max(jnp.abs(w)) <= 1
        assert jnp.max(jnp.abs(b)) <= 1
        assert not jnp.any(jnp.isnan(w))
        assert jnp.all(jnp.isfinite(w))
        assert not jnp.any(jnp.isnan(b))
        assert jnp.all(jnp.isfinite(b))
    if n_layers % 2 == 0:
        assert len(pre_concat) + len(post_concat) == n_layers + 2
    else:
        assert len(pre_concat) + len(post_concat) == n_layers + 1


@pytest.mark.parametrize("batch_size", [None, 128])
def test_forward(batch_size: int | None) -> None:
    sample_size = 256
    params = model.init_params(jax.random.key(42), 8, 128, 20)

    if batch_size is not None:
        coords = jax.random.uniform(jax.random.key(43), (batch_size, sample_size, 3))
        output = jax.jit(jax.vmap(model.forward, in_axes=(None, 0)))(params, coords)
        assert isinstance(output, jax.Array)
        assert output.shape == (batch_size, sample_size)
        assert not jnp.any(jnp.isnan(output))
        assert jnp.all(jnp.isfinite(output))
    else:
        coords = jax.random.uniform(jax.random.key(43), (sample_size, 3))
        output = jax.jit(model.forward)(params, coords)
        assert isinstance(output, jax.Array)
        assert output.shape == (sample_size,)
        assert not jnp.any(jnp.isnan(output))
        assert jnp.all(jnp.isfinite(output))


@pytest.mark.parametrize("batch_size", [None, 128])
@pytest.mark.parametrize("scaling", [(0.1, 1), (None, None)])
def test_loss(batch_size: int | None, scaling: tuple) -> None:
    sample_size = 256
    params = model.init_params(jax.random.key(42), 8, 128, 20)
    k, s = scaling
    slice_size_cm = 100

    if batch_size is not None:
        coords = jax.random.uniform(jax.random.key(43), (batch_size, sample_size, 3))
        gt = jax.random.uniform(jax.random.key(44), (batch_size,))
        sampling_distances = jax.random.uniform(jax.random.key(45), (batch_size, sample_size))
        output = jax.jit(jax.vmap(model.loss_fn, in_axes=(None, 0, 0, 0, None, None, None)))(
            params,
            coords,
            gt,
            sampling_distances,
            s,
            k,
            slice_size_cm,
        )
        assert isinstance(output, jax.Array)
        assert output.shape == (batch_size,)
        assert not jnp.any(jnp.isnan(output))
        assert jnp.all(jnp.isfinite(output))
    else:
        coords = jax.random.uniform(jax.random.key(43), (sample_size, 3))
        gt = jax.random.uniform(jax.random.key(44), (1,))
        sampling_distances = jax.random.uniform(jax.random.key(45), (sample_size,))
        output = jax.jit(model.loss_fn)(params, coords, gt, sampling_distances, s, k, slice_size_cm)
        assert isinstance(output, jax.Array)
        assert output.shape == (1,)
        assert not jnp.any(jnp.isnan(output))
        assert jnp.all(jnp.isfinite(output))
