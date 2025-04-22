"""Unit tests for ray sampling methods in the ctnerf package."""

import jax
import jax.numpy as jnp
import pytest

from ctnerf import ray_sampling


@pytest.fixture
def n_samples() -> int:
    return 256


@pytest.fixture
def key() -> jax.Array:
    return jax.random.key(42)


@pytest.mark.parametrize("batch_size", [None, 4096])
def test_uniform_sampling(key: jax.Array, n_samples: int, batch_size: int | None) -> None:
    if batch_size is not None:
        keys = jax.random.split(key, batch_size)
        vmapped = jax.jit(
            jax.vmap(ray_sampling.uniform_sampling, in_axes=(0, None, 0, None)),
            static_argnums=(1,),
        )
        samples = vmapped(keys, n_samples, None, None)
        assert samples.shape == (batch_size, n_samples)
        assert jnp.max(samples) <= 2
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, axis=1, append=2) >= 0)
        assert jnp.all(jnp.diff(samples, axis=1, append=2) <= 2 * 2 / n_samples)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))
    else:
        samples = jax.jit(ray_sampling.uniform_sampling, static_argnums=(1,))(
            key,
            n_samples,
            None,
            None,
        )
        assert samples.shape == (n_samples,)
        assert jnp.max(samples) <= 2
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, append=2) >= 0)
        assert jnp.all(jnp.diff(samples, append=2) <= 2 * 2 / n_samples)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))


@pytest.mark.parametrize("batch_size", [None, 4096])
def test_cylinder_sampling(key: jax.Array, n_samples: int, batch_size: int | None) -> None:
    if batch_size is not None:
        ray_bounds = jnp.array([[0, 2]] * batch_size)
        keys = jax.random.split(key, batch_size)
        vmapped = jax.jit(
            jax.vmap(ray_sampling.cylinder_sampling, in_axes=(0, None, 0, None)),
            static_argnums=(1,),
        )
        samples = vmapped(keys, n_samples, ray_bounds, None)
        assert samples.shape == (batch_size, n_samples)
        assert jnp.max(samples) <= 2
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, axis=1, append=2) >= 0)
        assert jnp.all(jnp.diff(samples, axis=1, append=2) <= 2 * 2 / n_samples)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))
    else:
        ray_bounds = jnp.array([0, 2])
        samples = jax.jit(ray_sampling.cylinder_sampling, static_argnums=(1,))(
            key,
            n_samples,
            ray_bounds,
            None,
        )
        assert samples.shape == (n_samples,)
        assert jnp.max(samples) <= 2
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, append=2) >= 0)
        assert jnp.all(jnp.diff(samples, append=2) <= 2 * 2 / n_samples)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))


@pytest.mark.parametrize("batch_size", [None, 4096])
def test_plateau_sampling(key: jax.Array, n_samples: int, batch_size: int | None) -> None:
    plateau_ratio = 10
    eps = 0.00001
    if batch_size is not None:
        keys = jax.random.split(key, batch_size)
        vmapped = jax.jit(
            jax.vmap(ray_sampling.plateau_sampling, in_axes=(0, None, 0, None)),
            static_argnums=(1,),
        )
        samples = vmapped(keys, n_samples, None, plateau_ratio)
        assert samples.shape == (batch_size, n_samples)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, axis=1, append=2 + eps) >= 0)
        assert jnp.all(jnp.diff(samples, axis=1, append=2) <= 2)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))
    else:
        samples = jax.jit(ray_sampling.plateau_sampling, static_argnums=(1,))(
            key,
            n_samples,
            None,
            plateau_ratio,
        )
        assert samples.shape == (n_samples,)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, append=2 + eps) >= 0)
        assert jnp.all(jnp.diff(samples, append=2) <= 2)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))


@pytest.mark.parametrize("batch_size", [None, 4096])
def test_plateau_cylinder_sampling(key: jax.Array, n_samples: int, batch_size: int | None) -> None:
    plateau_ratio = 10
    eps = 0.00001
    if batch_size is not None:
        ray_bounds = jnp.array([[0, 2]] * batch_size)
        keys = jax.random.split(key, batch_size)
        vmapped = jax.jit(
            jax.vmap(ray_sampling.plateau_cylinder_sampling, in_axes=(0, None, 0, None)),
            static_argnums=(1,),
        )
        samples = vmapped(keys, n_samples, ray_bounds, plateau_ratio)
        assert samples.shape == (batch_size, n_samples)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, axis=1, append=2 + eps) >= 0)
        assert jnp.all(jnp.diff(samples, axis=1, append=2) <= 2)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))
    else:
        ray_bounds = jnp.array([0, 2])
        samples = jax.jit(ray_sampling.plateau_cylinder_sampling, static_argnums=(1,))(
            key,
            n_samples,
            ray_bounds,
            plateau_ratio,
        )
        assert samples.shape == (n_samples,)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert jnp.all(jnp.diff(samples, append=2 + eps) >= 0)
        assert jnp.all(jnp.diff(samples, append=2) <= 2)
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))


@pytest.mark.parametrize("batch_size", [None, 4096])
def test_fine_sampling(key: jax.Array, n_samples: int, batch_size: int | None) -> None:
    eps = 0.00001
    n_fine_samples = 512
    if batch_size is not None:
        key1, key2, key3 = jax.random.split(key, 3)
        coarse_samples = jax.random.uniform(key1, (batch_size, n_samples), maxval=0.1)
        coarse_sampling_distances = jax.random.uniform(
            key2,
            (batch_size, n_samples),
            maxval=2 / n_samples,
        )
        keys = jax.random.split(key3, batch_size)
        vmapped = jax.jit(
            jax.vmap(ray_sampling.fine_sampling, in_axes=(0, None, 0, 0)),
            static_argnums=(1,),
        )
        samples = vmapped(keys, n_fine_samples, coarse_samples, coarse_sampling_distances)
        assert samples.shape == (batch_size, n_fine_samples)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))
    else:
        key, key2, key3 = jax.random.split(key, 3)
        coarse_samples = jax.random.uniform(key, n_samples, maxval=0.1)
        coarse_sampling_distances = jax.random.uniform(key2, n_samples, maxval=2 / n_samples)
        samples = jax.jit(ray_sampling.fine_sampling, static_argnums=(1,))(
            key3,
            n_fine_samples,
            coarse_samples,
            coarse_sampling_distances,
        )
        assert samples.shape == (n_fine_samples,)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))


@pytest.mark.parametrize("batch_size", [None, 4096])
def test_edge_fine_sampling(key: jax.Array, n_samples: int, batch_size: int | None) -> None:
    eps = 0.00001
    n_fine_samples = 512
    if batch_size is not None:
        key1, key2 = jax.random.split(key, 2)
        keys = jax.random.split(key2, batch_size)
        coarse_samples = jax.random.uniform(key1, (batch_size, n_samples), maxval=2).sort(axis=1)
        coarse_sampling_distances = jnp.diff(coarse_samples, append=2, axis=1)
        vmapped = jax.jit(
            jax.vmap(ray_sampling.edge_focused_fine_sampling, in_axes=(0, None, 0, 0)),
            static_argnums=(1,),
        )
        samples = vmapped(keys, n_fine_samples, coarse_samples, coarse_sampling_distances)
        assert samples.shape == (batch_size, n_fine_samples)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))
    else:
        key, key2 = jax.random.split(key, 2)
        coarse_samples = jax.random.uniform(key, n_samples, maxval=2).sort()
        coarse_sampling_distances = jnp.diff(coarse_samples, append=2)
        samples = jax.jit(ray_sampling.edge_focused_fine_sampling, static_argnums=(1,))(
            key2,
            n_fine_samples,
            coarse_samples,
            coarse_sampling_distances,
        )
        assert samples.shape == (n_fine_samples,)
        assert jnp.max(samples) <= 2 + eps
        assert jnp.min(samples) >= 0
        assert not jnp.any(jnp.isnan(samples))
        assert jnp.all(jnp.isfinite(samples))
