"""Test module for the rays module of CT-NeRF."""

import jax
import jax.numpy as jnp
import pytest

from ctnerf import ray_sampling, rays


@pytest.mark.parametrize(
    ("s", "k"),
    [
        (0.5, 0.1),
        (1.0, 0.5),
        (2.0, 1.0),
        (None, None),
    ],
)
@pytest.mark.parametrize("batch_size", [None, 1024])
def test_beer_lambert_law(s: float | None, k: float | None, batch_size: int | None) -> None:
    dsize = 256
    shape = (batch_size, dsize) if batch_size is not None else (dsize,)
    key = jax.random.key(42)
    key, *subkeys = jax.random.split(key, 3)
    distances = jax.random.uniform(subkeys[0], shape=shape, minval=0.0, maxval=1.0)
    attenuation_coeffs = jax.random.uniform(subkeys[1], shape=shape, minval=0.0, maxval=0.1)
    slice_size_cm = 150

    if batch_size is not None:
        vmapped = jax.vmap(rays.beer_lambert_law, in_axes=[0, 0, None, None, None])
        output = jax.jit(vmapped)(attenuation_coeffs, distances, s, k, slice_size_cm)
        assert output.size == batch_size
    else:
        output = jax.jit(rays.beer_lambert_law)(attenuation_coeffs, distances, s, k, slice_size_cm)
        assert output.size == 1

    assert not jnp.any(jnp.isnan(output))
    assert jnp.all(jnp.isfinite(output))
    if k is not None and s is not None:
        assert jnp.max(output >= jnp.log(k) / s)
        assert jnp.min(output <= jnp.log(1 + k) / s)
    else:
        assert jnp.max(output <= 1.0)
        assert jnp.min(output >= 0)


@pytest.mark.parametrize("batch_size", [None, 4096])
def test_get_rays(batch_size: int | None) -> None:
    key = jax.random.key(42)
    key, *subkeys = jax.random.split(key, 3)

    shape = (batch_size, 2) if batch_size is not None else (2,)
    pixel_pos = jax.random.uniform(subkeys[0], shape=shape, minval=0.0, maxval=511.0)
    img_shape = jnp.ones(shape) * 512
    shape = (batch_size, 1) if batch_size is not None else (1,)
    angle = jax.random.uniform(subkeys[1], shape=shape, minval=0.0, maxval=jnp.pi)

    if batch_size is not None:
        vmapped = jax.vmap(rays.get_rays)
        start_pos, heading_vector, ray_bounds = jax.jit(vmapped)(pixel_pos, angle, img_shape)
        assert start_pos.shape == (batch_size, 3)
        assert heading_vector.shape == (batch_size, 3)
        assert ray_bounds.shape == (batch_size, 2)
        assert jnp.all(jnp.isclose(jnp.linalg.norm(heading_vector, axis=1), 1.0))
        assert jnp.all(heading_vector[:, 2] == 0)
        assert jnp.all(jnp.linalg.norm(start_pos[:, :2], axis=1) >= 1)
        assert jnp.all(jnp.linalg.norm(start_pos[:, :2], axis=1) <= jnp.sqrt(2))
    else:
        start_pos, heading_vector, ray_bounds = jax.jit(rays.get_rays)(pixel_pos, angle, img_shape)
        assert start_pos.shape == (3,)
        assert heading_vector.shape == (3,)
        assert ray_bounds.shape == (2,)
        assert jnp.isclose(jnp.linalg.norm(heading_vector), 1.0)
        assert heading_vector[2] == 0
        assert jnp.linalg.norm(start_pos[:2]) >= 1
        assert jnp.linalg.norm(start_pos[:2]) <= jnp.sqrt(2)

    assert not jnp.any(jnp.isnan(start_pos))
    assert jnp.all(jnp.isfinite(start_pos))
    assert not jnp.any(jnp.isnan(heading_vector))
    assert jnp.all(jnp.isfinite(heading_vector))
    assert not jnp.any(jnp.isnan(ray_bounds))
    assert jnp.all(jnp.isfinite(ray_bounds))
    assert jnp.min(ray_bounds) >= 0
    assert jnp.max(ray_bounds) <= 2


@pytest.mark.parametrize("batch_size", [None, 128])
def test_z_rotation_matrix(batch_size: int | None) -> None:
    key = jax.random.key(42)
    shape = (batch_size, 1) if batch_size is not None else 1
    angle = jax.random.uniform(key, shape=shape, minval=0.0, maxval=jnp.pi)
    if batch_size is not None:
        vmapped = jax.vmap(rays._create_z_rotation_matrix)
        rotation_matrix, heading_vector = jax.jit(vmapped)(angle)
        assert rotation_matrix.shape == (batch_size, 3, 3)
        identity = jnp.eye(3)
        assert jnp.allclose(
            rotation_matrix @ jnp.transpose(rotation_matrix, (0, 2, 1)),
            jnp.broadcast_to(identity, (batch_size, 3, 3)),
            atol=1e-5,
        )
        assert jnp.allclose(jnp.linalg.det(rotation_matrix), jnp.ones(batch_size), atol=1e-5)
        assert jnp.allclose(
            rotation_matrix[:, 2, :],
            jnp.broadcast_to(jnp.array([0, 0, 1]), (batch_size, 3)),
            atol=1e-5,
        )
        assert jnp.allclose(
            rotation_matrix[:, :, 2],
            jnp.broadcast_to(jnp.array([0, 0, 1]), (batch_size, 3)),
            atol=1e-5,
        )
    else:
        rotation_matrix, heading_vector = jax.jit(rays._create_z_rotation_matrix)(angle)
        assert rotation_matrix.shape == (3, 3)
        identity = jnp.eye(3)
        assert jnp.allclose(rotation_matrix @ rotation_matrix.T, identity, atol=1e-5)
        assert jnp.isclose(jnp.linalg.det(rotation_matrix), 1.0, atol=1e-5)
        assert jnp.allclose(rotation_matrix[2, :], jnp.array([0, 0, 1]), atol=1e-5)
        assert jnp.allclose(rotation_matrix[:, 2], jnp.array([0, 0, 1]), atol=1e-5)
    assert not jnp.any(jnp.isnan(rotation_matrix))
    assert jnp.all(jnp.isfinite(rotation_matrix))
    assert not jnp.any(jnp.isnan(heading_vector))
    assert jnp.all(jnp.isfinite(heading_vector))


@pytest.mark.parametrize("batch_size", [None, 128])
@pytest.mark.parametrize("ray_limit", [True, False])
def test_sampling_distances(batch_size: int | None, ray_limit: bool) -> None:  # noqa: FBT001
    dsize = 256
    key = jax.random.key(42)
    key, *subkeys = jax.random.split(key, 3)

    shape = (batch_size, dsize) if batch_size is not None else (dsize,)
    t_samples = jax.random.uniform(subkeys[0], shape=shape, minval=0.0, maxval=2)
    t_samples = jnp.sort(t_samples)
    shape = (batch_size, 2) if batch_size is not None else (1,)
    ray_bounds = jnp.ones(shape) * 2 if ray_limit else None

    if batch_size is not None:
        vmapped = jax.jit(jax.vmap(rays.get_sampling_distances))
        diff = vmapped(t_samples, ray_bounds)
        assert diff.shape == (batch_size, dsize)
        assert diff.max() <= 2
        assert diff.min() >= 0
        assert jnp.all(jnp.sum(diff, axis=1) <= 2)
        assert jnp.all(jnp.sum(diff, axis=1) >= 0)
    else:
        diff = jax.jit(rays.get_sampling_distances)(t_samples, ray_bounds)
        assert diff.shape == (dsize,)
        assert diff.max() <= 2
        assert diff.min() >= 0
        assert jnp.sum(diff) <= 2
        assert jnp.sum(diff) >= 0
    assert not jnp.any(jnp.isnan(diff))
    assert jnp.all(jnp.isfinite(diff))


@pytest.mark.parametrize(
    "sampling_function",
    ["uniform_sampling", "cylinder_sampling", "plateau_sampling", "plateau_cylinder_sampling"],
)
@pytest.mark.parametrize("batch_size", [None, 128])
def test_coarse_sampling(batch_size: int, sampling_function: str) -> None:  # noqa: PLR0915
    key = jax.random.key(42)
    n_samples = 256
    plateau_ratio = 10
    eps = 0.00001 if "plateau" in sampling_function else 0
    sampling_function = getattr(ray_sampling, sampling_function)

    if batch_size is not None:
        start_pos = jnp.array([[0, -1, 0]] * batch_size)
        heading_vector = jnp.array([[0, 1, 0]] * batch_size)
        ray_bounds = jnp.array([[0, 2]] * batch_size)
        keys = jax.random.split(key, batch_size)

        vmapped = jax.jit(
            jax.vmap(rays.get_coarse_samples, in_axes=(0, 0, 0, None, 0, None, None)),
            static_argnums=(3, 6),
        )
        t_samples, sampled_points, sampling_distances = vmapped(
            keys,
            start_pos,
            heading_vector,
            n_samples,
            ray_bounds,
            plateau_ratio,
            sampling_function,
        )

        assert t_samples.shape == (batch_size, n_samples)
        assert jnp.max(t_samples) <= 2 + eps
        assert jnp.min(t_samples) >= 0
        assert jnp.all(jnp.diff(t_samples, axis=1, append=2 + eps) >= 0)
        assert jnp.all(jnp.diff(t_samples, axis=1, append=2) <= 2)

        assert sampled_points.shape == (batch_size, n_samples, 3)
        assert jnp.all(jnp.max(sampled_points) <= 1 + eps)
        assert jnp.all(jnp.min(sampled_points) >= -1 - eps)
        assert jnp.all(jnp.linalg.vector_norm(sampled_points[:, :, :2], axis=2) <= jnp.sqrt(2))
        z_min = jnp.min(sampled_points[:, :, 2], axis=1)
        z_max = jnp.max(sampled_points[:, :, 2], axis=1)
        assert jnp.all(z_min == z_max)

        assert sampling_distances.shape == (batch_size, n_samples)
        assert sampling_distances.max() <= 2
        assert sampling_distances.min() >= 0 - eps
        assert jnp.all(jnp.sum(sampling_distances, axis=1) <= 2 + eps)
        assert jnp.all(jnp.sum(sampling_distances, axis=1) >= 0)

    else:
        start_pos = jnp.array([0, -1, 0])
        heading_vector = jnp.array([0, 1, 0])
        ray_bounds = jnp.array([0, 2])

        func = jax.jit(rays.get_coarse_samples, static_argnums=(3, 6))
        t_samples, sampled_points, sampling_distances = func(
            key,
            start_pos,
            heading_vector,
            n_samples,
            ray_bounds,
            plateau_ratio,
            sampling_function,
        )

        assert t_samples.shape == (n_samples,)
        assert jnp.max(t_samples) <= 2 + eps
        assert jnp.min(t_samples) >= 0
        assert jnp.all(jnp.diff(t_samples, append=2 + eps) >= 0)
        assert jnp.all(jnp.diff(t_samples, append=2) <= 2)

        assert sampled_points.shape == (n_samples, 3)
        assert jnp.max(sampled_points) <= 1 + eps
        assert jnp.min(sampled_points) >= -1 - eps
        assert jnp.all(jnp.linalg.vector_norm(sampled_points[:, :2], axis=1) <= jnp.sqrt(2))
        z_min = jnp.min(sampled_points[:, 2], axis=0)
        z_max = jnp.max(sampled_points[:, 2], axis=0)
        assert z_min == z_max

        assert sampling_distances.shape == (n_samples,)
        assert sampling_distances.max() <= 2
        assert sampling_distances.min() >= 0 - eps
        assert jnp.sum(sampling_distances, axis=0) <= 2 + eps
        assert jnp.sum(sampling_distances, axis=0) >= 0
    assert not jnp.any(jnp.isnan(t_samples))
    assert jnp.all(jnp.isfinite(t_samples))
    assert not jnp.any(jnp.isnan(sampled_points))
    assert jnp.all(jnp.isfinite(sampled_points))
    assert not jnp.any(jnp.isnan(sampling_distances))
    assert jnp.all(jnp.isfinite(sampling_distances))


@pytest.mark.parametrize("batch_size", [None, 128])
def test_fine_sampling(batch_size: int) -> None:  # noqa: PLR0915
    key = jax.random.key(42)
    n_samples = 256
    eps = 0.00001

    if batch_size is not None:
        key1, key2, key3 = jax.random.split(key, 3)
        start_pos = jnp.array([[0, -1, 0]] * batch_size)
        heading_vector = jnp.array([[0, 1, 0]] * batch_size)
        ray_bounds = jnp.array([[0, 2]] * batch_size)

        keys = jax.random.split(key3, batch_size)
        coarse_sample_ts = jax.random.uniform(key1, (batch_size, n_samples), maxval=2).sort(axis=1)
        coarse_sample_values = jax.random.uniform(key2, (batch_size, n_samples), maxval=0.1)
        coarse_sampling_distances = jnp.diff(coarse_sample_ts, axis=1, append=2)

        vmapped = jax.jit(
            jax.vmap(rays.get_fine_samples, in_axes=(0, 0, 0, 0, 0, 0, 0, None)),
            static_argnums=(7,),
        )
        sampled_points, sampling_distances = vmapped(
            keys,
            start_pos,
            heading_vector,
            ray_bounds,
            coarse_sample_ts,
            coarse_sample_values,
            coarse_sampling_distances,
            n_samples,
        )

        assert sampled_points.shape == (batch_size, n_samples * 2, 3)
        assert jnp.all(jnp.max(sampled_points) <= 1 + eps)
        assert jnp.all(jnp.min(sampled_points) >= -1 - eps)
        assert jnp.all(jnp.linalg.vector_norm(sampled_points[:, :, :2], axis=2) <= jnp.sqrt(2))
        z_min = jnp.min(sampled_points[:, :, 2], axis=1)
        z_max = jnp.max(sampled_points[:, :, 2], axis=1)
        assert jnp.all(z_min == z_max)

        assert sampling_distances.shape == (batch_size, n_samples * 2)
        assert sampling_distances.max() <= 2
        assert sampling_distances.min() >= 0 - eps
        assert jnp.all(jnp.sum(sampling_distances, axis=1) <= 2 + eps)
        assert jnp.all(jnp.sum(sampling_distances, axis=1) >= 0)

    else:
        start_pos = jnp.array([0, -1, 0])
        heading_vector = jnp.array([0, 1, 0])
        ray_bounds = jnp.array([0, 2])

        key1, key2, key3 = jax.random.split(key, 3)
        coarse_sample_ts = jax.random.uniform(key1, n_samples, maxval=2).sort()
        coarse_sample_values = jax.random.uniform(key2, n_samples, maxval=0.1)
        coarse_sampling_distances = jnp.diff(coarse_sample_ts, append=2)

        func = jax.jit(
            rays.get_fine_samples,
            static_argnums=(7,),
        )
        sampled_points, sampling_distances = func(
            key3,
            start_pos,
            heading_vector,
            ray_bounds,
            coarse_sample_ts,
            coarse_sample_values,
            coarse_sampling_distances,
            n_samples,
        )

        assert sampled_points.shape == (n_samples * 2, 3)
        assert jnp.max(sampled_points) <= 1 + eps
        assert jnp.min(sampled_points) >= -1 - eps
        assert jnp.all(jnp.linalg.vector_norm(sampled_points[:, :2], axis=1) <= jnp.sqrt(2))
        z_min = jnp.min(sampled_points[:, 2])
        z_max = jnp.max(sampled_points[:, 2])
        assert jnp.all(z_min == z_max)

        assert sampling_distances.shape == (n_samples * 2,)
        assert sampling_distances.max() <= 2
        assert sampling_distances.min() >= 0 - eps
        assert jnp.sum(sampling_distances) <= 2 + eps
        assert jnp.sum(sampling_distances) >= 0
    assert not jnp.any(jnp.isnan(sampled_points))
    assert jnp.all(jnp.isfinite(sampled_points))
    assert not jnp.any(jnp.isnan(sampling_distances))
    assert jnp.all(jnp.isfinite(sampling_distances))
