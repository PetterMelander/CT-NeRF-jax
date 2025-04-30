"""Functions for sampling along rays."""

import jax
import jax.numpy as jnp


def uniform_sampling(
    rand_key: jax.Array,
    n_samples: int,
    ray_bounds: jax.Array,
    plateau_ratio: float,  # noqa: ARG001
) -> jax.Array:
    """Sample n_samples points evenly along the ray.

    Args:
        rand_key (jax.Array): random key for JAX's random number generator
        n_samples (int): number of samples
        ray_bounds (jax.Array): Unused, kept for compatibility with vmap
        plateau_ratio (float): Unused, kept for compatibility with vmap

    Returns:
        jax.Array: shape (n_samples,). Contains the sampled t's

    """
    interval_size = 2 / n_samples
    uniform_samples = jnp.arange(0, n_samples, dtype=ray_bounds.dtype)

    # Rescale each row to [t_min, t_max)
    uniform_samples = uniform_samples * interval_size
    perturbation = jax.random.uniform(rand_key, n_samples, dtype=ray_bounds.dtype) * interval_size
    return uniform_samples + perturbation


def cylinder_sampling(
    rand_key: jax.Array,
    n_samples: int,
    ray_bounds: jax.Array,
    plateau_ratio: float,  # noqa: ARG001
) -> jax.Array:
    """Sample n_samples points evenly along the ray inside the central cylinder.

    Args:
        rand_key (jax.Array): random key for JAX's random number generator
        n_samples (int): number of samples
        ray_bounds (jax.Array): shape (2,). Contains the min and max t values for sampling
        plateau_ratio (float): Unused, kept for compatibility with vmap

    Returns:
        jax.Array: shape (n_samples,). Contains the sampled t's

    """
    interval_size = (ray_bounds[1] - ray_bounds[0]) / n_samples
    uniform_samples = jnp.arange(0, n_samples, dtype=ray_bounds.dtype)

    # Rescale each row to [t_min, t_max)
    uniform_samples = uniform_samples * interval_size + ray_bounds[0]
    perturbation = jax.random.uniform(
        rand_key,
        n_samples,
        maxval=interval_size,
        dtype=ray_bounds.dtype,
    )
    return uniform_samples + perturbation


def plateau_sampling(
    rand_key: jax.Array,
    n_samples: int,
    ray_bounds: jax.Array,
    plateau_ratio: float,
) -> jax.Array:
    """Sample n_samples along the ray using a plateau distribution beginning at 0.

    Args:
        rand_key (jax.Array): random key for JAX's random number generator
        n_samples (int): number of samples
        ray_bounds (jax.Array): Unused, kept for compatibility with vmap
        plateau_ratio (float): ratio controlling the width of the plateau distribution

    Returns:
        jax.Array: shape (n_samples,). Contains the sampled t's

    """
    keys = jax.random.split(rand_key, 2)
    x1 = jax.random.uniform(keys[0], n_samples, dtype=ray_bounds.dtype) * plateau_ratio - (
        plateau_ratio / 2
    )
    x2 = jax.random.normal(keys[1], n_samples, dtype=ray_bounds.dtype)
    samples = x1 + x2
    samples = jnp.sort(samples, stable=False)

    # Rescale each row to [0, 2)
    s_min = samples[0]
    s_max = samples[-1]
    return (samples - s_min) / (s_max - s_min) * 2


def plateau_cylinder_sampling(
    rand_key: jax.Array,
    n_samples: int,
    ray_bounds: jax.Array,
    plateau_ratio: float,
) -> jax.Array:
    """Sample n_samples points along the ray using plateau sampling within the cylinder bounds.

    Args:
        rand_key (jax.Array): random key for JAX's random number generator
        n_samples (int): number of samples
        ray_bounds (jax.Array): shape (2,). Contains the min and max t values for sampling
        plateau_ratio (float): ratio controlling the width of the plateau distribution

    Returns:
        jax.Array: shape (n_samples,). Contains the sampled t's

    """
    keys = jax.random.split(rand_key, 2)
    x1 = jax.random.uniform(keys[0], n_samples, dtype=ray_bounds.dtype) * plateau_ratio - (
        plateau_ratio / 2
    )
    x2 = jax.random.normal(keys[1], n_samples, dtype=ray_bounds.dtype)
    samples = x1 + x2
    samples = jnp.sort(samples, stable=False)

    # Rescale each row to [t_min, t_max)
    t_min = ray_bounds[0]
    t_max = ray_bounds[1]
    s_min = samples[0]
    s_max = samples[-1]
    return (samples - s_min) / (s_max - s_min) * (t_max - t_min) + t_min


def fine_sampling(
    rand_key: jax.Array,
    n_samples: int,
    coarse_sample_values: jax.Array,
    coarse_sampling_distances: jax.Array,
) -> jax.Array:
    """Sample n_samples points along the ray using the density based sampling from the NeRF paper.

    Args:
        rand_key (jax.Array): random key for JAX's random number generation.
        n_samples (int): number of samples
        coarse_sample_values (jax.Array): shape (n_samples,). Contains the output values of the
            coarse model.
        coarse_sampling_distances (jax.Array): shape (n_samples,). Contains the sampling
            distances of the coarse sampling
    Returns:
        jax.Array: shape (n_samples,). Contains the sampled t's

    """
    keys = jax.random.split(rand_key, 2)

    # compute a "cdf" of the intensity found by the coarse model, accounting for sampling distances
    pdf = coarse_sample_values * coarse_sampling_distances
    pdf = pdf / jnp.sum(pdf)
    cdf = jnp.cumsum(pdf)
    cdf.at[-1].set(1 + 1e-5)  # to avoid rounding causing index out of bounds if x is close to 1

    # inverse transform sampling
    # each random x will fall between two sampling distances, lower and upper
    x = jax.random.uniform(keys[0], n_samples)
    inds = jnp.searchsorted(cdf, x, side="left")
    cum_sampling_distances = jnp.cumsum(coarse_sampling_distances)
    cum_sampling_distances = jnp.concatenate([jnp.array([0]), cum_sampling_distances])
    lower = cum_sampling_distances[inds]
    upper = cum_sampling_distances[inds + 1]

    # uniformly sample between lower and upper
    t = jax.random.uniform(keys[1], n_samples)
    return lower + t * (upper - lower)


def edge_focused_fine_sampling(
    rand_key: jax.Array,
    n_samples: int,
    coarse_sample_values: jax.Array,
    coarse_sampling_distances: jax.Array,
) -> jax.Array:
    """Edge focused fine sampling.

    Sample n_samples points along the ray using a sampling strategy based on extra sampling
    around edges found by the coarse model.

    Args:
        rand_key (jax.Array): random key for JAX's random number generation.
        n_samples (int): number of samples
        coarse_sample_values (jax.Array): shape (n_samples,). Contains the output values of the
            coarse model.
        coarse_sampling_distances (jax.Array): shape (n_samples,). Contains the sampling
            distances of the coarse model.

    Returns:
        jax.Array: shape (n_samples,). Contains the sampled t's

    """
    keys = jax.random.split(rand_key, 2)

    # compute a "cdf" of the intensity found by the coarse model, accounting for sampling distances
    diff = jnp.abs(jnp.diff(coarse_sample_values, append=0))
    pdf = diff / coarse_sampling_distances
    pdf = pdf / jnp.sum(pdf)
    cdf = jnp.cumsum(pdf)
    cdf.at[-1].set(1 + 1e-5)  # to avoid rounding causing index out of bounds if x is close to 1

    # inverse transform sampling
    # each random x will fall between two sampling distances, lower and upper
    x = jax.random.uniform(keys[0], n_samples)
    inds = jnp.searchsorted(cdf, x, side="left")
    cum_sampling_distances = jnp.cumsum(coarse_sampling_distances)
    cum_sampling_distances = jnp.concatenate([jnp.array([0]), cum_sampling_distances])
    lower = cum_sampling_distances[inds]
    upper = cum_sampling_distances[inds + 1]

    # uniformly sample between lower and upper
    t = jax.random.uniform(keys[1], n_samples)
    return lower + t * (upper - lower)
