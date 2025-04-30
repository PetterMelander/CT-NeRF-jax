"""Contains various functions for computing and using rays."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from ctnerf import ray_sampling


def beer_lambert_law(
    attenuation_coeffs: jax.Array,
    distances: jax.Array,
    s: float | None,
    k: float | None,
    slice_size_cm: float,
) -> jax.Array:
    """Use Beer-Lambert law to calculate transmittance of a ray.

    Uses Beer-Lambert law to calculate transmittance given the attenuation coefficients and
    distances of sampled points along ray. If s and k are provided, uses a version of the law that
    has been adjusted for scaling of the transmittance by adding k, taking the log, and dividing by
    s.

    Args:
        attenuation_coeffs (jax.Array): shape (N,)
        distances (jax.Array): shape (N,)
        s (float | None): scaling factor
        k (float | None): offset
        slice_size_cm (float): size of an axial slice in centimetres

    Returns:
        jax.Array: shape (,). Transmittance

    """
    # scale to cm because the CT creation scripts uses attenuation per cm
    distances = distances * slice_size_cm / 2

    exp = jnp.exp(-jnp.sum(attenuation_coeffs * distances))
    if s is not None and k is not None:
        return jnp.log(exp + k) / s
    return exp


@partial(jax.vmap, in_axes=(0, 0, None))
def get_rays(
    pixel_pos: jax.Array,
    angle: jax.Array,
    img_shape: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Get the start position, heading vector, and ray bounds defining a ray.

    Given the positions of a pixel in an image and the angle of the xray source, compute the start
    position and heading vector for the ray.

    Args:
        pixel_pos (jax.Array): shape (2,). Contains the integer x and y coordinates of the pixel
        angle (jax.Array): shape (,). Contains the rotation angle in radians
        img_shape (jax.Array): shape (2,). Size of the image in pixels

    Returns:
        jax.Array: shape (3). Start position
        jax.Array: shape (3). Heading vector
        jax.Array: shape (2). Ray bounds

    """
    # get normalized position before accounting for angle
    normalized_pos = 2 * pixel_pos / (img_shape - 1) - 1

    # start position is always at x = 1
    x = jnp.array([1])
    normalized_pos = jnp.concatenate([x, normalized_pos])

    # rotate to account for angle
    rotation_matrix, heading_vector = _create_z_rotation_matrix(angle)
    start_pos = jnp.matmul(rotation_matrix, normalized_pos)
    ray_bounds = _get_ray_bounds(start_pos, heading_vector)

    return start_pos, heading_vector, ray_bounds


def _get_ray_bounds(start_pos: jax.Array, heading_vector: jax.Array) -> jax.Array:
    """Get the two t values that define the bounds of the ray.

    Given a start position (a, b, c) and a direction vector (v_x, v_y, 0), a ray can be
    parameterized as (a + t*v_x, b + t*v_y, c). This function returns the t's that correspond to the
    ray passing through the cylinder x^2 + y^2 = 1, which defines the bounds of the image.

    Args:
        start_pos (jax.Array): shape (3,)
        heading_vector (jax.Array): shape (3,)

    Returns:
        jax.Array: An array of shape (2,), containing the two t's

    """
    a = start_pos[0]
    b = start_pos[1]
    v_x = heading_vector[0]
    v_y = heading_vector[1]
    # p_half = (a*v_x + b*v_y) / (v_x**2 + v_y**2)  # noqa: ERA001
    # q = (a**2 + b**2 - 1) / (v_x**2 + v_y**2)  # noqa: ERA001
    p_half = a * v_x + b * v_y
    q = a**2 + b**2 - 1
    sq_root = jnp.sqrt(p_half**2 - q)
    sq_root = jnp.nan_to_num(sq_root, nan=0)  # rounding errors can cause imaginary sq_root
    t1 = -p_half - sq_root
    t2 = -p_half + sq_root
    return jnp.array([t1, t2])


def get_coarse_samples(
    rand_key: jax.Array,
    start_pos: jax.Array,
    heading_vector: jax.Array,
    ray_bounds: jax.Array | None,
    n_samples: int,
    plateau_ratio: float | None,
    sampling_function: Callable[[jax.Array, int, jax.Array, float], jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Get the coarse samples along the ray.

    Samples n_samples points along the ray using plateau sampling within the cylinder bounds.

    Args:
        rand_key (jax.Array): random key for JAX's random number generation.
        start_pos (jax.Array): shape (3,). Starting position of the ray.
        heading_vector (jax.Array): shape (3,). Heading vector of the ray.
        ray_bounds (jax.Array): shape (2,). Ray bounds.
        n_samples (int): number of samples.
        plateau_ratio (float): ratio of plateau width to standard deviation.
        sampling_function (Callable[[int, int, dict], jax.Array]): function that
            performs the sampling.

    Returns:
        jax.Array: shape (n_samples,). t values of the sampled points.
        jax.Array: shape (n_samples, 3). Sampled points.
        jax.Array: shape (n_samples,). Distances between adjacent samples.

    """
    t_samples = sampling_function(
        rand_key,
        n_samples,
        ray_bounds,
        plateau_ratio,
    )
    sampling_distances = get_sampling_distances(t_samples, ray_bounds)
    sampled_points = start_pos + jnp.expand_dims(t_samples, 1) * jnp.expand_dims(heading_vector, 0)

    return t_samples, sampled_points, sampling_distances


def get_fine_samples(
    rand_key: jax.Array,
    start_pos: jax.Array,
    heading_vector: jax.Array,
    ray_bounds: jax.Array,
    coarse_sample_ts: jax.Array,
    coarse_sample_values: jax.Array,
    coarse_sampling_distances: jax.Array,
    n_samples: int,
) -> tuple[jax.Array, jax.Array]:
    """Get the fine samples along the ray.

    Samples n_samples points along the ray using edge focused sampling based on the outputs of the
    coarse model.

    Args:
        rand_key (jax.Array): random key for JAX's random number generation.
        start_pos (jax.Array): shape (3,). Starting position of the ray.
        heading_vector (jax.Array): shape (3,). Heading vector of the ray.
        ray_bounds (jax.Array): shape (2,). Ray bounds.
        coarse_sample_ts (jax.Array): shape (n_samples,). Contains the t values for the coarse
            samples.
        coarse_sample_values (jax.Array): shape (n_samples,). Contains the coarse model's
            outputs for the coarse samples.
        coarse_sampling_distances (jax.Array): shape (n_samples,). Contains the sampling
            distances of the coarse samples.
        n_samples (int): number of samples
    Returns:
        jax.Array: shape (n_samples,). Contains the sampled t's
        jax.Array: shape (n_samples, 3). Contains the distances between adjacent samples.

    """
    t_samples = ray_sampling.edge_focused_fine_sampling(
        rand_key,
        n_samples,
        coarse_sample_values,
        coarse_sampling_distances,
    )

    # concatenate the coarse samples with the fine samples and
    # sort them so distance between adjacent samples can be calculated
    t_samples = jnp.concatenate([coarse_sample_ts, t_samples])
    t_samples = jnp.sort(t_samples, stable=False)
    sampling_distances = get_sampling_distances(t_samples, ray_bounds)

    # sampled points should have shape (B, n_samples, 3)
    sampled_points = start_pos + jnp.expand_dims(t_samples, 1) * jnp.expand_dims(heading_vector, 0)
    return sampled_points, sampling_distances


def _create_z_rotation_matrix(angle: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Get rotation matrix for z-axis rotation and heading vector.

    Create a 3D rotation matricex for rotation around the z-axis.
    Also returns the heading vector.

    Args:
        angle (jax.Array): shape (,). Contains the rotation angle in radians.

    Returns:
        jax.Array: An array of shape (3, 3) containing the rotation matrix.
        jax.Array: An array of shape (3,) containing the heading vector.

    """
    cos_angle = jnp.cos(angle).squeeze()
    sin_angle = jnp.sin(angle).squeeze()

    rotation_matrix = jnp.zeros([3, 3])
    rotation_matrix = rotation_matrix.at[0, 0].set(cos_angle)
    rotation_matrix = rotation_matrix.at[0, 1].set(-sin_angle)
    rotation_matrix = rotation_matrix.at[1, 0].set(sin_angle)
    rotation_matrix = rotation_matrix.at[1, 1].set(cos_angle)
    rotation_matrix = rotation_matrix.at[2, 2].set(1.0)

    # heading vector should be in the negative x direction since the start position is at x = 1
    heading_vector = jnp.zeros(3)
    heading_vector = heading_vector.at[0].set(-cos_angle)
    heading_vector = heading_vector.at[1].set(-sin_angle)

    return rotation_matrix, heading_vector


def get_sampling_distances(
    t_samples: jax.Array,
    ray_bounds: jax.Array | None,
) -> jax.Array:
    """Get distances between adjacent sampled points along the ray.

    Gets the distance between adjacent sampled points along the ray. The distance associated with
    each sample is the distance between that sample and the next sample. The distance for the last
    sample is the distance between the last sample and the far bound of the image.

    This distance can be calculated directly from t values since the heading vector has magnitue 1.

    Args:
        t_samples (jax.Array): shape (n_samples,). Contains the t values for the samples.
        ray_bounds (jax.Array): shape (2,). Ray bounds.

    Returns:
        jax.Array: shape (n_samples,). Distances between adjacent samples.

    """
    # Append upper ray bound to end of each batch so last sample will have a distance.
    ray_limit = ray_bounds[1] if ray_bounds is not None else 2
    return jnp.diff(t_samples, append=ray_limit)
