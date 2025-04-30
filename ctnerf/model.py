"""Contains the MLP model used to generate CT images."""

from functools import partial

import jax
import jax.numpy as jnp
import jmp
from jax.nn.initializers import truncated_normal

from ctnerf.rays import beer_lambert_law


def init_params(
    key: jax.Array,
    n_layers: int,
    layer_dim: int,
    L: int,  # noqa: N803
) -> tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]]:
    """Initialize the model parameters for both pre and post concatenation layers.

    Args:
        key: Random number generator key for parameter initialization.
        n_layers: Number of layers in the model.
        layer_dim: Dimension of each hidden layer.
        L: Number of frequency bands for positional encoding.

    Returns:
        A tuple containing two lists of layer parameters:
        - pre_concat_layers: Parameters for layers before concatenation
        - post_concat_layers: Parameters for layers after concatenation

    """
    key, *subkeys = jax.random.split(key, 3)
    input_layer = _init__linear_layer(subkeys, 6 * L, layer_dim)

    pre_concat_layers = [input_layer]
    for _ in range(n_layers // 2):
        key, *subkeys = jax.random.split(key, 3)
        pre_concat_layers.append(_init__linear_layer(subkeys, layer_dim, layer_dim))

    key, *subkeys = jax.random.split(key, 3)
    middle_layer = _init__linear_layer(subkeys, layer_dim + 6 * L, layer_dim)

    post_concat_layers = [middle_layer]
    for _ in range(n_layers // 2 - 1):
        key, *subkeys = jax.random.split(key, 3)
        post_concat_layers.append(_init__linear_layer(subkeys, layer_dim, layer_dim))

    key, *subkeys = jax.random.split(key, 3)
    post_concat_layers.append(_init__linear_layer(subkeys, layer_dim, 1))

    return pre_concat_layers, post_concat_layers


def _init_exu_layer(keys: tuple[jax.Array], in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    w = truncated_normal(0.5)(keys[0], shape=(in_dim, out_dim)) + 4
    b = truncated_normal(0.5)(keys[1], shape=(in_dim))
    return {"w": w, "b": b}


def _init__linear_layer(keys: tuple[jax.Array], in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    w = jax.random.uniform(
        keys[0],
        shape=(out_dim, in_dim),
        minval=-jnp.sqrt(1 / in_dim),
        maxval=jnp.sqrt(1 / in_dim),
    )
    b = jax.random.uniform(
        keys[1],
        shape=(out_dim),
        minval=-jnp.sqrt(1 / in_dim),
        maxval=jnp.sqrt(1 / in_dim),
    )
    return {"w": w, "b": b}


def exu(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Apply the Exp-centered Unit (ExU) transformation to the input.

    Args:
        params (dict[str, jax.Array]): Dictionary containing weight ('w') and bias ('b') parameters.
        x (jax.Array): Input array to transform.

    Returns:
        jax.Array: The transformed input after applying the ExU operation.

    """
    return (x - params["b"]) @ jnp.exp(params["w"])


@partial(jax.vmap, in_axes=(None, 0, None))
def forward(
    params: tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]], dict],
    coords: jax.Array,
    policy: jmp.Policy,
) -> jax.Array:
    """Forward pass of the model.

    Args:
        params (tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]], dict):
            tuple containing two lists of layers, the pre-concatenation layers and the
            post-concatenation layers. Each list entry is a dict with the keys 'w' and 'b',
            corresponding to weight and bias arrays.
        coords (jax.Array): shape (3,). Input coordinates.
        policy (jmp.Policy): Mixed precision policy for type casting.

    Returns:
        jax.Array: shape (1,). Output of the model.

    """
    coords = policy.cast_to_compute(coords)
    pre_concat_layers, post_concat_layers = params
    L = pre_concat_layers[0]["w"].shape[1] / 6  # noqa: N806
    pos_enc = _positional_encoding(coords, L, policy)

    x = pos_enc
    for layer in pre_concat_layers:
        layer = policy.cast_to_compute(layer)
        x = jax.nn.relu(jnp.dot(layer["w"], x) + layer["b"])

    x = jnp.concat([x, pos_enc])
    for layer in post_concat_layers[:-1]:
        layer = policy.cast_to_compute(layer)
        x = jax.nn.relu(jnp.dot(layer["w"], x) + layer["b"])

    # no relu on final layer
    final_layer = policy.cast_to_compute(post_concat_layers[-1])
    return policy.cast_to_output((jnp.dot(final_layer["w"], x) + final_layer["b"]).squeeze())


def loss_fn(
    params: tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]],
    coords: jax.Array,
    gt: jax.Array,
    sampling_distances: jax.Array,
    s: float | None,
    k: float | None,
    slice_size_cm: float,
    policy: jmp.Policy,
) -> jax.Array:
    """Calculate the mean squared error loss between predicted and ground truth intensities.

    Args:
        params: Model parameters consisting of pre and post concatenation layers.
        coords: Input coordinates for the model.
        gt: Ground truth intensity values.
        sampling_distances: Distances between sampling points.
        s: Scaling parameter.
        k: Scaling parameter.
        slice_size_cm: Size of the CT slice in centimeters.
        policy (jmp.Policy): Mixed precision policy for type casting.

    Returns:
        Mean squared error loss between predicted and ground truth intensities.

    """
    attenuation_preds = forward(params, coords, policy)
    intensity_pred = beer_lambert_law(attenuation_preds, sampling_distances, s, k, slice_size_cm)
    return (intensity_pred - gt) ** 2


def _positional_encoding(coords: jax.Array, L: int, policy: jmp.Policy) -> jax.Array:  # noqa: N803
    """Compute the positional encoding for the input coordinates.

    This function applies a positional encoding to the input coordinates using
    sinusoidal functions of varying frequencies. The encoding is used to map
    the input coordinates into a higher-dimensional space, which helps the
    model to capture spatial relationships.

    Args:
        coords (jax.Array): An array of shape (3,) representing the input
            coordinates for which the positional encoding is to be computed.
        L (int): The number of frequency bands to use for the encoding.
        policy (jmp.Policy): Mixed precision policy for type casting.

    Returns:
        jax.Array: An array of shape (6 * L,) containing the positional
        encoding of the input coordinates.

    """
    dtype = (
        jnp.float32 if policy.compute_dtype == jnp.float16 else policy.compute_dtype
    )  # float16 will overflow if L >= 15
    freq_bands = jnp.power(2.0, jnp.arange(L, dtype=dtype)) * jnp.pi
    angles = jnp.expand_dims(coords, axis=-1) * jnp.expand_dims(freq_bands, axis=0)
    angles = policy.cast_to_compute(angles.reshape(-1))
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])
