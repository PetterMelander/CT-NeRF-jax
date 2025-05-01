"""Contains functions for training the model."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import SimpleITK as sitk
from aim import Image, Run
from flax.training import checkpoints
from PIL import Image as PILImage
from tqdm import tqdm

from ctnerf import model
from ctnerf.image_creation.ct_creation import array_to_sitk, run_inference
from ctnerf.rays import get_coarse_samples
from ctnerf.setup import setup_functions
from ctnerf.setup.config import TrainingConfig, get_training_config


def train(config_path: Path) -> None:
    """Train the model."""
    conf = get_training_config(config_path)
    model_policy = setup_functions.get_dtype_policy(conf)
    scaler = setup_functions.get_loss_scaler(conf)
    key = jax.random.key(conf.sampling_seed)
    coarse_model = setup_functions.get_model(conf)
    optimizer, opt_state = setup_functions.get_optimizer(conf, coarse_model)
    dataloader = setup_functions.get_dataloader(conf)
    initial_state_dict = {
        "params": coarse_model,
        "opt_state": opt_state,
        "step": 0,
        "run_hash": "",
    }
    if conf.resume_training:
        if not conf.checkpoint_dir.exists():
            msg = f"Checkpoint directory {conf.checkpoint_dir} does not exist"
            raise FileNotFoundError(msg)
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=conf.checkpoint_dir,
            target=initial_state_dict,
        )
        coarse_model = restored_state["params"]
        opt_state = restored_state["opt_state"]
        start_step = restored_state["step"] + 1
        run_hash = restored_state["run_hash"]
        checkpoint_dir = conf.checkpoint_dir.parent
    else:
        start_step = 0
        run_hash = ""
        checkpoint_dir = conf.checkpoint_dir
    aim_run = setup_functions.get_aim_run(conf, run_hash)

    def single_loss_fn(
        rand_key: jax.Array,
        params: tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]],
        start_positions: jax.Array,
        heading_vectors: jax.Array,
        intensities: jax.Array,
        ray_bounds: jax.Array,
    ) -> jax.Array:
        coarse_sample_ts, coarse_samples, coarse_sampling_distances = get_coarse_samples(
            rand_key,
            start_positions,
            heading_vectors,
            ray_bounds,
            conf.n_coarse_samples,
            conf.plateau_ratio,
            conf.coarse_sampling_function,
        )

        return model.loss_fn(
            params,
            coarse_samples,
            intensities,
            coarse_sampling_distances,
            conf.s,
            conf.k,
            conf.slice_size_cm,
            model_policy,
        )

    def batch_loss_fn(
        params: tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]],
        batch_rand_key: jax.Array,
        start_positions: jax.Array,
        heading_vectors: jax.Array,
        intensities: jax.Array,
        ray_bounds: jax.Array,
        scaler: jmp.LossScale,
    ) -> jax.Array:
        batch_size = start_positions.shape[0]
        keys = jax.random.split(batch_rand_key, batch_size)

        losses = jax.vmap(single_loss_fn, in_axes=(0, None, 0, 0, 0, 0))(
            keys,
            params,
            start_positions,
            heading_vectors,
            intensities,
            ray_bounds,
        )
        return scaler.scale(jnp.sum(losses))

    @jax.jit
    def step(
        params: optax.Params,
        opt_state: optax.OptState,
        step_rand_key: jax.Array,
        batch_start_positions: jax.Array,
        batch_heading_vectors: jax.Array,
        batch_intensities: jax.Array,
        batch_ray_bounds: jax.Array,
        scaler: jmp.LossScale,
    ) -> tuple[jax.Array, optax.Params, optax.OptState]:
        total_batch_loss, grads = jax.value_and_grad(batch_loss_fn)(
            params,
            step_rand_key,
            batch_start_positions,
            batch_heading_vectors,
            batch_intensities,
            batch_ray_bounds,
            scaler,
        )
        total_batch_loss, grads = scaler.unscale((total_batch_loss, grads))
        grads_finite = jmp.all_finite(grads)
        next_scaler = scaler.adjust(grads_finite)
        updates, next_opt_state = optimizer.update(grads, opt_state)
        next_params = optax.apply_updates(params, updates)
        next_params, next_opt_state = jmp.select_tree(
            grads_finite,
            (next_params, next_opt_state),
            (params, opt_state),
        )

        return total_batch_loss, next_params, next_opt_state, next_scaler

    for epoch in range(start_step, 10000):
        for i, (start_positions, heading_vectors, intensities, ray_bounds) in enumerate(
            tqdm(dataloader),
        ):
            start_positions, heading_vectors, intensities, ray_bounds = (
                jnp.array(start_positions, dtype=conf.dtypes["input_dtype"]),
                jnp.array(heading_vectors, dtype=conf.dtypes["input_dtype"]),
                jnp.array(intensities, dtype=conf.dtypes["input_dtype"]),
                jnp.array(ray_bounds, dtype=conf.dtypes["input_dtype"]),
            )
            key, step_key = jax.random.split(key)
            loss, coarse_model, opt_state, scaler = step(
                coarse_model,
                opt_state,
                step_key,
                start_positions,
                heading_vectors,
                intensities,
                ray_bounds,
                scaler,
            )
            if i % 100 == 0:
                aim_run.track(loss, name="coarse_loss", step=i + epoch * len(dataloader))

        _eval(
            coarse_model,
            aim_run,
            "coarse",
            conf,
        )
        if epoch % conf.checkpoint_interval == 0:
            state_to_save = {
                "params": coarse_model,
                "opt_state": opt_state,
                "step": epoch,
                "run_hash": aim_run.hash,
            }
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=state_to_save,
                step=epoch,
                prefix="checkpoint_",
                keep=10,
                overwrite=False,
            )


def _eval(
    model: tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]],
    aim_run: Run,
    tag: str,
    conf: TrainingConfig,
) -> None:
    if conf.source_ct_path is not None:
        generated_ct = run_inference(
            model,
            conf.ct_size,
            4096 * 64,
            conf.attenuation_scaling_factor,
        )
        ct_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        generated_ct = array_to_sitk(generated_ct, direction=ct_direction)
        generated_ct = sitk.GetArrayFromImage(generated_ct)

        source_ct_image = sitk.ReadImage(str(conf.source_ct_path))
        source_ct_image = sitk.GetArrayFromImage(source_ct_image)

        mae = np.mean(np.abs(generated_ct - source_ct_image))
        aim_run.track(mae, name="mae", context={"model": tag})

        ct_slice = generated_ct[::-1, :, generated_ct.shape[2] // 2]
        image = PILImage.fromarray(
            ((ct_slice.astype(np.float32) + 1024) / 4095 * 255).astype(np.uint8),
        )
        aim_run.track(
            Image(image),
            name="cross section",
            context={"model": tag},
        )
