"""Contains functions for training the model."""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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

from ctnerf import model, ray_sampling
from ctnerf.image_creation.ct_creation import array_to_sitk, run_inference
from ctnerf.rays import get_coarse_samples, get_fine_samples
from ctnerf.setup import setup_functions
from ctnerf.setup.config import TrainingConfig, get_training_config


class ModelState(NamedTuple):
    """Class holding the state of a model during training.

    Attributes:
        params: Model parameters
        opt_state: Optimizer state
        scaler_state: Loss scaler state for mixed precision training

    """

    params: optax.Params
    opt_state: optax.OptState
    scaler: jmp.LossScale


class TrainState(NamedTuple):
    """Container for both coarse and fine model states during training.

    Attributes:
        coarse: ModelState for the coarse model
        fine: Optional ModelState for the fine model

    """

    coarse: ModelState
    fine: ModelState | None  # Fine model is optional


def _initialize_or_restore_state(
    conf: TrainingConfig,
    key: jax.Array,
) -> tuple[TrainState, int, str, jax.Array]:
    """Initialize or restore training state for coarse and fine models."""
    key, coarse_key, fine_key = jax.random.split(key, 3)

    # Initialize Coarse
    initial_coarse_params = setup_functions.get_model(conf, fine=False, key=coarse_key)
    optimizer, initial_coarse_opt_state = setup_functions.get_optimizer(
        conf,
        initial_coarse_params,
    )
    initial_coarse_scaler_state = setup_functions.get_loss_scaler(conf)

    initial_coarse_model_state = ModelState(
        params=initial_coarse_params,
        opt_state=initial_coarse_opt_state,
        scaler=initial_coarse_scaler_state,
    )

    # Initialize Fine (Optional)
    initial_fine_model_state = None
    if conf.n_fine_samples > 0:
        initial_fine_params = setup_functions.get_model(conf, fine=True, key=fine_key)
        _, initial_fine_opt_state = setup_functions.get_optimizer(
            conf,
            initial_fine_params,
        )
        initial_fine_scaler_state = setup_functions.get_loss_scaler(conf)

        initial_fine_model_state = ModelState(
            params=initial_fine_params,
            opt_state=initial_fine_opt_state,
            scaler=initial_fine_scaler_state,
        )

    initial_train_state = TrainState(
        coarse=initial_coarse_model_state,
        fine=initial_fine_model_state,
    )

    initial_save_state = {
        "coarse_params": initial_coarse_params,
        "coarse_opt_state": initial_coarse_opt_state,
        "coarse_scaler_scale": 2.0**15,
        "fine_params": initial_fine_params if initial_fine_model_state else None,
        "fine_opt_state": initial_fine_opt_state if initial_fine_model_state else None,
        "fine_scaler_scale": 2.0**15 if initial_fine_model_state else None,
        "step": 0,
        "run_hash": "",
    }

    start_step = 0
    run_hash = ""
    train_state = initial_train_state

    if conf.resume_training:
        if not conf.checkpoint_dir.exists():
            msg = f"Checkpoint directory {conf.checkpoint_dir} does not exist"
            raise FileNotFoundError(msg)
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=conf.checkpoint_dir,
            target=initial_save_state,
        )
        if restored_state is initial_save_state:
            msg = "Restored state is the same as initial state. Checkpoint may not exist."
            raise RuntimeError(msg)
        if isinstance(initial_coarse_model_state.scaler, jmp.DynamicLossScale):
            coarse_scaler = jmp.DynamicLossScale(initial_save_state["coarse_scaler_scale"])
        else:
            coarse_scaler = jmp.NoOpLossScale()
        coarse_model_state = ModelState(
            params=restored_state["coarse_params"],
            opt_state=restored_state["coarse_opt_state"],
            scaler=coarse_scaler,
        )
        fine_model_state = None
        if conf.n_fine_samples > 0 and restored_state.get("fine_params") is not None:
            if isinstance(initial_fine_model_state.scaler, jmp.DynamicLossScale):
                fine_scaler = jmp.DynamicLossScale(initial_save_state["fine_scaler_scale"])
            else:
                fine_scaler = jmp.NoOpLossScale()
            fine_model_state = ModelState(
                params=restored_state["fine_params"],
                opt_state=restored_state["fine_opt_state"],
                scaler=fine_scaler,
            )

        train_state = TrainState(coarse=coarse_model_state, fine=fine_model_state)
        start_step = restored_state["step"] + 1
        run_hash = restored_state.get("run_hash", "")

    return train_state, start_step, run_hash, key, optimizer


def _save_checkpoint(
    conf: TrainingConfig,
    epoch: int,
    train_state: TrainState,
    aim_run_hash: str,
) -> None:
    """Save the training state for coarse and fine models."""
    save_dict = {
        "coarse_params": train_state.coarse.params,
        "coarse_opt_state": train_state.coarse.opt_state,
        "coarse_scaler_scale": train_state.coarse.scaler.loss_scale,
        "step": epoch,
        "run_hash": aim_run_hash,
    }
    if train_state.fine is not None:
        save_dict["fine_params"] = train_state.fine.params
        save_dict["fine_opt_state"] = train_state.fine.opt_state
        save_dict["fine_scaler_scale"] = train_state.fine.scaler.loss_scale

    checkpoints.save_checkpoint(
        ckpt_dir=conf.checkpoint_dir.parent if conf.resume_training else conf.checkpoint_dir,
        target=save_dict,
        step=epoch,
        prefix="checkpoint_",
        keep=10,
        overwrite=False,
    )


def _eval(
    model: tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]],
    aim_run: Run,
    tag: str,
    epoch: int,
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
        aim_run.track(mae, name="mae", context={"model": tag}, epoch=epoch)

        ct_slice = generated_ct[::-1, :, generated_ct.shape[2] // 2]
        image = PILImage.fromarray(
            ((ct_slice.astype(np.float32) + 1024) / 4095 * 255).astype(np.uint8),
        )
        aim_run.track(Image(image), name="cross section", context={"model": tag}, epoch=epoch)


def create_training_step(
    optimizer: optax.GradientTransformation,
    conf: TrainingConfig,
    model_policy: jmp.Policy,
) -> Callable:
    """Create the JIT-compiled training step function for coarse and fine models."""

    def compute_coarse_loss_and_aux(
        coarse_params: optax.Params,
        rand_key: jax.Array,
        start_positions: jax.Array,
        heading_vectors: jax.Array,
        intensities: jax.Array,
        ray_bounds: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute coarse loss and return data needed for fine sampling."""
        key_coarse, key_fine = jax.random.split(rand_key)

        coarse_sample_ts, coarse_samples, coarse_sampling_distances = get_coarse_samples(
            key_coarse,
            start_positions,
            heading_vectors,
            ray_bounds,
            conf.n_coarse_samples,
            conf.plateau_ratio,
            getattr(ray_sampling, conf.coarse_sampling_function),
        )

        coarse_loss, coarse_attenuation_preds = model.loss_fn(
            coarse_params,
            coarse_samples,
            intensities,
            coarse_sampling_distances,
            conf.s,
            conf.k,
            conf.slice_size_cm,
            model_policy,
        )

        aux = {
            "coarse_ts": coarse_sample_ts,
            "coarse_preds": coarse_attenuation_preds,
            "coarse_dists": coarse_sampling_distances,
            "key_fine_sampling": key_fine,
            "start_pos": start_positions,
            "heading_vector": heading_vectors,
            "ray_bounds": ray_bounds,
            "intensities": intensities,
        }
        return coarse_loss, aux

    def batched_coarse_loss(
        coarse_params: optax.Params,
        keys: jax.Array,
        batch_start_positions: jax.Array,
        batch_heading_vectors: jax.Array,
        batch_intensities: jax.Array,
        batch_ray_bounds: jax.Array,
        coarse_scaler: jmp.LossScale,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        loss, aux = jax.vmap(compute_coarse_loss_and_aux, in_axes=(None, 0, 0, 0, 0, 0))(
            coarse_params,
            keys,
            batch_start_positions,
            batch_heading_vectors,
            batch_intensities,
            batch_ray_bounds,
        )
        loss = coarse_scaler.scale(jnp.sum(loss))
        return loss, aux

    def compute_fine_loss(
        fine_params: optax.Params,
        coarse_aux: dict[str, jax.Array],
    ) -> jax.Array:
        """Compute fine loss using fine samples derived from coarse results."""
        fine_samples, fine_sampling_distances = get_fine_samples(
            coarse_aux["key_fine_sampling"],
            coarse_aux["start_pos"],
            coarse_aux["heading_vector"],
            coarse_aux["ray_bounds"],
            coarse_aux["coarse_ts"],
            coarse_aux["coarse_preds"],
            coarse_aux["coarse_dists"],
            conf.n_fine_samples,
            getattr(ray_sampling, conf.fine_sampling_function),
        )

        fine_loss, _ = model.loss_fn(
            fine_params,
            fine_samples,
            coarse_aux["intensities"],
            fine_sampling_distances,
            conf.s,
            conf.k,
            conf.slice_size_cm,
            model_policy,
        )
        return fine_loss

    def batched_fine_loss(
        fine_params: optax.Params,
        coarse_aux: dict[str, jax.Array],
        fine_scaler: jmp.LossScale,
    ) -> jax.Array:
        loss = jax.vmap(compute_fine_loss, in_axes=(None, 0))(
            fine_params,
            coarse_aux,
        )
        return fine_scaler.scale(jnp.sum(loss))

    def _step_internal(
        train_state: TrainState,
        step_rand_key: jax.Array,
        batch_start_positions: jax.Array,
        batch_heading_vectors: jax.Array,
        batch_intensities: jax.Array,
        batch_ray_bounds: jax.Array,
    ) -> tuple[dict[str, jax.Array], TrainState]:
        keys = jax.random.split(step_rand_key, batch_start_positions.shape[0])

        # Compute coarse loss and grads
        coarse_loss_vg_fn = jax.value_and_grad(batched_coarse_loss, has_aux=True)
        (coarse_loss, coarse_aux_batch), coarse_grads = coarse_loss_vg_fn(
            train_state.coarse.params,
            keys,
            batch_start_positions,
            batch_heading_vectors,
            batch_intensities,
            batch_ray_bounds,
            train_state.coarse.scaler,
        )
        (coarse_loss, coarse_grads) = train_state.coarse.scaler.unscale((coarse_loss, coarse_grads))

        # Update coarse state
        coarse_grads_finite = jmp.all_finite(coarse_grads)
        next_coarse_scaler = train_state.coarse.scaler.adjust(coarse_grads_finite)
        coarse_updates, next_coarse_opt_state = optimizer.update(
            coarse_grads,
            train_state.coarse.opt_state,
        )
        next_coarse_params = optax.apply_updates(train_state.coarse.params, coarse_updates)
        next_coarse_params, next_coarse_opt_state = jmp.select_tree(
            coarse_grads_finite,
            (next_coarse_params, next_coarse_opt_state),
            (train_state.coarse.params, train_state.coarse.opt_state),
        )

        # Compute fine loss and grads
        fine_loss = jnp.nan
        if train_state.fine is not None:
            fine_loss_vg_fn = jax.value_and_grad(batched_fine_loss)
            fine_loss, fine_grads = fine_loss_vg_fn(
                train_state.fine.params,
                coarse_aux_batch,
                train_state.fine.scaler,
            )
            (fine_loss, fine_grads) = train_state.fine.scaler.unscale((fine_loss, fine_grads))

            # Update fine state
            fine_grads_finite = jmp.all_finite(fine_grads)
            next_fine_scaler = train_state.fine.scaler.adjust(fine_grads_finite)
            fine_updates, next_fine_opt_state = optimizer.update(
                fine_grads,
                train_state.fine.opt_state,
            )
            next_fine_params = optax.apply_updates(train_state.fine.params, fine_updates)
            next_fine_params, next_fine_opt_state = jmp.select_tree(
                fine_grads_finite,
                (next_fine_params, next_fine_opt_state),
                (train_state.fine.params, train_state.fine.opt_state),
            )

        next_coarse_ms = ModelState(
            next_coarse_params,
            next_coarse_opt_state,
            next_coarse_scaler,
        )
        if train_state.fine is not None:
            next_fine_ms = ModelState(
                next_fine_params,
                next_fine_opt_state,
                next_fine_scaler,
            )
        else:
            next_fine_ms = None

        next_train_state = TrainState(coarse=next_coarse_ms, fine=next_fine_ms)

        return coarse_loss, fine_loss, next_train_state

    return jax.jit(_step_internal)


def train(config_path: Path) -> None:
    """Train the CT-NeRF model using the provided configuration.

    Args:
        config_path: Path to the configuration file containing training parameters.

    """
    conf = get_training_config(config_path)
    model_policy = setup_functions.get_dtype_policy(conf)
    key = jax.random.key(conf.seed)
    train_state, start_epoch, run_hash, key, optimizer = _initialize_or_restore_state(conf, key)
    dataloader = setup_functions.get_dataloader(conf)
    aim_run = setup_functions.get_aim_run(conf, run_hash)

    training_step = create_training_step(
        optimizer=optimizer,
        conf=conf,
        model_policy=model_policy,
    )

    for epoch in range(start_epoch, 10000):
        for i, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            batch = jax.device_put(
                jax.tree.map(lambda x: jnp.array(x, dtype=conf.dtypes["input_dtype"]), batch_data),
            )
            start_positions, heading_vectors, intensities, ray_bounds = batch

            key, step_key = jax.random.split(key)
            coarse_loss, fine_loss, train_state = training_step(
                train_state,
                step_key,
                start_positions,
                heading_vectors,
                intensities,
                ray_bounds,
            )

            if i % 100 == 0:
                step_num = i + epoch * len(dataloader)
                aim_run.track(coarse_loss, name="coarse_loss", step=step_num, epoch=epoch)
                if train_state.fine is not None:
                    aim_run.track(fine_loss, name="fine_loss", step=step_num, epoch=epoch)
                if isinstance(train_state.coarse.scaler, jmp.DynamicLossScale):
                    aim_run.track(
                        train_state.coarse.scaler.loss_scale,
                        name="coarse_loss_scale",
                        step=step_num,
                        epoch=epoch,
                    )
                if train_state.fine and isinstance(train_state.fine.scaler, jmp.DynamicLossScale):
                    aim_run.track(
                        train_state.fine.scaler.loss_scale,
                        name="fine_loss_scale",
                        step=step_num,
                        epoch=epoch,
                    )

        _eval(train_state.coarse.params, aim_run, "coarse", epoch, conf)
        if train_state.fine is not None:
            _eval(train_state.fine.params, aim_run, "fine", epoch, conf)

        if (epoch + 1) % conf.checkpoint_interval == 0:
            _save_checkpoint(
                conf,
                epoch,
                train_state,
                aim_run.hash,
            )
