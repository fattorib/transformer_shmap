"""
Checkpointing utilities for saving and resuming during training.
"""

from typing import Any, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints, train_state


def to_bf16(t: Any) -> Any:
    """Cast pytree to bf16."""
    return jax.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t
    )


def to_f32(t: Any) -> Any:
    """Cast pytree to fp32."""
    return jax.tree_map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t
    )


def save_checkpoint_params(params: Any, step: int, workdir: str) -> None:
    """Save a copy of params to checkpoint location. Params are cast to bf16 before saving."""
    if jax.process_index() == 0:
        params = jax.device_get(params)

        params = to_bf16(params)

        faux_state = train_state.TrainState(
            step=step, apply_fn=None, params=params, tx=None, opt_state=None
        )
        checkpoints.save_checkpoint(
            workdir, faux_state, step, keep=5, overwrite=True, prefix="params_"
        )


def save_checkpoint_optimizer(opt_state: Any, step: int, workdir: str) -> None:
    """Saves a copy of opt_state to checkpoint location."""
    if jax.process_index() == 0:
        opt_state = jax.device_get(opt_state)
        faux_state = train_state.TrainState(
            step=step, apply_fn=None, params=None, tx=None, opt_state=opt_state
        )
        checkpoints.save_checkpoint(
            workdir, faux_state, step, keep=5, overwrite=True, prefix="opt_"
        )


def restore_checkpoint_params(workdir: str, param_spec: Any) -> Tuple[Any, Any, int]:
    """Restores and reconstructs the most recent parameter dict."""
    restored = checkpoints.restore_checkpoint(workdir, target=None, prefix="params_")
    with jax.default_device(jax.devices("cpu")[0]):
        params = flax.core.freeze(restored["params"])
        params = jax.tree_map(
            lambda x, y: nn.Partitioned(
                value=jnp.array(y["value"]), names=x, mesh=None
            ),
            param_spec,
            params,
        )
        # make sure to recast params to f32 upon deserializing
        params = to_f32(params)

        return params, restored["step"]


def restore_checkpoint_opt(opt_spec: Any, workdir: str) -> Tuple[Any, Any, int]:
    """Restores and reconstructs the most recent opt state dict."""
    restored = checkpoints.restore_checkpoint(workdir, target=None, prefix="opt_")

    with jax.default_device(jax.devices("cpu")[0]):
        mu_pytree = jax.tree_map(
            lambda x: jnp.array(x), restored["opt_state"]["1"]["0"]["mu"]
        )
        mu_pytree = jax.tree_map(
            lambda x, y: nn.Partitioned(
                value=jnp.array(y["value"]), names=x, mesh=None
            ),
            opt_spec[1][0].mu,
            flax.core.freeze(mu_pytree),
        )

        nu_pytree = jax.tree_util.tree_map(
            lambda x: jnp.array(x), restored["opt_state"]["1"]["0"]["nu"]
        )

        nu_pytree = jax.tree_map(
            lambda x, y: nn.Partitioned(
                value=jnp.array(y["value"]), names=x, mesh=None
            ),
            opt_spec[1][0].nu,
            flax.core.freeze(nu_pytree),
        )

        count_pytree = jax.tree_map(
            lambda x: jnp.array(x), restored["opt_state"]["1"]["0"]["count"]
        )

        restoredadamstate = optax.ScaleByAdamState(
            count_pytree,
            flax.core.FrozenDict(mu_pytree),
            flax.core.FrozenDict(nu_pytree),
        )

        opt_state = (
            optax.EmptyState(),
            (
                restoredadamstate,
                optax.MaskedState(inner_state=optax.EmptyState()),
                optax.ScaleByScheduleState(count=jnp.array(restored["step"])),
            ),
        )
        return opt_state
