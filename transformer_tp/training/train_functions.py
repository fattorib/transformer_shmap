from typing import Any, Callable, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax.experimental import multihost_utils
from jax.lax import with_sharding_constraint


def convert_arr_sharding(x, mesh, batch_spec):
    """Convert local array to global jax.Array."""
    return multihost_utils.host_local_array_to_global_array(x, mesh, batch_spec)


def to_bf16(t: Any) -> Any:
    """Cast pytree to bf16."""
    return jax.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t
    )


def global_norm_sq(t: Any) -> Any:
    """Compute squared global norm of a pytree."""
    pre_sqrt = sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(t)])

    return pre_sqrt


def train_step(
    params: Any,
    batch: jnp.array,
    accum_steps: int = 8,
    model: Any = None,
):
    """Distributed loss and grad computation."""
    _, context = batch.shape

    # reshape to add a microbatch dimension
    batch = jax.lax.reshape(
        batch, (accum_steps, batch.shape[0] // accum_steps, context)
    )

    def loss_fn(params, batch):
        _, loss = model.apply(
            {"params": params["params"]},
            x=batch,
            labels=batch,
        )
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn)

    # accumulate gradients
    def cumul_minibatch_step(carry, x_y):
        cumul_loss, cumul_grads = carry
        minibatch = x_y
        loss, grads = grad_fn(to_bf16(params), minibatch)
        cumul_grads = jax.tree_map(jnp.add, cumul_grads, grads)
        return (cumul_loss + loss, cumul_grads), None

    grad_init = to_bf16(jax.tree_util.tree_map(jnp.zeros_like, params))

    with jax.named_scope("scanned_microbatch"):
        (loss, grads), _ = jax.lax.scan(
            cumul_minibatch_step,
            init=(jnp.zeros(()), grad_init),
            xs=batch,
        )

    with jax.named_scope("gradient_all_reduce"):
        grads = jax.lax.pmean(grads, axis_name="dp")
        loss = jax.lax.pmean(loss, axis_name="dp")

    loss, grads = jax.tree_util.tree_map(lambda x: x / accum_steps, (loss, grads))

    grad_norm = jax.lax.psum(global_norm_sq(grads), axis_name="mp")

    metrics = {
        "train/loss": loss,
        "train/ppl": jnp.exp(loss),
        "train/norms/grad_norm": jnp.sqrt(grad_norm),
    }

    return grads, metrics


def eval_step(
    params: Any,
    batch: jnp.array,
    model: Any,
):
    """Distributed eval step."""
    _, loss = model.apply({"params": params["params"]}, x=batch, labels=batch)
    loss = jax.lax.pmean(jnp.mean(loss), axis_name="dp")
    metrics = {"validation/loss": loss, "validation/ppl": jnp.exp(loss)}
    return metrics


def update_opt_state(
    params: Any, grads: Any, opt_state: Any, optimizer: Any, tp_spec: Any
):
    """Updates the optimizer state and params."""
    params = with_sharding_constraint(params, tp_spec)
    grads = with_sharding_constraint(grads, tp_spec)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def create_train_state(
    rng: jax.random.PRNGKey,
    learning_rate_fn: Union[float, Callable],
    weight_decay: float,
    model: nn.Module,
) -> Tuple[Any, optax.GradientTransformation]:
    """Gets the abstract model shapes and sets up the optimizer."""

    batch = jax.numpy.ones(shape=(1, model.config.block_size), dtype=jnp.int32)
    param_abstract = jax.eval_shape(model.init, rng, batch)

    # no wd for bias or LN terms
    mask = jax.tree_map(
        lambda x: x.ndim != 1,
        param_abstract,
    )

    tx = optax.chain(
        optax.clip(1.0),
        optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=weight_decay,
            mask=mask,
            b2=0.95,
            mu_dtype=jnp.bfloat16,
        ),
    )

    return param_abstract, tx
