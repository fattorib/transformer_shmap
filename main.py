import argparse
import logging
from dataclasses import fields, replace
from functools import partial
from time import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from transformer_tp.models import Transformer, model_getter
from transformer_tp.partitioning import create_opt_spec
from transformer_tp.training import (create_train_state, eval_step, train_step,
                                     update_opt_state)
from transformer_tp.utils import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse():
    parser = argparse.ArgumentParser(description="Tensor Parallelism with shard_map")
    parser.add_argument("--cfg", default="conf/config.yaml", type=str)
    parser.add_argument("--model-cfg", default="conf/model_config.yaml", type=str)
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    # getting system information
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_host = num_devices // num_local_devices
    platform = jax.local_devices()[0].platform

    mesh = Mesh(
        np.array(jax.devices()).reshape(cfg.training.dp, cfg.training.mp), ("dp", "mp")
    )

    if jax.process_index() == 0:
        logger.debug(f"VM setup with {num_devices} devices.")
        logger.debug(f"Host setup with {num_local_devices} devices.")
        logger.debug(f"Using platform: {platform}.")
        logger.debug(f"Mesh Shape (dp,mp): {(cfg.training.dp, cfg.training.mp)}.")

    # setting up GCP bucket/client info if training on TPU
    checkpoint_prefix = ""
    client = None
    if platform == "tpu":
        if cfg.data.bucket_path is not None:
            from google.cloud import storage
            client = storage.Client()
            checkpoint_prefix = "gs://"

    model, model_config = model_getter(
        cfg.model.size, config_path=args.model_cfg, return_cfg=True
    )

    # set up sharded config and model
    replaced_args = {
        "num_shard": mesh.shape["mp"],
        "tp_comms": True if mesh.shape["mp"] > 1 else False,
    }
    sharded_config = replace(model_config, **replaced_args)
    model_shard = Transformer(sharded_config)

    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=cfg.training.peak_learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        decay_steps=cfg.training.total_steps,
        end_value=cfg.training.end_learning_rate,
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    resume_step = 0

    param_abstract, tx = create_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=cfg.training.weight_decay,
        model=model,
    )

    # Setup partition specs
    param_spec = nn.get_partition_spec(param_abstract)

    grad_spec = param_spec
    batch_spec = P("dp", None)

    # Setup params and optimizer states
    with mesh:
        # do actual layer init wrapping with pjit
        if not args.resume:
            batch = jnp.ones(
                (cfg.training.batch_size // 4, model.config.block_size), dtype=jnp.int32
            )
            params = pjit(
                model.init,
                in_axis_resources=(P(None), batch_spec),
                out_axis_resources=param_spec,
            )(rng, batch)

        opt_state_shapes = jax.eval_shape(tx.init, param_abstract)

        opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

        if not args.resume:
            opt_state = pjit(
                tx.init,
                in_axis_resources=(param_spec,),
                out_axis_resources=opt_state_spec,
            )(params)

    if jax.process_index() == 0:
        logger.debug(f"Params and Optimizer state compiled and sharded")

    train_step_tp = jax.jit(
        shard_map(
            partial(
                train_step,
                model=model_shard,
                accum_steps=cfg.training.gradient_accumulation_steps,
            ),
            in_specs=(param_spec, batch_spec),
            out_specs=(grad_spec, P(None)),
            mesh=mesh,
            check_rep=False,
        )
    )

    eval_step_tp = jax.jit(
        shard_map(
            partial(eval_step, model=model_shard),
            in_specs=(param_spec, batch_spec),
            out_specs=(P(None)),
            mesh=mesh,
            check_rep=False,
        )
    )

    with mesh:
        update_opt_step_tp = pjit(
            partial(update_opt_state, optimizer=tx, tp_spec=grad_spec),
            in_axis_resources=(param_spec, grad_spec, opt_state_spec),
            out_axis_resources=(param_spec, opt_state_spec),
            donate_argnums=0,
        )

    if num_host > 1:
        convert_arr_sharding = (
            lambda x: multihost_utils.host_local_array_to_global_array(
                x, mesh, batch_spec
            )
        )
    else:
        convert_arr_sharding = lambda x: x

    if args.resume:
        params, step = restore_checkpoint_params(
            workdir=f"{checkpoint_prefix}{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/params",
            param_spec=param_spec,
        )
        resume_step = int(step)

        opt_state = restore_checkpoint_opt(
            opt_state_spec,
            workdir=f"{checkpoint_prefix}{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/opt",
        )

        import gc

        gc.collect()

        with mesh:
            params = pjit(
                lambda x: x,
                in_axis_resources=(param_spec,),
                out_axis_resources=param_spec,
                donate_argnums=0,
            )(params)
            opt_state = pjit(
                lambda x: x,
                in_axis_resources=(opt_state_spec,),
                out_axis_resources=opt_state_spec,
                donate_argnums=0,
            )(opt_state)

        if jax.process_index() == 0:
            logger.debug(f"Resuming training from step {resume_step}")

    local_batch_size = cfg.training.batch_size // (cfg.training.dp)

    total_tokens = compute_tokens(
        cfg.training.total_steps,
        max_context=cfg.training.train_context,
        num_host=jax.host_count(),
        batch_per_host=cfg.training.batch_size,
    )

    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(id=id, resume="allow", project=cfg.data.wandb_project)
        flat_dict = flatten_dict(cfg)

        for field in fields(model_config):
            flat_dict[f"model.{field.name}"] = getattr(model_config, field.name)

        flat_dict["training.local_batch_size"] = local_batch_size
        flat_dict["meta.runtime"] = platform
        flat_dict["training.total_tokens"] = total_tokens
        flat_dict["meta.total_devices"] = num_devices
        flat_dict["meta.mesh_mp"] = cfg.training.mp
        flat_dict["meta.mesh_dp"] = cfg.training.dp

        wandb.config.update(flat_dict)

    train_data = np.memmap(cfg.data.train_bin_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg.data.validation_bin_path, dtype=np.uint16, mode="r")

    def get_batch(
        rng: jax.random.KeyArray, split: str, block_size: int, batch_size: int
    ) -> jnp.array:
        """Deterministic random sampler."""

        data = train_data if split == "train" else val_data
        ix = jax.random.randint(
            rng, minval=0, maxval=len(data) - block_size, shape=(batch_size,)
        )
        x = jnp.stack([((data[i : i + block_size]).astype(np.int64)) for i in ix])
        return x

    train_batch_sampler = partial(
        get_batch,
        split="train",
        block_size=cfg.training.train_context,
        batch_size=cfg.training.batch_size,
    )

    val_batch_sampler = partial(
        get_batch,
        split="validation",
        block_size=cfg.training.train_context,
        batch_size=cfg.training.batch_size // 4,
    )

    # quick way to track global step count when resuming a run
    new_steps = 0

    # offset resume_step to avoid re-evaluating an already-existing checkpoint
    if resume_step > 0:
        resume_step += 1

    batch_rng = jax.random.PRNGKey(23)

    for train_it in tqdm(
        range(cfg.training.total_steps), disable=not jax.process_index() == 0
    ):
        if (resume_step + new_steps) > cfg.training.total_steps:
            if jax.process_index() == 0:
                logger.debug("Training has completed.")

            return True

        batch_rng, step_rng = jax.random.split(batch_rng)
        batch = train_batch_sampler(step_rng)

        t0 = time()
        grads, metrics = train_step_tp(params, convert_arr_sharding(batch))
        with mesh:
            params, opt_state = update_opt_step_tp(params, grads, opt_state)
        metrics["train/lr"] = learning_rate_fn(resume_step + new_steps)
        t1 = time()
        metrics["train/step_time"] = t1 - t0

        train_metrics_np = {
            k: np.mean([metric[k] for metric in metrics]) for k in metrics
        }

        validation_metrics = []

        absolute_step = resume_step + new_steps

        train_metrics_np["train/tokens"] = compute_tokens(
            absolute_step,
            max_context=cfg.data.max_shard_context,
            num_host=jax.host_count(),
            batch_per_host=cfg.training.batch_size,
        )

        new_steps += 1

        if (absolute_step) % (cfg.training.evaluation_frequency) == 0:
            for val_it in tqdm(
                range(cfg.training.maximum_evaluation_steps),
                disable=not jax.process_index() == 0,
            ):
                batch_rng, step_rng = jax.random.split(batch_rng)
                val_batch = val_batch_sampler(step_rng)
                metrics = eval_step_tp(params, convert_arr_sharding(val_batch))
                validation_metrics.append(metrics)

            validation_metrics_np = {
                k: np.mean([metrics[k] for metrics in validation_metrics])
                for k in validation_metrics[0]
            }

            if jax.process_index() == 0:
                train_metrics_np.update(validation_metrics_np)
                wandb.log(train_metrics_np)

                save_checkpoint_params(
                    params,
                    absolute_step,
                    workdir=f"{checkpoint_prefix}{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/params",
                )
                save_checkpoint_optimizer(
                    opt_state,
                    absolute_step,
                    workdir=f"{checkpoint_prefix}{cfg.data.bucket_path}/{cfg.data.checkpoint_directory}/opt",
                )

        else:
            if jax.process_index() == 0:
                wandb.log(train_metrics_np)


if __name__ == "__main__":
    main()
