"""GPU Benchmark for shard_map tensor and data parallelism."""
import argparse
from dataclasses import replace
from functools import partial
from time import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf
from tqdm import tqdm

from transformer_tp.models import Transformer, model_getter
from transformer_tp.partitioning import create_opt_spec
from transformer_tp.training import (convert_arr_sharding, create_train_state,
                                     train_step, update_opt_state)

jax.config.update("jax_threefry_partitionable", True)


def parse():
    parser = argparse.ArgumentParser(description="tensor-parallelism benchmark")
    parser.add_argument("--emulation", default=False, action="store_true")
    parser.add_argument("--dp", type=int)
    parser.add_argument("--mp", type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    rng = jax.random.PRNGKey(23)
    args = parse()

    platform = jax.local_devices()[0].platform

    if platform == "tpu":
        raise ValueError(
            "Expected GPU environment, but found TPU environment. Run `benchmark_tpu.py` for TPU benchmarking."
        )

    num_iter = 10

    if args.emulation:
        print(f"Emulating {jax.local_device_count()} devices")
        accum_steps = 16
        batch_size = 16
        context = 32
        model_name = "test"

    else:
        batch_size = 8
        accum_steps = 8
        context = 1024
        model_name = "ws_1xGPU"

    mesh = Mesh(np.array(jax.devices()).reshape(args.dp, args.mp), ("dp", "mp"))

    if jax.process_index() == 0:
        print(mesh)

    batch_spec = P("dp", None)
    no_shard = P(None)

    model, model_config = model_getter(model_name, return_cfg=True)

    replaced_args = {
        "num_shard": mesh.shape["mp"],
        "tp_comms": True if mesh.shape["mp"] > 1 else False,
    }

    sharded_config = replace(model_config, **replaced_args)
    model_shard = Transformer(sharded_config)

    batch_tok = jax.random.randint(rng, shape=(1, context), maxval=50257, minval=0)

    param_abstract, tx = create_train_state(
        rng,
        3e-4,
        weight_decay=0.0,
        model=model,
    )

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(param_abstract))

    if jax.process_index() == 0:
        print(f"Distributing model with {param_count} params across {args.mp} devices.")

    param_spec = nn.get_partition_spec(param_abstract)

    if args.mp > 1:
        with mesh:
            # do actual layer init wrapping with pjit
            batch = jnp.ones((1, context), dtype=jnp.int32)
            params = pjit(
                model.init,
                out_axis_resources=param_spec,
            )(rng, batch)
        grad_spec = param_spec

    else:
        param_spec = no_shard
        grad_spec = no_shard
        with mesh:
            batch = jnp.ones((1, context), dtype=jnp.int32)
            params = pjit(
                model.init,
                out_axis_resources=param_spec,
            )(rng, batch)

    configs = OmegaConf.load("conf/model_config.yaml")
    model_info = configs[model_name]

    layers, heads, h_dim, context = (
        model_info.num_layers,
        model_info.num_attention_heads,
        model_info.embedding_dim // model_info.num_attention_heads,
        context,
    )

    train_step_tp = jax.jit(
        shard_map(
            partial(train_step, model=model_shard, accum_steps=accum_steps),
            in_specs=(param_spec, batch_spec),
            out_specs=(grad_spec, P(None)),
            mesh=mesh,
            check_rep=False,
        )
    )

    opt_state_shapes = jax.eval_shape(tx.init, param_abstract)
    opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

    with mesh:
        opt_state = pjit(
            tx.init,
            in_axis_resources=(param_spec,),
            out_axis_resources=opt_state_spec,
        )(params)

    convert_batch = jax.jit(
        partial(convert_arr_sharding, mesh=mesh, batch_spec=batch_spec)
    )

    rng, dropout_rng = jax.random.split(rng, 2)

    init_batch = jax.random.randint(
        dropout_rng,
        shape=(batch_size, context),
        minval=0,
        maxval=model.config.vocab_size,
    )

    grads, metrics = train_step_tp(params, convert_batch(init_batch))

    with mesh:
        update_opt_step_tp = pjit(
            partial(update_opt_state, optimizer=tx, tp_spec=grad_spec),
            in_axis_resources=(param_spec, grad_spec, opt_state_spec),
            out_axis_resources=(param_spec, opt_state_spec),
            donate_argnums=0,
        )
    with mesh:
        params, opt_state = update_opt_step_tp(params, grads, opt_state)

    rng_batch = jax.random.PRNGKey(0)
    start = time()
    for i in tqdm(range(num_iter), disable=not jax.process_index() == 0):
        batch = jax.random.randint(
            rng_batch,
            shape=(batch_size, context),
            minval=0,
            maxval=model.config.vocab_size,
        )
        grads, metrics = train_step_tp(params, convert_batch(batch))

        with mesh:
            params, opt_state = update_opt_step_tp(params, grads, opt_state)

    jnp.zeros((10, 10)).block_until_ready()
    end = time()
    total_time = end - start
    print(metrics)

    print(
        f"TP Step - Per-Host BS {batch_size} - accum steps {accum_steps} - Num Executions {num_iter}"
    )
    print(f"Model Size: {model_name}")
    print(f"Total Time: {total_time:.4f}s")

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(param_abstract))

    flops_per_token = 6 * param_count + (12 * layers * heads * h_dim * context)
    flops_per_fwdbwd = flops_per_token * context
    flops_per_iter = flops_per_fwdbwd * batch_size

    total_flops = flops_per_iter * num_iter

    effective_tflops = total_flops / (total_time)

    perf_log = {}

    perf_log["batch_size"] = batch_size
    perf_log["model_name"] = model_name
    perf_log["accum_steps"] = accum_steps
    perf_log["num_iter"] = num_iter
    perf_log["dp"] = args.dp
    perf_log["mp"] = args.mp

    perf_log["total_time"] = total_time
    if jax.process_index() == 0:
        print(f"Param Count: {param_count}")
        print(f"Effective TFLOPS: {total_flops / (total_time)/1e12:.06}")
        print(perf_log)
