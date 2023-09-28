"""Test the core functionality of the training loop."""

import os
from dataclasses import replace
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from transformer_tp.models import Transformer, model_getter
from transformer_tp.partitioning import create_opt_spec
from transformer_tp.training import (create_train_state, train_step,
                                     update_opt_state)

# default github actions runner has 2 cores
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture
def train_config():
    return {
        "num_iter": 10,
        "accum_steps": 2,
        "batch_size": 8,
        "num_context": 32,
        "model": "test",
    }


@pytest.fixture
def dp_mesh():
    mesh = Mesh(
        np.array(jax.devices()).reshape(jax.local_device_count(), 1), ("dp", "mp")
    )
    return mesh


@pytest.fixture
def dp_shard_specs():
    return {
        "batch_spec": P("dp", None),
        "grad_spec": P(None),
        "param_spec": P(None),
        "null_spec": P(None),
    }


@pytest.fixture
def setup_dp_params(train_config, dp_mesh, dp_shard_specs):
    """Setup params for DP training."""
    rng = jax.random.PRNGKey(0)

    model = model_getter(train_config["model"])
    mesh = dp_mesh

    with mesh:
        batch = jnp.ones((1, train_config["num_context"]), dtype=jnp.int32)
        params = pjit(
            model.init,
            in_axis_resources=(
                dp_shard_specs["null_spec"],
                dp_shard_specs["null_spec"],
            ),
            out_axis_resources=dp_shard_specs["param_spec"],
        )(rng, batch)

    param_abstract, tx = create_train_state(
        rng,
        3e-4,
        weight_decay=0.0,
        model=model,
    )

    opt_state_shapes = jax.eval_shape(tx.init, param_abstract)
    opt_state_spec = create_opt_spec(dp_shard_specs["null_spec"], opt_state_shapes)

    with mesh:
        opt_state = pjit(
            tx.init,
            in_axis_resources=(dp_shard_specs["null_spec"],),
            out_axis_resources=opt_state_spec,
        )(params)

    return {
        "params": params,
        "shard_specs": dp_shard_specs,
        "model": model,
        "config": train_config,
        "mesh": mesh,
        "opt_state_spec": opt_state_spec,
        "opt_state": opt_state,
        "tx": tx,
    }


@pytest.fixture
def tp_mesh():
    mesh = Mesh(
        np.array(jax.devices()).reshape(1, jax.local_device_count()), ("dp", "mp")
    )
    return mesh


@pytest.fixture
def tp_shard_specs():
    return {"batch_spec": P("dp", None), "null_spec": P(None)}


@pytest.fixture
def setup_tp_params(train_config, tp_mesh, tp_shard_specs):
    """Setup params for TP training."""
    rng = jax.random.PRNGKey(0)

    model, model_config = model_getter(train_config["model"], return_cfg=True)
    mesh = tp_mesh

    replaced_args = {
        "num_shard": mesh.shape["mp"],
        "tp_comms": True if mesh.shape["mp"] > 1 else False,
    }

    sharded_config = replace(model_config, **replaced_args)
    model_shard = Transformer(sharded_config)
    batch = jnp.ones(shape=(1, train_config["num_context"]), dtype=jnp.int32)

    param_abstract, tx = create_train_state(
        rng,
        3e-4,
        weight_decay=0.0,
        model=model,
    )

    param_spec = nn.get_partition_spec(param_abstract)
    grad_spec = param_spec

    with mesh:
        params = pjit(
            model.init,
            in_axis_resources=(
                tp_shard_specs["null_spec"],
                tp_shard_specs["null_spec"],
            ),
            out_axis_resources=param_spec,
        )(rng, batch)

    tp_shard_specs["param_spec"] = param_spec
    tp_shard_specs["grad_spec"] = grad_spec

    opt_state_shapes = jax.eval_shape(tx.init, param_abstract)
    opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

    with mesh:
        opt_state = pjit(
            tx.init,
            in_axis_resources=(param_spec,),
            out_axis_resources=opt_state_spec,
        )(params)

    return {
        "params": params,
        "shard_specs": tp_shard_specs,
        "model": model_shard,
        "config": train_config,
        "mesh": mesh,
        "opt_state_spec": opt_state_spec,
        "opt_state": opt_state,
        "tx": tx,
    }


@pytest.fixture
def hybrid_mesh():
    mesh = Mesh(
        np.array(jax.devices()).reshape(
            jax.local_device_count() // 2, jax.local_device_count() // 2
        ),
        ("dp", "mp"),
    )
    return mesh


@pytest.fixture
def hybrid_shard_specs():
    return {"batch_spec": P("dp", None), "null_spec": P(None)}


@pytest.fixture
def setup_hybrid_params(train_config, hybrid_mesh, hybrid_shard_specs):
    """Setup params for DP + TP training."""
    rng = jax.random.PRNGKey(0)

    model, model_config = model_getter(train_config["model"], return_cfg=True)
    mesh = hybrid_mesh

    replaced_args = {
        "num_shard": mesh.shape["mp"],
        "tp_comms": True if mesh.shape["mp"] > 1 else False,
    }

    sharded_config = replace(model_config, **replaced_args)
    model_shard = Transformer(sharded_config)
    batch = jnp.ones(shape=(1, train_config["num_context"]), dtype=jnp.int32)

    param_abstract, tx = create_train_state(
        rng,
        3e-4,
        weight_decay=0.0,
        model=model,
    )

    param_spec = nn.get_partition_spec(param_abstract)
    grad_spec = param_spec

    with mesh:
        params = pjit(
            model.init,
            in_axis_resources=(
                hybrid_shard_specs["null_spec"],
                hybrid_shard_specs["null_spec"],
            ),
            out_axis_resources=param_spec,
        )(rng, batch)

    hybrid_shard_specs["param_spec"] = param_spec
    hybrid_shard_specs["grad_spec"] = grad_spec

    opt_state_shapes = jax.eval_shape(tx.init, param_abstract)
    opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)

    with mesh:
        opt_state = pjit(
            tx.init,
            in_axis_resources=(param_spec,),
            out_axis_resources=opt_state_spec,
        )(params)

    return {
        "params": params,
        "shard_specs": hybrid_shard_specs,
        "model": model_shard,
        "config": train_config,
        "mesh": mesh,
        "opt_state_spec": opt_state_spec,
        "opt_state": opt_state,
        "tx": tx,
    }


@pytest.mark.parametrize(
    "setup_params",
    ["setup_tp_params", "setup_dp_params", "setup_hybrid_params"],
)
def test_fwd_no_opt(setup_params, request):
    """Test call to train_step without optimizer update."""
    setup_params = request.getfixturevalue(setup_params)
    params = setup_params["params"]
    shard_specs_dict = setup_params["shard_specs"]
    model = setup_params["model"]
    config = setup_params["config"]
    mesh = setup_params["mesh"]

    param_spec, batch_spec, grad_spec = (
        shard_specs_dict["param_spec"],
        shard_specs_dict["batch_spec"],
        shard_specs_dict["grad_spec"],
    )
    accum_steps = config["accum_steps"]

    train_step_partial = jax.jit(
        shard_map(
            partial(train_step, model=model, accum_steps=accum_steps),
            in_specs=(param_spec, batch_spec),
            out_specs=(grad_spec, P(None)),
            mesh=mesh,
            check_rep=False,
        )
    )
    batch = jax.numpy.ones(
        shape=(config["batch_size"], config["num_context"]), dtype=jax.numpy.int32
    )

    grads, metrics = train_step_partial(params, batch)


@pytest.mark.parametrize(
    "setup_params",
    ["setup_tp_params", "setup_dp_params", "setup_hybrid_params"],
)
def test_fwd_opt(setup_params, request):
    """Test call to train_step without optimizer update."""
    setup_params = request.getfixturevalue(setup_params)
    params = setup_params["params"]
    shard_specs_dict = setup_params["shard_specs"]
    model = setup_params["model"]
    config = setup_params["config"]
    mesh = setup_params["mesh"]
    tx = setup_params["tx"]
    opt_state_spec = setup_params["opt_state_spec"]
    opt_state = setup_params["opt_state"]

    param_spec, batch_spec, grad_spec = (
        shard_specs_dict["param_spec"],
        shard_specs_dict["batch_spec"],
        shard_specs_dict["grad_spec"],
    )
    accum_steps = config["accum_steps"]

    train_step_partial = jax.jit(
        shard_map(
            partial(train_step, model=model, accum_steps=accum_steps),
            in_specs=(param_spec, batch_spec),
            out_specs=(grad_spec, P(None)),
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

    batch = jax.numpy.ones(
        shape=(config["batch_size"], config["num_context"]), dtype=jax.numpy.int32
    )

    grads, metrics = train_step_partial(params, batch)

    with mesh:
        params, opt_state = update_opt_step_tp(params, grads, opt_state)
