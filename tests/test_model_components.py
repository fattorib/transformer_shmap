import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from optax import softmax_cross_entropy

from transformer_tp.models.transformer import (CausalAttention, MLPBlock,
                                               Transformer, TransformerBlock,
                                               TransformerConfig)


@pytest.fixture
def setup_config():
    init_rng, rng = random.split(random.PRNGKey(0))
    config = TransformerConfig(
        vocab_size=256,
        embedding_dim=128,
        block_size=512,
        num_attention_heads=8,
        num_layers=10,
        tp_comms=False,
        remat=False,
    )

    init_batch = jnp.ones(
        (1, config.block_size, config.embedding_dim), dtype=jnp.float32
    )
    test_batch = random.normal(rng, (2, config.block_size, config.embedding_dim))

    return (init_rng, config, init_batch, test_batch)


@pytest.fixture
def setup_transformer_config():
    init_rng, rng = random.split(random.PRNGKey(0))
    config = TransformerConfig(
        vocab_size=256,
        embedding_dim=128,
        block_size=512,
        num_attention_heads=8,
        num_layers=10,
        tp_comms=False,
        remat=False,
    )

    init_batch = jnp.ones((1, config.block_size), dtype=jnp.int32)
    test_batch = jax.random.randint(
        rng, (2, config.block_size), maxval=config.vocab_size, minval=0
    )

    return (init_rng, config, init_batch, test_batch)


def test_MLP_create(setup_config):
    init_rng, config, init_batch, test_batch = setup_config

    mlp = MLPBlock(config)
    params = mlp.init(init_rng, init_batch)


def test_MLP_fwd(setup_config):
    init_rng, config, init_batch, test_batch = setup_config

    mlp = MLPBlock(config)
    params = mlp.init(init_rng, init_batch)

    out = mlp.apply(
        {"params": params["params"]},
        test_batch,
    )


def test_MLP_fwd_shape(setup_config):
    init_rng, config, init_batch, test_batch = setup_config

    mlp = MLPBlock(config)
    params = mlp.init(init_rng, init_batch)

    out = mlp.apply(
        {"params": params["params"]},
        test_batch,
    )

    assert out.shape == test_batch.shape


def test_attn_create(setup_config):
    init_rng, config, init_batch, test_batch = setup_config
    attn = CausalAttention(config)
    params = attn.init(init_rng, init_batch)


def test_attn_fwd(setup_config):
    init_rng, config, init_batch, test_batch = setup_config
    attn = CausalAttention(config)
    params = attn.init(init_rng, init_batch)
    out = attn.apply(
        {"params": params["params"]},
        test_batch,
    )


def test_attn_fwd_shape(setup_config):
    init_rng, config, init_batch, test_batch = setup_config
    attn = CausalAttention(config)
    params = attn.init(init_rng, init_batch)
    out = attn.apply(
        {"params": params["params"]},
        test_batch,
    )
    assert out.shape == test_batch.shape


def test_transformer_create(setup_config):
    init_rng, config, init_batch, test_batch = setup_config
    transformer_block = TransformerBlock(config)
    params = transformer_block.init(init_rng, init_batch)


def test_transformer_block_fwd(setup_config):
    init_rng, config, init_batch, test_batch = setup_config
    transformer_block = TransformerBlock(config)
    params = transformer_block.init(init_rng, init_batch)
    out = transformer_block.apply(
        {"params": params["params"]},
        test_batch,
    )


def test_transformer_block_fwd_shape(setup_config):
    init_rng, config, init_batch, test_batch = setup_config
    transformer_block = TransformerBlock(config)
    params = transformer_block.init(init_rng, init_batch)
    out = transformer_block.apply(
        {"params": params["params"]},
        test_batch,
    )

    assert out.shape == test_batch.shape


def test_transformer_create(setup_transformer_config):
    init_rng, config, init_batch, test_batch = setup_transformer_config
    transformer = Transformer(config)
    params = transformer.init(init_rng, init_batch)


def test_transformer_fwd(setup_transformer_config):
    init_rng, config, init_batch, test_batch = setup_transformer_config
    transformer = Transformer(config)
    params = transformer.init(init_rng, init_batch)
    out = transformer.apply(
        {"params": params["params"]},
        test_batch,
    )


def test_transformer_fwd_shape(setup_transformer_config):
    init_rng, config, init_batch, test_batch = setup_transformer_config
    transformer = Transformer(config)
    params = transformer.init(init_rng, init_batch.astype(jnp.int32))
    out = transformer.apply(
        {"params": params["params"]},
        test_batch,
    )
    assert out.shape == (test_batch.shape[0], test_batch.shape[1], config.vocab_size)


def test_transformer_loss(setup_transformer_config):
    init_rng, config, init_batch, test_batch = setup_transformer_config
    transformer = Transformer(config)
    params = transformer.init(init_rng, init_batch.astype(jnp.int32))
    logits, loss = transformer.apply(
        {"params": params["params"]}, test_batch, labels=test_batch
    )

    labels_shifted = test_batch[..., 1:].reshape(-1)
    logits_shifted = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    oh_labels_shifted = jax.nn.one_hot(labels_shifted, num_classes=config.vocab_size)
    external_loss = softmax_cross_entropy(logits_shifted, oh_labels_shifted)

    assert jnp.mean(loss) == jnp.mean(external_loss)
