import math
from dataclasses import dataclass
from typing import List, Tuple, Union

import flax.linen as nn
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf
from optax import softmax_cross_entropy

from transformer_tp.models.replicated_utils import f_psum, g_psum

# 'default' precision blows up mem on GPU ¯\_(ツ)_/¯
DEFAULT_PRECISION = jax.lax.Precision("default")


def get_slopes(n: int) -> List:
    """Create ALiBi head slopes."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def create_mask(seq_len_k, slopes):
    """Create ALiBi distance mask."""

    a = -jnp.tril(
        jnp.tile(jnp.arange(seq_len_k).reshape(seq_len_k, 1), (1, seq_len_k))
        + jnp.arange(0, -seq_len_k, step=-1)
    )

    a = a * (slopes.reshape(slopes.shape[0], 1, 1))

    alibi_mask = a[:, seq_len_k - 1, :].reshape(a.shape[0], 1, a.shape[2])

    return alibi_mask


@dataclass
class TransformerConfig:
    vocab_size: int
    embedding_dim: int
    block_size: int
    num_attention_heads: int
    num_layers: int
    remat: bool

    # extra tensor-parallel arguments, disabled by default
    num_shard: int = 1
    tp_comms: bool = False


class MLPBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        x = nn.Dense(
            features=(4 * self.config.embedding_dim) // self.config.num_shard,
            name="fc_in",
            kernel_init=nn.with_partitioning(
                initializers.normal(
                    stddev=jnp.sqrt(2 / (5 * self.config.embedding_dim))
                ),
                P(None, "mp"),
            ),
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(x)
        x = nn.gelu(x)
        out = nn.Dense(
            features=self.config.embedding_dim,
            name="fc_residual",
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(
                    stddev=(
                        2
                        / (self.config.num_layers * jnp.sqrt(self.config.embedding_dim))
                    )
                ),
                P("mp", None),
            ),
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(x)

        return out


class CausalAttention(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.config.heads_per_shard = (
            self.config.num_attention_heads // self.config.num_shard
        )
        self.config.slopes = jnp.array(get_slopes(self.config.num_attention_heads))
        self.config.mask = jnp.tril(
            jnp.ones((self.config.block_size, self.config.block_size), dtype=jnp.int8)
        ).reshape(1, 1, self.config.block_size, self.config.block_size)
        self.config.alibi_mask = create_mask(self.config.block_size, self.config.slopes)

        if self.config.num_shard > 1:
            # add shard dimension so each model shard pulls the correct slopes
            self.config.alibi_mask = self.config.alibi_mask.reshape(
                self.config.num_shard, self.config.heads_per_shard, 1, -1
            )

        self.config.head_dim = (
            self.config.embedding_dim // self.config.num_attention_heads
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
    ) -> jnp.array:
        key = nn.Dense(
            name="key_proj",
            features=self.config.embedding_dim // self.config.num_shard,
            kernel_init=nn.with_partitioning(
                initializers.normal(
                    stddev=jnp.sqrt(2 / (5 * self.config.embedding_dim))
                ),
                P(None, "mp"),
            ),
            bias_init=initializers.zeros,
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(x)

        value = nn.Dense(
            name="value_proj",
            features=self.config.embedding_dim // self.config.num_shard,
            kernel_init=nn.with_partitioning(
                initializers.normal(
                    stddev=jnp.sqrt(2 / (5 * self.config.embedding_dim))
                ),
                P(None, "mp"),
            ),
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(x)

        query = nn.Dense(
            name="query_proj",
            features=self.config.embedding_dim // self.config.num_shard,
            kernel_init=nn.with_partitioning(
                initializers.normal(
                    stddev=jnp.sqrt(2 / (5 * self.config.embedding_dim))
                ),
                P(None, "mp"),
            ),
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(x)

        key = rearrange(
            key,
            "b sq (nh hd) -> b sq nh hd",
            nh=self.config.heads_per_shard,
            hd=(self.config.embedding_dim // self.config.num_attention_heads),
        )
        query = rearrange(
            query,
            "b sq (nh hd) -> b sq nh hd",
            nh=self.config.heads_per_shard,
            hd=(self.config.embedding_dim // self.config.num_attention_heads),
        )
        value = rearrange(
            value,
            "b sq (nh hd) -> b sq nh hd",
            nh=self.config.heads_per_shard,
            hd=(self.config.embedding_dim // self.config.num_attention_heads),
        )

        attn_full = jnp.einsum(
            "...qhd,...khd->...hqk", query, key, precision=DEFAULT_PRECISION
        )

        attn_full /= jnp.sqrt(self.config.head_dim)

        if self.config.tp_comms:
            mp_index = jax.lax.axis_index("mp")
            attn_full = attn_full + self.config.alibi_mask[mp_index]
        else:
            attn_full = attn_full + self.config.alibi_mask

        masked_attn = jnp.where(
            self.config.mask, attn_full.astype(jnp.float32), jnp.finfo(jnp.float32).min
        )

        attn_scores = nn.softmax(masked_attn, axis=-1)

        attn_out = jnp.einsum(
            "...hqk,...khd->...qhd", attn_scores, value, precision=DEFAULT_PRECISION
        )

        attn_out = rearrange(attn_out, "b sq nh hd -> b sq (nh hd)")

        out = nn.Dense(
            name="residual_out",
            features=self.config.embedding_dim,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(
                    stddev=(
                        2
                        / (self.config.num_layers * jnp.sqrt(self.config.embedding_dim))
                    )
                ),
                P("mp", None),
            ),
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(attn_out)

        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
    ) -> jnp.array:
        ln_x = nn.LayerNorm(
            use_bias=False,
            scale_init=nn.with_partitioning(jax.nn.initializers.ones, P(None)),
        )(x)

        if self.config.tp_comms:
            ln_x = f_psum(ln_x)

        attn_out = CausalAttention(self.config)(
            ln_x,
        )

        mlp_out = MLPBlock(self.config)(
            ln_x,
        )

        if self.config.tp_comms:
            out = g_psum(mlp_out + attn_out)
        else:
            out = mlp_out + attn_out

        x = x + out

        return x


class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.array,
        labels: jnp.array = None,
    ) -> Union[jnp.array, Tuple[jnp.array, jnp.array]]:
        if self.config.tp_comms:
            dim_per_shard = self.config.vocab_size // self.config.num_shard
            shard_start_index = jax.lax.axis_index("mp") * dim_per_shard
            input_onehot = jax.nn.one_hot(x - shard_start_index, dim_per_shard)
        else:
            input_onehot = jax.nn.one_hot(x, self.config.vocab_size)
        out = nn.Dense(
            name="wte",
            features=self.config.embedding_dim,
            kernel_init=nn.with_partitioning(
                initializers.normal(
                    stddev=jnp.sqrt(2 / (5 * self.config.embedding_dim))
                ),
                P("mp", None),
            ),
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(input_onehot)

        if self.config.tp_comms:
            out = g_psum(out)

        if self.config.remat:
            layer = nn.checkpoint(TransformerBlock)
        else:
            layer = TransformerBlock

        for _ in range(self.config.num_layers):
            out = layer(self.config)(out)

        out = nn.LayerNorm(
            use_bias=False,
            scale_init=nn.with_partitioning(jax.nn.initializers.ones, P(None)),
        )(out)

        if self.config.tp_comms:
            out = f_psum(out)

        logits = nn.Dense(
            name="logits_untied",
            features=self.config.vocab_size // self.config.num_shard,
            kernel_init=nn.with_partitioning(
                initializers.normal(
                    stddev=jnp.sqrt(2 / (5 * self.config.embedding_dim))
                ),
                P(None, "mp"),
            ),
            use_bias=False,
            precision=DEFAULT_PRECISION,
        )(out)

        if self.config.tp_comms:
            if labels is None:
                logits = jax.lax.all_gather(logits, axis_name="mp")
                logits = jnp.concatenate(logits, axis=-1)
                return logits

            else:
                # each mp shard computes local loss and then we all-gather these to reduce
                # total comm volume
                # loss calculation from mesh-transformer-jax:
                # https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py#L569
                labels = labels[..., 1:]
                logits = logits[..., :-1, :].astype(jnp.float32)
                dim_per_shard = self.config.vocab_size // self.config.num_shard
                shard_start_index = jax.lax.axis_index("mp") * dim_per_shard
                global_max = jax.lax.pmax(
                    jax.lax.stop_gradient(logits.max(-1, keepdims=True)), "mp"
                )
                logits -= jax.lax.stop_gradient(global_max)

                gt_onehot = jax.nn.one_hot(labels - shard_start_index, dim_per_shard)
                predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
                predicted_logits = g_psum(predicted_logits)

                exp_logits = jnp.exp(logits)

                sum_exp_logits = exp_logits.sum(axis=-1)
                sum_exp_logits = g_psum(sum_exp_logits)

                loss = jnp.log(sum_exp_logits) - predicted_logits

                return logits, loss

        else:
            if labels is None:
                return logits
            else:
                labels_shifted = labels[..., 1:]
                logits_shifted = logits[..., :-1, :]

                oh_labels_shifted = jax.nn.one_hot(
                    labels_shifted, num_classes=self.config.vocab_size
                )

                loss = softmax_cross_entropy(logits_shifted, oh_labels_shifted)

                return logits, loss


def model_getter(
    model_size,
    config_path="conf/model_config.yaml",
    return_cfg=False,
) -> nn.Module:
    """Load a model configuration from YAML file and return the model class."""

    configs = OmegaConf.load(config_path)
    assert model_size in list(configs.keys()), "Invalid model name provided"
    if return_cfg:
        return Transformer(TransformerConfig(**configs[model_size])), TransformerConfig(
            **configs[model_size]
        )
    else:
        return Transformer(TransformerConfig(**configs[model_size]))
