import pytest
from omegaconf import OmegaConf

from transformer_tp.models import model_getter


@pytest.fixture
def config():
    config = OmegaConf.load("conf/model_config.yaml")
    return config


def test_call_valid(config):
    model_size = "test"
    model = model_getter(model_size, "conf/model_config.yaml")

    assert model.config.embedding_dim == config[model_size].embedding_dim
    assert model.config.vocab_size == config[model_size].vocab_size
    assert model.config.num_attention_heads == config[model_size].num_attention_heads
    assert model.config.block_size == config[model_size].block_size
    assert model.config.num_layers == config[model_size].num_layers
    assert model.config.remat == config[model_size].remat


def test_call_invalid():
    model_size = "Humongous"
    with pytest.raises(AssertionError):
        model_getter(model_size, "conf/model_config.yaml")
