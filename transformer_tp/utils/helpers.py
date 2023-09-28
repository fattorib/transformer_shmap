"""General helper utilities."""
from collections.abc import MutableMapping


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    """Flatten YAML structure for easier logging in Wandb."""
    return dict(_flatten_dict_gen(d, parent_key, sep))


def compute_tokens(
    absolute_step: int, num_host: int, batch_per_host: int, max_context: int
) -> int:
    """Compute GTokens."""
    return (absolute_step * num_host * batch_per_host * max_context) / 1e9
