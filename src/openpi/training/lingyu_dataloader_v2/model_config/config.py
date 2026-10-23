"""Unified config entry: specify which embodied-AI model is in use and expose functions to read its configs.

Other programs always get configs via load_action_chunk_length() and never import a model's config file directly,
so switching models only requires changing that single import line at the top of this file.

Every model config file must provide the same-named constants:
    ACTION_CHUNK_LENGTH  one sample's action sequence length, i.e. the length of [current action, the next n-1 actions]
    STATE_CONCAT / ACTION_CONCAT
                         ((mcap topic, field name), ...), i.e. the concat order of the state/action vectors
"""
from __future__ import annotations

from openpi.training.lingyu_dataloader_v2.model_config import pi0  # Switching models only changes this line


MODEL = "pi0"  # TODO: pick which embodied-AI model to use


#####
# Load one sample's action sequence length
#####

def load_action_chunk_length() -> int:
    """Return the current model's action sequence length per sample (including the current action)."""
    if MODEL == "pi0":
        return pi0.ACTION_CHUNK_LENGTH
    # TODO: elif other models' constant configs
    else:
        raise ValueError(
            f"Cannot find the relevant model configuration of {MODEL!r} "
            f"in lingyu_dataloader_v2/model_config."
        )


#####
# Load the concat order of the state / action vectors
#####

def load_state_and_action_concat() -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Return the current model's (STATE_CONCAT, ACTION_CONCAT), elements being (mcap topic, field name)."""
    if MODEL == "pi0":
        return pi0.STATE_CONCAT, pi0.ACTION_CONCAT
    # TODO: elif other models' constant configs
    else:
        raise ValueError(
            f"Cannot find the relevant model configuration of {MODEL!r} "
            f"in lingyu_dataloader_v2/model_config."
        )
