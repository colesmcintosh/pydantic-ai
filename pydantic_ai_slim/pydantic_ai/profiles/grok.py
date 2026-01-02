from __future__ import annotations as _annotations

from ..builtin_tools import XSearchTool
from . import ModelProfile


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    return ModelProfile(
        supported_builtin_tools=frozenset({XSearchTool}),
    )
