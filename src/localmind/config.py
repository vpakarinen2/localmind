from __future__ import annotations

import os

from typing import Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_WORKSPACE = Path("workspace")
DEFAULT_SEARXNG_URL = "http://localhost:8080"
DeviceMode = Literal["auto", "cpu", "cuda"]
PromptFormat = Literal["chat", "alpaca"]


class LocalMindConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = DEFAULT_MODEL
    lora_model: str | None = None
    prompt_format: PromptFormat = "chat"
    workspace: Path = Field(default_factory=lambda: DEFAULT_WORKSPACE, validate_default=True)
    memory_enabled: bool = False
    enable_thinking: bool = False
    max_new_tokens: int = Field(default=1_024, ge=1)
    thinking_max_new_tokens: int = Field(default=4_096, ge=1)
    coding_mode: bool = False
    direct_mode: bool = False
    device: DeviceMode = "auto"
    search_enabled: bool = False
    searxng_url: str | None = Field(default=None, validate_default=True)

    @field_validator("lora_model")
    @classmethod
    def normalize_lora_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("workspace")
    @classmethod
    def normalize_workspace(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("searxng_url")
    @classmethod
    def default_searxng_url(cls, value: str | None) -> str:
        url = value or os.getenv("LOCALMIND_SEARXNG_URL") or DEFAULT_SEARXNG_URL
        return url.rstrip("/")
