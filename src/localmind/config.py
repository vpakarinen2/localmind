from __future__ import annotations

import os

from typing import Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_WORKSPACE = Path("workspace")
DEFAULT_SEARXNG_URL = "http://localhost:8080"
DeviceMode = Literal["auto", "cpu", "cuda"]


class LocalMindConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = DEFAULT_MODEL
    workspace: Path = Field(default_factory=lambda: DEFAULT_WORKSPACE, validate_default=True)
    enable_thinking: bool = False
    device: DeviceMode = "auto"
    search_enabled: bool = False
    searxng_url: str | None = Field(default=None, validate_default=True)

    @field_validator("workspace")
    @classmethod
    def normalize_workspace(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("searxng_url")
    @classmethod
    def default_searxng_url(cls, value: str | None) -> str:
        url = value or os.getenv("LOCALMIND_SEARXNG_URL") or DEFAULT_SEARXNG_URL
        return url.rstrip("/")
