from __future__ import annotations

import json
import re

from dataclasses import dataclass
from typing import Any


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, dict):
            continue
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def strip_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()
