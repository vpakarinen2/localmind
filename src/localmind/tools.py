from __future__ import annotations

import operator
import math
import ast

from dataclasses import dataclass
from typing import Any, Callable
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from localmind.search import SearxngSearch


ToolHandler = Callable[..., str]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_model_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class WorkspaceViolation(ValueError):
    pass


class SafeCalculator:
    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    _names = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }
    _functions = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    def calculate(self, expression: str) -> str:
        tree = ast.parse(expression, mode="eval")
        value = self._eval(tree.body)
        if isinstance(value, float):
            return format(value, ".12g")
        return str(value)

    def _eval(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._binary_ops:
                raise ValueError("Operator is not allowed")
            return self._binary_ops[op_type](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._unary_ops:
                raise ValueError("Operator is not allowed")
            return self._unary_ops[op_type](self._eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id not in self._names:
                raise ValueError(f"Name is not allowed: {node.id}")
            return self._names[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self._functions:
                raise ValueError("Function is not allowed")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed")
            args = [self._eval(arg) for arg in node.args]
            return self._functions[node.func.id](*args)
        raise ValueError("Expression is not allowed")


class WorkspaceFiles:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {target.relative_to(self.root).as_posix()}"

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            raise WorkspaceViolation("Absolute paths are not allowed")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceViolation("Path escapes the workspace")
        return resolved


class ToolRegistry:
    def __init__(
        self,
        workspace: Path,
        search_enabled: bool = False,
        searxng_url: str | None = None,
    ) -> None:
        calculator = SafeCalculator()
        files = WorkspaceFiles(workspace)
        self._tools: dict[str, ToolSpec] = {
            "calculate": ToolSpec(
                name="calculate",
                description="Evaluate a safe arithmetic expression.",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate.",
                        }
                    },
                    "required": ["expression"],
                },
                handler=calculator.calculate,
            ),
            "current_time": ToolSpec(
                name="current_time",
                description="Return the current local date and time.",
                parameters={"type": "object", "properties": {}},
                handler=current_time,
            ),
            "read_file": ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file from the LocalMind workspace.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative file path."}
                    },
                    "required": ["path"],
                },
                handler=files.read_file,
            ),
            "write_file": ToolSpec(
                name="write_file",
                description="Write a UTF-8 text file in the LocalMind workspace.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative file path."},
                        "content": {"type": "string", "description": "File content to write."},
                    },
                    "required": ["path", "content"],
                },
                handler=files.write_file,
            ),
        }
        if search_enabled:
            search = SearxngSearch(searxng_url or "http://localhost:8080")
            self._tools["web_search"] = ToolSpec(
                name="web_search",
                description=(
                    "Search the web using the configured SearXNG instance. Use this for current, "
                    "recent, external, or source-backed information."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return, from 1 to 10.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                handler=search.search,
            )

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.as_model_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Unknown tool: {name}"
        try:
            return self._tools[name].handler(**arguments)
        except Exception as exc:
            return f"Tool error in {name}: {exc}"


def current_time() -> str:
    try:
        tz = ZoneInfo("Europe/Helsinki")
        now = datetime.now(tz)
    except Exception:
        now = datetime.now().astimezone()
    return now.isoformat(timespec="seconds")
