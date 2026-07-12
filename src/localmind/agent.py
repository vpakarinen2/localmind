from __future__ import annotations

import json

from typing import Any

from localmind.tool_parser import parse_tool_calls, strip_tool_calls
from localmind.config import LocalMindConfig
from localmind.tools import ToolRegistry
from localmind.model import ChatModel


SYSTEM_PROMPT = """You are LocalMind, a small local agent.

Answer style:
- Be clear, practical, and concise.
- Prefer plain language over jargon.
- For technical questions, explain the core idea first, then add one useful example when it helps.
- If a term has multiple likely meanings, briefly mention the relevant meanings instead of guessing only one.
- If you are unsure, say what you know and what would need checking.

Tool use:
- Use tools when they improve accuracy or complete the user's request.
- Treat tool results as the source of truth for that step.
- When a tool result is enough, answer directly and do not over-explain.

Time awareness:
- You do not know the real current date or time from memory.
- For questions involving today, now, current date or time, tomorrow, yesterday, deadlines, schedules, elapsed time, or time-sensitive facts, call the current_time tool first.
- Use the current_time result as the source of truth for date and time reasoning.

Web search:
- If web_search is available, use it for current, latest, recent, news, prices, changing APIs, laws, schedules, or facts likely to have changed.
- Do not use web_search for stable general knowledge unless the user asks you to verify or cite sources.
- When using web_search, cite source URLs from the tool result in your final answer.
- Do not invent release dates, prices, version numbers, or availability details. State them only when they appear in the tool result.
- For "latest" product/game/software questions, prefer official publisher, platform, or project sources over wiki or forum results.
- If the user asks for a release date and the first search result does not include one, run one targeted follow-up search that includes the item name and "release date official".
- If search results disagree or do not include the requested date, say what the sources show and note the uncertainty instead of guessing.
"""


class LocalMindAgent:
    def __init__(
        self,
        config: LocalMindConfig,
        model: ChatModel,
        tools: ToolRegistry | None = None,
        max_tool_rounds: int = 4,
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools or ToolRegistry(
            config.workspace,
            search_enabled=config.search_enabled,
            searxng_url=config.searxng_url,
        )
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
        ]

    def ask(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        executed_calls: set[str] = set()
        last_tool_result: dict[str, Any] | None = None

        for _ in range(self.max_tool_rounds):
            response = self.model.generate(
                self.messages,
                self.tools.schemas,
                self.config.enable_thinking,
            )
            tool_calls = parse_tool_calls(response)
            if not tool_calls:
                clean_response = strip_tool_calls(response)
                self.messages.append({"role": "assistant", "content": clean_response})
                return clean_response

            self.messages.append({"role": "assistant", "content": response})
            for call in tool_calls:
                call_key = json.dumps(
                    {"name": call.name, "arguments": call.arguments},
                    sort_keys=True,
                    ensure_ascii=True,
                )
                if call_key in executed_calls:
                    fallback = self._answer_from_tool_result(last_tool_result)
                    self.messages.append({"role": "assistant", "content": fallback})
                    return fallback
                executed_calls.add(call_key)
                result = self.tools.execute(call.name, call.arguments)
                if call.name == "web_search" and (
                    result.startswith("Search error:") or result == "No search results found."
                ):
                    self.messages.append({"role": "assistant", "content": result})
                    return result
                last_tool_result = {
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": result,
                }
                self.messages.append(
                    {
                        "role": "tool",
                        "content": self._format_tool_message(last_tool_result),
                    }
                )

        fallback = self._answer_from_tool_result(last_tool_result)
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback

    def _system_prompt(self) -> str:
        mode = "/think" if self.config.enable_thinking else "/no_think"
        return f"{SYSTEM_PROMPT}\n{mode}"

    def _format_tool_message(self, payload: dict[str, Any]) -> str:
        instruction = (
            "Tool result. Use this result to answer the user now. "
            "Do not call the same tool with the same arguments again unless the result is an error."
        )
        return f"{instruction}\n{json.dumps(payload, ensure_ascii=True)}"

    def _answer_from_tool_result(self, payload: dict[str, Any] | None) -> str:
        if payload is None:
            return "I reached the tool-use limit before I could finish."

        result = str(payload.get("result", ""))
        if result.startswith("Search error:") or result == "No search results found.":
            return result
        if payload.get("name") == "web_search":
            return (
                "I found search results, but I could not turn them into a final answer before "
                f"the tool-use limit. Raw search result:\n{result}"
            )
        return (
            "I used a tool, but I could not turn the result into a final answer before "
            f"the tool-use limit. Tool result:\n{result}"
        )


class StaticChatModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        enable_thinking: bool,
    ) -> str:
        self.calls.append(
            {"messages": list(messages), "tools": list(tools), "enable_thinking": enable_thinking}
        )
        if not self.responses:
            return "No response configured."
        return self.responses.pop(0)
