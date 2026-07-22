from __future__ import annotations

import json

from typing import Any

from localmind.config import LocalMindConfig
from localmind.tools import ToolRegistry
from localmind.model import ChatModel

from localmind.tool_parser import parse_tool_calls, strip_tool_calls
from localmind.prompts import build_system_prompt

from localmind.response_cleanup import (
    decode_literal_unicode_escapes,
    looks_like_generic_refusal,
    normalize_requested_paragraphs,
    strip_thinking,
)
from localmind.routing import (
    response_format_instruction,
    search_query,
    search_result_limit,
    select_tool_schemas,
    should_presearch,
)
from localmind.search_context import (
    append_search_sources,
    clean_leaked_tool_answer,
    format_tool_message,
    limit_tool_result,
    looks_like_search_instruction_echo,
    looks_like_source_dump,
    looks_like_tool_message_leak,
    prepare_search_result,
    strip_search_prompt_transcript,
)


class LocalMindAgent:
    def __init__(
        self,
        config: LocalMindConfig,
        model: ChatModel,
        tools: ToolRegistry | None = None,
        max_tool_rounds: int = 4,
        max_message_characters: int = 60_000,
        max_tool_result_characters: int = 20_000,
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools or ToolRegistry(
            config.workspace,
            search_enabled=config.search_enabled,
            searxng_url=config.searxng_url,
        )
        self.max_tool_rounds = max_tool_rounds
        system_prompt = build_system_prompt(config)
        self.max_message_characters = max(
            max_message_characters, len(system_prompt) + 1
        )
        self.max_tool_result_characters = max(2, max_tool_result_characters)
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

    def ask(self, prompt: str) -> str:
        history_before_turn = list(self.messages)
        self.messages.append({"role": "user", "content": prompt})
        turn_tools = select_tool_schemas(prompt, self.tools.schemas)
        format_instruction = response_format_instruction(prompt)
        allowed_tool_names = {
            str(schema.get("name")) for schema in turn_tools if schema.get("name")
        }
        executed_calls: set[str] = set()
        last_tool_result: dict[str, Any] | None = None
        search_sources: list[dict[str, Any]] = []
        search_source_numbers: dict[str, int] = {}

        if should_presearch(prompt, search_available=self._has_tool("web_search")):
            arguments = {
                "query": search_query(prompt),
                "max_results": search_result_limit(prompt),
            }
            result = self.tools.execute("web_search", arguments)
            if self._is_search_error(result):
                return self._finish_turn(
                    history_before_turn, result, discard_search_turn=True
                )
            result = prepare_search_result(
                result,
                search_sources,
                search_source_numbers,
                self.max_tool_result_characters,
            )
            last_tool_result = {
                "name": "web_search",
                "arguments": arguments,
                "result": result,
            }
            executed_calls.add(self._tool_call_key("web_search", arguments))
            self.messages.append(
                {
                    "role": "tool",
                    "content": format_tool_message(
                        last_tool_result, format_instruction
                    ),
                }
            )

        for _ in range(self.max_tool_rounds):
            self._compact_messages()
            response = self.model.generate(
                self.messages,
                turn_tools,
                self.config.enable_thinking,
            )
            tool_calls = parse_tool_calls(response)
            if not tool_calls:
                clean_response = self._clean_generated_answer(response, last_tool_result)
                if not clean_response:
                    clean_response = self._retry_empty_response(
                        response, last_tool_result, turn_tools
                    ) or self._empty_response_fallback(last_tool_result)
                if self.config.direct_mode and looks_like_generic_refusal(clean_response):
                    direct_response = self._retry_direct_refusal(
                        clean_response, last_tool_result, turn_tools
                    )
                    if direct_response is not None:
                        clean_response = direct_response
                if search_sources and self._is_malformed_search_answer(clean_response):
                    revised_response = self._revise_malformed_search_answer(
                        clean_response,
                        last_tool_result,
                        turn_tools,
                        format_instruction,
                    )
                    clean_response = revised_response or (
                        "I found web results, but the model could not produce a reliable "
                        "synthesized answer from them."
                    )
                if looks_like_tool_message_leak(clean_response):
                    clean_response = clean_leaked_tool_answer(last_tool_result)
                clean_response = normalize_requested_paragraphs(
                    clean_response, prompt
                )
                final_response = append_search_sources(clean_response, search_sources)
                return self._finish_turn(
                    history_before_turn,
                    final_response,
                    discard_search_turn=bool(search_sources),
                )

            self.messages.append({"role": "assistant", "content": response})
            for call in tool_calls:
                call_key = self._tool_call_key(call.name, call.arguments)
                if call_key in executed_calls:
                    return self._finish_turn(
                        history_before_turn,
                        self._answer_from_tool_result(last_tool_result),
                        discard_search_turn=bool(search_sources),
                    )
                executed_calls.add(call_key)
                if call.name not in allowed_tool_names:
                    result = (
                        f"Tool error: {call.name} is not enabled for this request. "
                        "Do not call it; answer using the available information."
                    )
                else:
                    result = self.tools.execute(call.name, call.arguments)
                if call.name == "web_search" and self._is_search_error(result):
                    return self._finish_turn(
                        history_before_turn, result, discard_search_turn=True
                    )
                if call.name == "web_search":
                    result = prepare_search_result(
                        result,
                        search_sources,
                        search_source_numbers,
                        self.max_tool_result_characters,
                    )
                else:
                    result = limit_tool_result(
                        result, self.max_tool_result_characters
                    )
                last_tool_result = {
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": result,
                }
                self.messages.append(
                    {
                        "role": "tool",
                        "content": format_tool_message(
                            last_tool_result, format_instruction
                        ),
                    }
                )

        return self._finish_turn(
            history_before_turn,
            self._answer_from_tool_result(last_tool_result),
            discard_search_turn=bool(search_sources),
        )

    def reset(self) -> None:
        self.messages = [
            {"role": "system", "content": build_system_prompt(self.config)}
        ]

    def _finish_turn(
        self,
        history_before_turn: list[dict[str, str]],
        answer: str,
        *,
        discard_search_turn: bool = False,
    ) -> str:
        if discard_search_turn or not self.config.memory_enabled:
            self.messages = history_before_turn
        else:
            self.messages.append({"role": "assistant", "content": answer})
        return answer

    def _clean_generated_answer(
        self, response: str, last_tool_result: dict[str, Any] | None
    ) -> str:
        answer = strip_tool_calls(response)
        answer = strip_thinking(answer)
        answer = decode_literal_unicode_escapes(answer)
        if last_tool_result is not None and last_tool_result.get("name") == "web_search":
            answer = strip_search_prompt_transcript(answer)
        return answer

    def _retry_empty_response(
        self,
        empty_response: str,
        last_tool_result: dict[str, Any] | None,
        tool_schemas: list[dict[str, Any]],
    ) -> str | None:
        retry_messages = [
            *self.messages,
            {"role": "assistant", "content": empty_response},
            {
                "role": "user",
                "content": (
                    "Your previous response contained no final answer. Answer my latest request "
                    "now. Return only the final answer, without thinking text, tool calls, tool "
                    "instructions, JSON, or a source list."
                ),
            },
        ]
        retried = self.model.generate(retry_messages, tool_schemas, False)
        if parse_tool_calls(retried):
            return None
        cleaned = self._clean_generated_answer(retried, last_tool_result)
        if not cleaned or looks_like_tool_message_leak(cleaned):
            return None
        return cleaned

    def _retry_direct_refusal(
        self,
        refusal: str,
        last_tool_result: dict[str, Any] | None,
        tool_schemas: list[dict[str, Any]],
    ) -> str | None:
        retry_messages = [
            *self.messages,
            {"role": "assistant", "content": refusal},
            {
                "role": "user",
                "content": (
                    "Answer my latest request directly and factually. Do not repeat a generic "
                    "refusal, moral lecture, or unrelated warning. If one narrow part cannot be "
                    "answered, state that briefly and still answer the useful parts. Return only "
                    "the answer."
                ),
            },
        ]
        retried = self.model.generate(retry_messages, tool_schemas, False)
        if parse_tool_calls(retried):
            return None
        cleaned = self._clean_generated_answer(retried, last_tool_result)
        if (
            not cleaned
            or looks_like_generic_refusal(cleaned)
            or looks_like_tool_message_leak(cleaned)
        ):
            return None
        return cleaned

    def _revise_malformed_search_answer(
        self,
        draft: str,
        last_tool_result: dict[str, Any] | None,
        tool_schemas: list[dict[str, Any]],
        format_instruction: str | None,
    ) -> str | None:
        instruction = (
            "Rewrite the previous draft as a direct, self-contained answer to my original "
            "question. Start with the actual answer. Synthesize the evidence by topic rather "
            "than describing sources one by one. Discard copied tool instructions, tool JSON, "
            "citation blocks, and material from previous answers that does not address the current "
            "question. Do not use citation-led lines, snippet quotes, a Sources section, or raw "
            "URLs. Do not include bracket citations; LocalMind appends the numbered sources. "
            "Return only the revised answer."
        )
        if format_instruction:
            instruction += f" Output format: {format_instruction}"
        revised = self.model.generate(
            [
                *self.messages,
                {"role": "assistant", "content": draft},
                {"role": "user", "content": instruction},
            ],
            tool_schemas,
            self.config.enable_thinking,
        )
        if parse_tool_calls(revised):
            return None
        cleaned = self._clean_generated_answer(revised, last_tool_result)
        if not cleaned or self._is_malformed_search_answer(cleaned):
            return None
        return cleaned

    def _empty_response_fallback(
        self, last_tool_result: dict[str, Any] | None
    ) -> str:
        if last_tool_result is not None:
            return clean_leaked_tool_answer(last_tool_result)
        return "The model returned no final answer. Please try the request again."

    @staticmethod
    def _answer_from_tool_result(payload: dict[str, Any] | None) -> str:
        if payload is None:
            return "I reached the tool-use limit before I could finish."
        result = str(payload.get("result", ""))
        if LocalMindAgent._is_search_error(result):
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

    def _compact_messages(self) -> None:
        system = self.messages[0]
        remaining = self.max_message_characters - len(system["content"])
        kept: list[dict[str, str]] = []
        for message in reversed(self.messages[1:]):
            content = message["content"]
            if len(content) <= remaining:
                kept.append(message)
                remaining -= len(content)
                continue
            if remaining > 0:
                marker = "[Earlier content truncated]\n"
                tail_length = max(0, remaining - len(marker))
                tail = content[-tail_length:] if tail_length else ""
                kept.append({"role": message["role"], "content": marker + tail})
            break
        self.messages = [system, *reversed(kept)]

    def _has_tool(self, name: str) -> bool:
        return any(schema.get("name") == name for schema in self.tools.schemas)

    @staticmethod
    def _is_search_error(result: str) -> bool:
        return result.startswith("Search error:") or result == "No search results found."

    @staticmethod
    def _is_malformed_search_answer(answer: str) -> bool:
        return (
            looks_like_source_dump(answer)
            or looks_like_tool_message_leak(answer)
            or looks_like_search_instruction_echo(answer)
        )

    @staticmethod
    def _tool_call_key(name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=True,
        )
