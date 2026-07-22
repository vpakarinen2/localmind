from __future__ import annotations

import re

from typing import Any


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
TOP_COUNT_PATTERN = r"\btop\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
SOURCE_COUNT_PATTERN = (
    r"\b(?:(?:use|with|from|include|using)\s+)?"
    r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:web\s+)?sources?\b(?:\s+and\b)?"
)


def select_tool_schemas(
    prompt: str, schemas: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lowered = prompt.lower()
    selected: set[str] = set()
    if re.search(r"\b(use|call)\s+(a\s+|the\s+)?tools?\b", lowered):
        return schemas
    if re.search(r"\d\s*(?:\+|-|\*|/|%|\^)\s*\d", lowered) or any(
        word in lowered for word in ("calculate", "compute", "arithmetic")
    ):
        selected.add("calculate")
    if any(
        phrase in lowered
        for phrase in (
            "current time",
            "what time",
            "current date",
            "what date",
            "today's date",
            "tomorrow",
            "yesterday",
            "deadline",
            "schedule",
        )
    ):
        selected.add("current_time")
    selected.update(file_tools_for_prompt(lowered))
    explicit_search = any(
        phrase in lowered
        for phrase in (
            "web search",
            "search the web",
            "search online",
            "look it up",
            "release date",
        )
    )
    if explicit_search or re.search(
        r"\b(?:latest|current|recent|news|price|weather|research|sources?|cite)\b",
        lowered,
    ):
        selected.add("web_search")
    return [schema for schema in schemas if schema.get("name") in selected]


def file_tools_for_prompt(prompt: str) -> set[str]:
    explicit_tool_names = set(re.findall(r"\b(read_file|write_file)\b", prompt))
    prompt_without_links = re.sub(r"https?://\S+|\b\S+@\S+\b", "", prompt)
    path_reference = re.search(
        r"(?:^|\s)(?:\.{0,2}[\\/]|[a-z]:[\\/])?"
        r"[^\s\\/:*?\"<>|]+(?:[\\/][^\s\\/:*?\"<>|]+)*"
        r"\.[a-z0-9]{1,12}\b",
        prompt_without_links,
        re.IGNORECASE,
    )
    filesystem_noun = re.search(
        r"\b(?:file|folder|directory|workspace|filepath|filename)\b",
        prompt_without_links,
    )
    if path_reference is None and filesystem_noun is None:
        return explicit_tool_names
    selected = set(explicit_tool_names)
    if re.search(r"\b(?:read|open|load|inspect|show|display|summari[sz]e)\b", prompt):
        selected.add("read_file")
    if re.search(r"\b(?:write|save|create|edit|update|append|store)\b", prompt):
        selected.add("write_file")
    return selected


def should_presearch(prompt: str, *, search_available: bool) -> bool:
    if not search_available:
        return False
    lowered = prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "use web search",
            "web search",
            "search the web",
            "look it up",
            "search online",
        )
    )


def search_query(prompt: str) -> str:
    query = re.sub(r"\b(use\s+)?web\s+search\b", "", prompt, flags=re.IGNORECASE)
    query = re.sub(r"\bsearch\s+the\s+web\b", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\blook\s+it\s+up\b", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\bsearch\s+online\b", "", query, flags=re.IGNORECASE)
    query = re.sub(TOP_COUNT_PATTERN, "", query, flags=re.IGNORECASE)
    query = re.sub(SOURCE_COUNT_PATTERN, "", query, flags=re.IGNORECASE)
    format_patterns = (
        r"\b(?:(?:as|in|using|with)\s+)?(?:a\s+)?(?:bullet(?:ed)?(?: point)? list|bullet points?|bullets)\b",
        r"\b(?:(?:as|in|using|with)\s+)?(?:a\s+)?(?:numbered list|numbered points?|numbered format)\b",
        r"\b(?:as|in|using|with)\s+markdown\b",
        r"\b(?:(?:as|in)\s+)?(?:a\s+)?(?:single\s+|one\s+)?paragraph\b",
        r"\b\d{1,2}\s+paragraphs?\b",
    )
    for pattern in format_patterns:
        query = re.sub(pattern, "", query, flags=re.IGNORECASE)
    query = re.sub(
        r"\b(?:give|show|return|present|format|use)(?:\s+me)?(?:\s+the)?(?=\s*[.,!?;:]|$)",
        "",
        query,
        flags=re.IGNORECASE,
    )
    trailing_command = (
        r"(?:^|[.!?;:]\s*)\b(?:give|show|return|present|format|use)"
        r"(?:\s+me)?(?:\s+the)?\s*[.,!?;:]*\s*$"
    )
    previous_query = None
    while query != previous_query:
        previous_query = query
        query = re.sub(trailing_command, "", query, flags=re.IGNORECASE).strip()
    query = query.replace("?", " ").strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"\s+([.,!?:;])", r"\1", query).strip()
    query = re.sub(r"(?:\s*[.,!;:])+\s*$", "", query).strip()
    return query or prompt


def requested_top_count(prompt: str) -> int | None:
    match = re.search(TOP_COUNT_PATTERN, prompt, flags=re.IGNORECASE)
    if match is None:
        return None
    raw_count = match.group(1).lower()
    count = int(raw_count) if raw_count.isdigit() else NUMBER_WORDS[raw_count]
    return max(1, min(count, 10))


def requested_source_count(prompt: str) -> int | None:
    match = re.search(SOURCE_COUNT_PATTERN, prompt, flags=re.IGNORECASE)
    if match is None:
        return None
    raw_count = match.group(1).lower()
    count = int(raw_count) if raw_count.isdigit() else NUMBER_WORDS[raw_count]
    return max(1, min(count, 10))


def search_result_limit(prompt: str) -> int:
    return requested_source_count(prompt) or 10


def response_format_instruction(prompt: str) -> str | None:
    lowered = prompt.lower()
    top_count = requested_top_count(prompt)
    instruction: str | None = None
    if top_count is not None:
        instruction = (
            f"Return a numbered list with up to {top_count} concrete supported items. "
            "Do not invent items to reach the requested count."
        )
    elif re.search(r"\b(?:numbered list|numbered points?|numbered format)\b", lowered):
        instruction = "Return a concise numbered list of concrete supported items."
    elif re.search(r"\b(?:bullet(?:ed)?(?: point)? list|bullet points?|bullets)\b", lowered):
        instruction = "Return a concise bullet-point list of concrete supported items."
    elif paragraph_match := re.search(r"\b(\d{1,2})\s+paragraphs?\b", lowered):
        count = max(1, min(int(paragraph_match.group(1)), 10))
        instruction = (
            f"Return exactly {count} concise prose paragraphs separated by blank lines. "
            "Do not number, bullet, label, or add headings to the paragraphs."
        )
    elif re.search(r"\b(?:a |one |single )?paragraph\b|\bin prose\b", lowered):
        instruction = (
            "Return one concise paragraph in normal prose. Do not format it as a list."
        )
    elif re.search(r"\b(?:latest|recent|news|headlines|updates?)\b", lowered):
        instruction = (
            "Return a numbered list of up to 5 concrete recent developments, including "
            "dates when the evidence provides them."
        )
    if re.search(r"\bmarkdown\b", lowered):
        markdown = "Use clear Markdown syntax."
        instruction = f"{instruction} {markdown}" if instruction else markdown
    return instruction
