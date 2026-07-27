from __future__ import annotations

import re


RESPONSE_WRAPPER_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:"
    r"(?:BEGIN|END)_(?:ANSWER|RESPONSE)"
    r"|<\s*/?\s*(?:answer|response)\s*>"
    r")[ \t]*(?:\r?\n|$)"
)
LIST_ITEM_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<marker>[-*+]|\d+[.)])\s+(?P<content>.+?)\s*$"
)
LABELED_LIST_ITEM_PATTERN = re.compile(
    r"^(?!https?://)"
    r"(?P<label>(?=[^:\n]{1,120}:)(?=[^:\n]*[A-Za-z])[^:\n]+?)"
    r"\s*:\s*\S",
    re.IGNORECASE,
)


def strip_thinking(answer: str, *, assume_leading_thinking: bool = False) -> str:
    without_blocks = re.sub(
        r"<\s*think\s*>.*?<\s*/\s*think\s*>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if assume_leading_thinking:
        orphan_closing_tag = re.search(
            r"<\s*/\s*think\s*>", without_blocks, flags=re.IGNORECASE
        )
        if orphan_closing_tag is not None:
            without_blocks = without_blocks[orphan_closing_tag.end() :]
    without_blocks = re.sub(
        r"<\s*think\s*>.*$",
        "",
        without_blocks,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return re.sub(
        r"<\s*/?\s*think\s*>", "", without_blocks, flags=re.IGNORECASE
    ).strip()


def decode_literal_unicode_escapes(answer: str) -> str:
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        answer,
    )


def strip_response_wrappers(answer: str) -> str:
    return RESPONSE_WRAPPER_PATTERN.sub("", answer).strip()


def deduplicate_list_entries(answer: str) -> str:
    lines = answer.splitlines()
    seen: set[str] = set()
    cleaned_lines: list[str] = []
    removed_duplicate = False
    removed_numbered_duplicate = False

    for line in lines:
        match = LIST_ITEM_PATTERN.match(line)
        if match is None:
            cleaned_lines.append(line)
            continue

        content = match.group("content")
        label_match = LABELED_LIST_ITEM_PATTERN.match(content)
        if label_match is not None:
            label = re.sub(r"[*_`~]", "", label_match.group("label"))
            normalized_label = re.sub(
                r"[^\w]+", " ", label, flags=re.UNICODE
            ).casefold().strip()
            key = f"label:{normalized_label}"
        else:
            normalized = re.sub(r"\s+", " ", content).casefold().strip(" .,:;")
            key = f"item:{normalized}"

        if key in seen:
            removed_duplicate = True
            if match.group("marker")[0].isdigit():
                removed_numbered_duplicate = True
            continue
        seen.add(key)
        cleaned_lines.append(line)

    if not removed_duplicate:
        return answer
    if removed_numbered_duplicate:
        next_number = 1
        renumbered_lines: list[str] = []
        for line in cleaned_lines:
            match = LIST_ITEM_PATTERN.match(line)
            if match is not None and match.group("marker")[0].isdigit():
                line = (
                    f"{match.group('indent')}{next_number}. "
                    f"{match.group('content')}"
                )
                next_number += 1
            renumbered_lines.append(line)
        cleaned_lines = renumbered_lines
    return "\n".join(cleaned_lines).strip()


def strip_inline_urls(answer: str) -> str:
    without_markdown_urls = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", answer)
    return re.sub(r"https?://\S+", "", without_markdown_urls)


def strip_inline_citations(answer: str) -> str:
    without_citations = re.sub(
        r"[ \t]*\[(?:\d+(?:\s*[-,]\s*\d+)*)\]",
        "",
        answer,
    )
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", without_citations).strip()


def strip_model_sources(answer: str) -> str:
    return re.split(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?Sources\s*:?\s*\n",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def looks_like_generic_refusal(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer.lower()).strip()
    refusal_phrases = (
        "i cannot provide",
        "i can't provide",
        "i cannot assist",
        "i can't assist",
        "i cannot help",
        "i can't help",
        "i'm unable to",
        "i am unable to",
        "illegal and unethical",
        "against my programming",
    )
    return any(phrase in normalized for phrase in refusal_phrases)


def normalize_requested_paragraphs(answer: str, prompt: str) -> str:
    """Remove list markers when a model lists the exact requested paragraph count."""
    match = re.search(r"\b(\d{1,2})\s+paragraphs?\b", prompt, re.IGNORECASE)
    if match is None:
        return answer
    requested_count = max(1, min(int(match.group(1)), 10))
    marker_pattern = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+")
    markers = list(marker_pattern.finditer(answer))
    if len(markers) != requested_count or answer[: markers[0].start()].strip():
        return answer
    paragraphs: list[str] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(answer)
        paragraph = answer[marker.end() : end].strip()
        if not paragraph:
            return answer
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)
