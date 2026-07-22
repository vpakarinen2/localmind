from __future__ import annotations

import re


def strip_thinking(answer: str) -> str:
    without_blocks = re.sub(
        r"<\s*think\s*>.*?<\s*/\s*think\s*>",
        "",
        answer,
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
