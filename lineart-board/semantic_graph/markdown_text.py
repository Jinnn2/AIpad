from __future__ import annotations

import re


_BLOCK_MD_RE = re.compile(
    r"(^|\n)\s{0,3}(#{1,6}\s+\S|[-*+]\s+(?:\[[ xX]\]\s+)?\S|\d+\.\s+\S|>\s+\S|```)",
    re.MULTILINE,
)
_INLINE_MD_RE = re.compile(
    r"(\*\*[^*\n][^*\n]*\*\*|__[^_\n][^_\n]*__|!\[[^\]\n]*\]\([^)]+\)|\[[^\]\n]+\]\([^)]+\)|`[^`\n]+`)"
)


def looks_like_markdown_text(value: str) -> bool:
    if not value:
        return False
    source = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return bool(_BLOCK_MD_RE.search(source) or _INLINE_MD_RE.search(source))


def _strip_inline_markdown(line: str) -> str:
    text = line
    for _ in range(4):
        prev = text
        text = re.sub(r"!\[([^\]\n]*)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\[([^\]\n]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"`([^`\n]+)`", r"\1", text)
        text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_\n]+)__", r"\1", text)
        text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
        text = re.sub(r"_([^_\n]+)_", r"\1", text)
        text = re.sub(r"~~([^~\n]+)~~", r"\1", text)
        if text == prev:
            break
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!>])", r"\1", text)


def markdown_to_plain_text(value: str) -> str:
    if not value:
        return ""
    source = str(value).replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    in_fence = False
    for raw_line in source.split("\n"):
        line = raw_line.replace("\t", "  ")
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(_strip_inline_markdown(line))
            continue

        match = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
        if match:
            out.append(_strip_inline_markdown(match.group(2)).strip())
            continue

        match = re.match(r"^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$", line)
        if match:
            indent = " " * min(len(match.group(1) or ""), 6)
            mark = "[x]" if (match.group(2) or "").lower() == "x" else "[ ]"
            out.append(f"{indent}{mark} {_strip_inline_markdown(match.group(3)).strip()}")
            continue

        match = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if match:
            indent = " " * min(len(match.group(1) or ""), 6)
            out.append(f"{indent}- {_strip_inline_markdown(match.group(2)).strip()}")
            continue

        match = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if match:
            indent = " " * min(len(match.group(1) or ""), 6)
            out.append(f"{indent}{match.group(2)}. {_strip_inline_markdown(match.group(3)).strip()}")
            continue

        match = re.match(r"^\s*>\s?(.*)$", line)
        if match and trimmed.startswith(">"):
            out.append(f"> {_strip_inline_markdown(match.group(1)).strip()}".rstrip())
            continue

        out.append(_strip_inline_markdown(line))

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip()


def markdown_to_semantic_text(value: str) -> str:
    if not value:
        return ""
    source = str(value)
    if not looks_like_markdown_text(source):
        return source.replace("\r\n", "\n").replace("\r", "\n").strip()
    return markdown_to_plain_text(source).strip()
