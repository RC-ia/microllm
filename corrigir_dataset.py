"""Remove Markdown visual formatting from message content in a conversation JSON."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


FENCE_RE = re.compile(r"(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n(.*?)^[ \t]*\1[ \t]*$")


def _clean_inline(text: str) -> str:
    # Images and links: retain the human-readable label and destination.
    text = re.sub(r"!\[([^\]]*)\]\((?:<([^>]+)>|([^\s)]+))(?:\s+['\"][^)]*['\"])?\)",
                  lambda m: m.group(1).strip(), text)
    text = re.sub(r"\[([^\]]+)\]\((?:<([^>]+)>|([^\s)]+))(?:\s+['\"][^)]*['\"])?\)",
                  lambda m: f"{m.group(1).strip()} ({m.group(2) or m.group(3)})", text)
    text = re.sub(r"\[([^\]]+)\]\[([^\]]*)\]", lambda m: m.group(1), text)
    text = re.sub(r"(?<!\\)`+([^`\n]+)`+", r"\1", text)
    # Remove emphasis/strike markers while retaining their contents.
    text = re.sub(r"(?<!\\)(\*\*\*|___)(.+?)\1", r"\2", text, flags=re.S)
    text = re.sub(r"(?<!\\)(\*\*|__)(.+?)\1", r"\2", text, flags=re.S)
    text = re.sub(r"(?<![\w\\])([*_~])([^\n]+?)\1(?!\w)", r"\2", text)
    text = re.sub(r"\\([\\`*_{}\[\]()#+.!|>~-])", r"\1", text)
    text = re.sub(r"<((?:https?://|mailto:)[^ >]+)>", r"\1", text)
    text = re.sub(r"<\/?(?:a|abbr|b|blockquote|br|code|del|div|em|h[1-6]|i|li|ol|p|pre|s|span|strong|sub|sup|u|ul)\b[^>]*>", "", text, flags=re.I)
    return html.unescape(text)


def _clean_prose(text: str) -> str:
    text = _clean_inline(text)
    output = []
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        # Drop reference definitions, thematic breaks, and Markdown table rulers.
        if re.match(r"^\s{0,3}\[[^\]]+\]:\s+\S+", line):
            continue
        if re.match(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", line):
            continue
        if re.match(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", line):
            continue
        # Strip headings, quotes, unordered/ordered list and task-list markers.
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"^(\s*)(?:[-+*]|\d+[.)])\s+", r"\1", line)
        line = re.sub(r"^\s*\[[ xX]\]\s+", "", line)
        # Turn table cells into readable text instead of retaining pipe syntax.
        is_table_row = "|" in line and (
            (i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i+1]))
            or (i > 0 and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i-1]))
        )
        if is_table_row:
            line = " — ".join(cell.strip() for cell in line.strip().strip("|").split("|") if cell.strip())
        output.append(line)
    text = "\n".join(output)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_markdown(text: str) -> str:
    """Remove Markdown syntax and flatten model-style escaped/newline formatting."""
    chunks = []
    cursor = 0
    for match in FENCE_RE.finditer(text):
        chunks.append(_clean_prose(text[cursor:match.start()]))
        chunks.append(match.group(2).strip("\n"))
        cursor = match.end()
    chunks.append(_clean_prose(text[cursor:]))
    result = "\n\n".join(part for part in chunks if part)
    # Some exporters serialize newlines as literal backslash-n sequences; handle
    # those as well as actual newlines before removing remaining backslashes.
    result = re.sub(r"\\[nr]", " ", result)
    result = result.replace("\\", "")
    result = re.sub(r"\s+([,.;!?])", r"\1", result)
    result = re.sub(r"([!?])(?=[^\W\d_])", r"\1 ", result)
    return re.sub(r"\s+", " ", result).strip()


def clean_dataset(data):
    changed = 0
    for record in data if isinstance(data, list) else [data]:
        if not isinstance(record, dict):
            continue
        for message in record.get("messages", []):
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                old = message["content"]
                message["content"] = strip_markdown(old)
                changed += old != message["content"]
    return data, changed


def main():
    parser = argparse.ArgumentParser(description="Converte Markdown em texto simples nos campos messages[].content.")
    parser.add_argument("input", nargs="?", default="dataset.json", help="JSON de entrada (padrão: dataset.json)")
    parser.add_argument("-o", "--output", default="dataset_texto_normal.json", help="arquivo de saída separado")
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        parser.error("o arquivo de saída precisa ser diferente do original")
    data = json.loads(source.read_text(encoding="utf-8"))
    data, changed = clean_dataset(data)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"input": str(source), "output": str(target), "messages_changed": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
