import re
import unicodedata


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_CHARACTERS = re.compile(r"[\u200b-\u200d\ufeff]")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(content: str) -> str:
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    cleaned = unicodedata.normalize("NFC", content).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHARACTERS.sub("", cleaned)
    cleaned = _ZERO_WIDTH_CHARACTERS.sub("", cleaned)
    cleaned = _HORIZONTAL_WHITESPACE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return _EXCESS_BLANK_LINES.sub("\n\n", cleaned).strip()
