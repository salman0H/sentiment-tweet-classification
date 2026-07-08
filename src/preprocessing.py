"""
Text preprocessing pipeline for tweet sentiment classification.

The raw corpus (train_text.txt) has a few quirks that need to be handled
before tokenization, most of which come from the way the tweets were
originally serialized:

    - Some lines are wrapped in stray double quotes ("...").
    - Internal quotes were escaped as literal backslash sequences
      (e.g.  \" instead of a real quote character).
    - Some punctuation and unicode characters were written out as literal
      escape codes instead of the characters themselves
      (e.g. "can\u2019t" instead of "can't", "guys\u002c" instead of "guys,").
    - Trailing/leading whitespace and stray newlines are common.

Everything below is intentionally explicit and non-destructive by default:
each step can be toggled independently through `PreprocessingConfig`, so
that "preprocessing strategy" can itself be treated as an experimental
variable, as required by the assignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_USER_MENTION_RE = re.compile(r"@user\b", re.IGNORECASE)


@dataclass
class PreprocessingConfig:
    """Every flag below corresponds to one independent cleaning step."""

    fix_literal_unicode_escapes: bool = True   # "\u2019" -> "'"
    unescape_literal_quotes: bool = True        # '\"' -> '"'
    strip_wrapping_quotes: bool = True          # '"text"' -> 'text'
    normalize_whitespace: bool = True
    replace_user_mentions: bool = False         # "@user" -> "[USER]"
    strip_urls: bool = False
    lowercase: bool = False
    collapse_repeated_chars: bool = False       # "sooooo" -> "soo"


def _fix_literal_unicode_escapes(text: str) -> str:
    return _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _unescape_literal_quotes(text: str) -> str:
    return text.replace('\\"', '"')


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return stripped[1:-1]
    return stripped


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _replace_user_mentions(text: str) -> str:
    return _USER_MENTION_RE.sub("[USER]", text)


def _strip_urls(text: str) -> str:
    return _URL_RE.sub("[URL]", text)


def _collapse_repeated_chars(text: str) -> str:
    return _REPEATED_CHAR_RE.sub(lambda m: m.group(1) * 2, text)


_PIPELINE_STEPS = [
    ("fix_literal_unicode_escapes", _fix_literal_unicode_escapes),
    ("unescape_literal_quotes", _unescape_literal_quotes),
    ("strip_wrapping_quotes", _strip_wrapping_quotes),
    ("strip_urls", _strip_urls),
    ("replace_user_mentions", _replace_user_mentions),
    ("collapse_repeated_chars", _collapse_repeated_chars),
    ("normalize_whitespace", _normalize_whitespace),
]


def clean_text(text: str, config: PreprocessingConfig) -> str:
    """Apply every enabled step, in a fixed and documented order."""
    for flag_name, step_fn in _PIPELINE_STEPS:
        if getattr(config, flag_name):
            text = step_fn(text)
    if config.lowercase:
        text = text.lower()
    return text


def clean_corpus(texts: List[str], config: PreprocessingConfig) -> List[str]:
    return [clean_text(t, config) for t in texts]
