"""Shared Unicode NFC text normalization helpers."""

from __future__ import annotations

import unicodedata
from typing import Any


def normalize_nfc(text: Any) -> str:
    """Normalize text to Unicode NFC while stripping NUL bytes and outer whitespace."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).replace("\x00", "").strip()
