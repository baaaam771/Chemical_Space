"""Korean particle selection, so breadcrumbs read correctly.

``물성`` ends in a final consonant and takes ``으로``; ``모이어티`` does not and
takes ``로``.  Getting this wrong is small but reads as broken Korean in the one
place the user looks most -- the exploration path.
"""

from __future__ import annotations

_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3
_JONG_RIEUL = 8


def _jongseong(word: str) -> int | None:
    """Final-consonant index of the last syllable, or None if not Hangul."""
    if not word:
        return None
    ch = word.strip()[-1]
    code = ord(ch)
    if not (_HANGUL_START <= code <= _HANGUL_END):
        return None
    return (code - _HANGUL_START) % 28


def ro(word: str) -> str:
    """``word`` + 으로/로."""
    jong = _jongseong(word)
    if jong is None:
        return f"{word}(으)로"
    if jong == 0 or jong == _JONG_RIEUL:
        return f"{word}로"
    return f"{word}으로"


def eun(word: str) -> str:
    """``word`` + 은/는."""
    jong = _jongseong(word)
    if jong is None:
        return f"{word}은(는)"
    return f"{word}는" if jong == 0 else f"{word}은"


def eul(word: str) -> str:
    """``word`` + 을/를."""
    jong = _jongseong(word)
    if jong is None:
        return f"{word}을(를)"
    return f"{word}를" if jong == 0 else f"{word}을"
