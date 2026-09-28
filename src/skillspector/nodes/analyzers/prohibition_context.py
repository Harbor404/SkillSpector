# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize a small, bounded grammar of direct defensive instructions.

This is deliberately not a nearby-negation heuristic. Only a prohibition that
immediately governs the matched action can exempt it; unknown grammar retains
ordinary detection. Callers must pass match offsets in the same content view.
"""

from __future__ import annotations

import re

from .common import LOGICAL_LINE_BREAK

_CONTEXT_CHARS = 512
_MARKDOWN = str.maketrans("", "", "*_`")
_SPACE = r"\s+"
_DIRECT_PREFIX = re.compile(
    r"(?:\A|(?<=[\n\r.!?;,\[]))[ \t]*"
    r"(?:[-+]\s+|\d+[.)]\s+)?"
    rf"(?:please{_SPACE})?"
    rf"(?:(?:you|(?:the{_SPACE})?(?:agent|assistant|model)){_SPACE})?"
    rf"(?:do{_SPACE}not|don['’]t|never|no|must{_SPACE}not|shall{_SPACE}not){_SPACE}"
    rf"(?:(?:ever|first|verbatim|exactly|word{_SPACE}for{_SPACE}word){_SPACE})?\Z",
    re.IGNORECASE,
)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_CONTRACTION = re.compile(r"(?<=\w)['’](?=\w)")
_SENTENCE_END = re.compile(r"[.!?;]")
_EXCEPTION = re.compile(r"\b(?:unless|except|until|but|however|instead)\b", re.IGNORECASE)

_DISAVOWAL = re.compile(
    r"\b(?:ignore|disregard|override|obsolete|invalid|bypass|suspend|violate)\b",
    re.IGNORECASE,
)


def is_directly_prohibited(content: str, start: int, end: int) -> bool:
    """Whether a bounded, unambiguous prohibition governs this exact action.

    Markdown emphasis and inline-code delimiters may surround the prohibition;
    quotation marks and arbitrary intervening words are not exempted. Exceptions
    in the same clause also retain detection. Work per match is constant-bounded,
    and incomplete context fails conservatively.
    """
    if not 0 <= start < end <= len(content):
        return False
    left = max(0, start - _CONTEXT_CHARS)
    prefix = content[left:start].translate(_MARKDOWN)
    prohibited = _DIRECT_PREFIX.search(prefix)
    if prohibited is None or (left and prohibited.start() == 0):
        return False
    leading = prefix[: prohibited.start()]
    quoted = _CONTRACTION.sub("", leading)
    if sum(quoted.count(char) for char in '"“”') % 2:
        return False
    if sum(quoted.count(char) for char in "'‘’") % 2:
        return False
    if _DISAVOWAL.search(leading):
        return False
    if _BLANK_LINE.search(LOGICAL_LINE_BREAK.sub("\n", prohibited.group())):
        return False

    # A long unfinished clause may hide a later exception. Do not infer safety
    # from an arbitrarily clipped fragment of that clause.
    tail = content[end : end + _CONTEXT_CHARS]
    if _DISAVOWAL.search(tail):
        return False
    boundary = _SENTENCE_END.search(tail)
    if boundary is None and end + len(tail) < len(content):
        return False
    clause = tail[: boundary.start()] if boundary is not None else tail
    return _EXCEPTION.search(clause) is None
