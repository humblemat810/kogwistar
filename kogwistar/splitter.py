from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start_char: int
    end_char: int  # exclusive
    text: str


def split_doc_deterministic(
    *,
    doc_id: str,
    content: str,
    max_chars: int = 6000,
    overlap_chars: int = 400,
    prefer_window: int | None = None,
) -> List[Chunk]:
    """
    Deterministic, offset-preserving splitter.

    - Always returns chunks with exact (start,end) offsets.
    - Tries to split on "nice" boundaries near the limit:
        1) paragraph breaks: '\\n\\n'
        2) single newlines: '\\n'
        3) whitespace
        4) punctuation (incl. CJK)
      within a backward window of `prefer_window`.
    - If none found, does a hard cut at max_chars.
    - Applies overlap by moving next start back by overlap_chars.

    overlap_chars: a hard limit that may split words into letters

    Multilingual-safe because it doesn't require word/sentence tokenization.
    """
    if prefer_window is None:
        prefer_window = overlap_chars // 2
    if prefer_window < 0:
        raise ValueError("prefer_window must be >= 0")
    guaranteed_progression = max_chars - (prefer_window + overlap_chars)

    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if guaranteed_progression <= 0:
        raise ValueError(
            "max_chars must be at larger than overlap_chars+prefer_window "
        )

    n = len(content)
    if n == 0:
        return [Chunk(doc_id=doc_id, start_char=0, end_char=0, text="")]

    # Some common “good” boundary characters across languages
    punct = set(".!?;:。！？；：")  # extend if you want
    whitespace = set([" ", "\t", "\n", "\r"])

    def find_break(start: int, hard_end: int) -> int:
        """
        Choose an end offset <= hard_end that is a good boundary.
        Searches backwards within a window.
        Returns an end offset (exclusive).
        """
        if hard_end >= n:
            hard_end = n

        # If we’re already at end, done
        if hard_end <= start:
            return start

        # Backward search window [lo, hard_end]
        lo = max(start + 1, hard_end - prefer_window)

        window = content[lo:hard_end]

        # 1) Paragraph break: split *after* the break
        idx = window.rfind("\n\n")
        if idx != -1:
            return lo + idx + 2

        # 2) Single newline
        idx = window.rfind("\n")
        if idx != -1:
            return lo + idx + 1

        # 3) Any whitespace boundary
        for i in range(lo, hard_end, 1):
            if content[i] in whitespace:
                return i + 1  # end after whitespace

        # 4) Punctuation boundary (end after punctuation)
        for i in range(lo, hard_end, 1):
            if content[i] in punct:
                return i + 1

        # 5) No boundary found → hard cut
        return hard_end

    chunks: List[Chunk] = []
    start = 0
    cnt = 0
    while start < n:
        # start here is overlap included
        if n - max(start, 0) <= max_chars:
            text = content[start:n]
            chunks.append(Chunk(doc_id=doc_id, start_char=start, end_char=n, text=text))
            break
        hard_end = min(n, start + max_chars)
        end = find_break(start, hard_end)

        # Safety: ensure forward progress
        if end <= start:
            end = hard_end
            if end <= start:
                break

        text = content[start:end]
        chunks.append(Chunk(doc_id=doc_id, start_char=start, end_char=end, text=text))

        if end >= n:
            break
        # if n - end <= overlap_chars:
        #     text = content[end:n]
        #     chunks.append(Chunk(doc_id=doc_id, start_char=end, end_char=n, text=text))
        #     break
        # Next start with overlap

        start = find_break(max(0, end - overlap_chars), end - (overlap_chars // 2))
        if start == end - (overlap_chars // 2):  # no break found
            start = max(0, end - overlap_chars)

        # Prevent infinite loops if overlap is too large and boundary chooser returns same end
        if (
            chunks
            and start < chunks[-1].end_char
            and (chunks[-1].end_char - start) >= max_chars
        ):
            start = chunks[-1].end_char  # force progress, but indeed never happen

        # quick check if it exceed a hard limit of maximum possible chunk iteration if implemented correctly
        cnt += 1
        if cnt >= int(n // guaranteed_progression) + 1:
            raise Exception("Infinite loop or incorrect implementation")
    # Final assertion: source-map correctness
    for c in chunks:
        assert c.text == content[c.start_char : c.end_char]

    return chunks
