import re
from dataclasses import dataclass

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    document: str
    page: int
    chunk_id: int
    content: str


def chunk_pages(
    pages: list[dict],
    document: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        for words in _pack_sentences(_split_sentences(page["text"], chunk_size, overlap), chunk_size, overlap):
            chunks.append(
                Chunk(
                    document=document,
                    page=page["page"],
                    chunk_id=len(chunks),
                    content=" ".join(words),
                )
            )
    return chunks


def _split_sentences(text: str, max_words: int, overlap: int) -> list[list[str]]:
    sentences = []
    for sentence in SENTENCE_BOUNDARY.split(text):
        words = sentence.split()
        if len(words) <= max_words:
            if words:
                sentences.append(words)
            continue
        step = max_words - overlap
        for start in range(0, len(words), step):
            sentences.append(words[start:start + max_words])
            if start + max_words >= len(words):
                break
    return sentences


def _pack_sentences(
    sentences: list[list[str]],
    chunk_size: int,
    overlap: int,
) -> list[list[str]]:
    chunks = []
    current: list[list[str]] = []
    current_words = 0

    for sentence in sentences:
        if current and current_words + len(sentence) > chunk_size:
            chunks.append([word for s in current for word in s])
            current = _overlap_tail(current, overlap)
            current_words = sum(len(s) for s in current)
            # Drop the overlap if it leaves no room for the next sentence.
            if current_words + len(sentence) > chunk_size:
                current, current_words = [], 0

        current.append(sentence)
        current_words += len(sentence)

    # Always holds at least one sentence not yet emitted, so the final chunk
    # is never pure overlap.
    if current:
        chunks.append([word for s in current for word in s])
    return chunks


def _overlap_tail(sentences: list[list[str]], overlap: int) -> list[list[str]]:
    tail = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > overlap:
            break
        tail.insert(0, sentence)
        total += len(sentence)
    return tail
