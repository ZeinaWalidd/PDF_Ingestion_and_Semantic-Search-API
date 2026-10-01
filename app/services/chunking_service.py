import re
from dataclasses import dataclass

BOUNDARY_CANDIDATE = re.compile(r"[.!?][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
    "fig", "figs", "eq", "eqs", "ref", "refs", "vol", "ch", "sec", "approx",
    "inc", "ltd", "corp", "dept", "al", "cf",
}


@dataclass
class Chunk:
    document: str
    page: int
    chunk_id: int
    content: str


class SentenceChunker:

    def __init__(self, chunk_size: int, overlap: int):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, pages: list[dict], document: str) -> list[Chunk]:
        return chunk_pages(pages, document, self._chunk_size, self._overlap)


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


def split_into_sentences(text: str) -> list[str]:
    sentences = []
    start = 0
    for match in BOUNDARY_CANDIDATE.finditer(text):
        if text[match.start()] == "." and _is_abbreviation(text[start:match.start()]):
            continue
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    sentences.append(text[start:].strip())
    return [sentence for sentence in sentences if sentence]


def _is_abbreviation(text_before_period: str) -> bool:
    words = text_before_period.split()
    if not words:
        return False
    word = words[-1].lstrip("\"'([").lower()
    # Single initials ("J. Smith") and dotted forms ("e.g", "U.S", "Ph.D").
    if len(word) == 1 and word.isalpha():
        return True
    if "." in word and word.replace(".", "").isalpha():
        return True
    return word in ABBREVIATIONS


def _split_sentences(text: str, max_words: int, overlap: int) -> list[list[str]]:
    sentences = []
    for sentence in split_into_sentences(text):
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
