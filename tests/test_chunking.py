"""Unit tests for the chunker. These run without Docker: pytest tests/test_chunking.py"""

import pytest

from app.services.chunking_service import SentenceChunker, split_into_sentences


def sentence(n: int, words: int = 10) -> str:
    return " ".join(f"s{n}w{i}" for i in range(words - 1)) + f" end{n}."


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The cat sat. The dog ran! Did it? Yes.", ["The cat sat.", "The dog ran!", "Did it?", "Yes."]),
        ("Dr. Smith arrived. He sat down.", ["Dr. Smith arrived.", "He sat down."]),
        ("See Fig. 3 for details. It shows the trend.", ["See Fig. 3 for details.", "It shows the trend."]),
        ("Written by J. Smith in 2020. It was popular.", ["Written by J. Smith in 2020.", "It was popular."]),
        ("He moved to the U.S. Army base. Then he left.", ["He moved to the U.S. Army base.", "Then he left."]),
        ("Use a model, e.g. the small one. It is fast.", ["Use a model, e.g. the small one.", "It is fast."]),
        ('She said "Stop." Then she left.', ['She said "Stop."', "Then she left."]),
        ("Version 2.5 is out. Upgrade now.", ["Version 2.5 is out.", "Upgrade now."]),
    ],
)
def test_sentence_boundaries(text, expected):
    assert split_into_sentences(text) == expected


def test_chunks_respect_word_limit_and_end_on_sentences():
    text = " ".join(sentence(n) for n in range(40))
    chunks = SentenceChunker(chunk_size=50, overlap=10).chunk([{"page": 1, "text": text}], "doc.pdf")

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content.split()) <= 50
        assert chunk.content.endswith(".")


def test_consecutive_chunks_overlap_by_whole_sentences():
    text = " ".join(sentence(n) for n in range(20))
    chunks = SentenceChunker(chunk_size=50, overlap=10).chunk([{"page": 1, "text": text}], "doc.pdf")

    for previous, current in zip(chunks, chunks[1:]):
        last_sentence = previous.content.split()[-10:]
        assert current.content.split()[:10] == last_sentence


def test_no_text_is_lost():
    sentences = [sentence(n) for n in range(30)]
    chunks = SentenceChunker(chunk_size=50, overlap=10).chunk([{"page": 1, "text": " ".join(sentences)}], "doc.pdf")

    joined = " ".join(chunk.content for chunk in chunks)
    for s in sentences:
        assert s in joined


def test_long_unpunctuated_run_is_split_with_a_sliding_window():
    words = [f"w{i}" for i in range(400)]
    chunks = SentenceChunker(chunk_size=150, overlap=30).chunk([{"page": 1, "text": " ".join(words)}], "doc.pdf")

    assert len(chunks) > 1
    assert all(len(chunk.content.split()) <= 150 for chunk in chunks)
    covered = {word for chunk in chunks for word in chunk.content.split()}
    assert covered == set(words)


def test_chunks_stay_on_their_page_and_ids_are_sequential():
    pages = [{"page": 1, "text": sentence(1)}, {"page": 2, "text": sentence(2)}]
    chunks = SentenceChunker(chunk_size=150, overlap=30).chunk(pages, "doc.pdf")

    assert [(c.page, c.chunk_id) for c in chunks] == [(1, 0), (2, 1)]
    assert all(c.document == "doc.pdf" for c in chunks)
    assert "end1" in chunks[0].content and "end2" not in chunks[0].content


def test_short_document_is_one_chunk():
    chunks = SentenceChunker(chunk_size=150, overlap=30).chunk([{"page": 1, "text": "Just one short sentence."}], "doc.pdf")
    assert [c.content for c in chunks] == ["Just one short sentence."]


@pytest.mark.parametrize("overlap", [50, 60])
def test_overlap_must_be_smaller_than_chunk_size(overlap):
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=50, overlap=overlap)
