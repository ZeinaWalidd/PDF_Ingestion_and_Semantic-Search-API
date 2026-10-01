from dataclasses import dataclass


@dataclass
class Chunk:
    document: str
    page: int
    chunk_id: int
    content: str


def chunk_pages(
    pages: list[dict],
    document: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            content = " ".join(words[start:end]).strip()

            if content:
                chunks.append(
                    Chunk(
                        document=document,
                        page=page["page"],
                        chunk_id=chunk_id,
                        content=content,
                    )
                )
                chunk_id += 1

            if end >= len(words):
                break

            start += step

    return chunks
