import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def create_chunks(text: str, chunk_size: int = 100, overlap: int = 20):
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks


def process_document_pages(pages):
    processed_chunks = []

    for page in pages:
        page_number = page["page"]
        cleaned_text = clean_text(page["text"])
        chunks = create_chunks(cleaned_text)

        for index, chunk in enumerate(chunks):
            processed_chunks.append({
                "page": page_number,
                "chunk_index": index + 1,
                "text": chunk
            })

    return processed_chunks