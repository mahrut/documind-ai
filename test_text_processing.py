from app.services.text_processor import (
    clean_text,
    create_chunks,
    process_document_pages
)


# Test 1: Text cleaning

text = "Hello     AI\n\n   World"

cleaned = clean_text(text)

print("Cleaned text:")
print(cleaned)


# Test 2: Chunking

sample_text = " ".join(
    [f"word{i}" for i in range(1, 251)]
)

chunks = create_chunks(
    sample_text,
    chunk_size=100,
    overlap=20
)

print("\nNumber of chunks:", len(chunks))

for index, chunk in enumerate(chunks):

    print(f"\nChunk {index + 1}:")
    print("Word count:", len(chunk.split()))


# Test 3: Preserve page information

pages = [
    {
        "page": 1,
        "text": "Artificial Intelligence is fascinating."
    },
    {
        "page": 2,
        "text": "Machine Learning is a subset of AI."
    }
]

result = process_document_pages(pages)

print("\nProcessed document:")
print(result)