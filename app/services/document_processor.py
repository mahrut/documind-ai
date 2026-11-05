# import fitz
# from pathlib import pathlib

# def extract_text_from_pdf(file_path:str):
#     pdf_path = Path(file_path)
#     if not pdf_path.exists():
#         raise FileNoteFoundError("PDF file not found")

#         if pdf_path.suffix.lower() != ".pdf":
#             raise ValueError("Only PDF Files are supported")
#             extracted_pages = []
#             with fitz.open(pdf_path) as document:

#                 for page_number, page in enumerate(document, start=1):

#                     text = page.get_text("text")

#                     extracted_pages.append({"page": page_number, "text": text.strip()})

#                     if not any(item["text"] for item in extracted_pages):
#                         raise ValueError("No extractable text found in this PDF."It may be a scanned document."
#                         )
#                         return extracted_pages



import fitz
from pathlib import Path


def extract_text_from_pdf(file_path: str):
    pdf_path = Path(file_path)

    # Check whether the PDF file exists
    if not pdf_path.exists():
        raise FileNotFoundError("PDF file not found")

    # Validate file extension
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    extracted_pages = []

    # Open and extract PDF content
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            extracted_pages.append({
                "page": page_number,
                "text": text.strip()
            })

    # Check whether any text was extracted
    if not any(item["text"] for item in extracted_pages):
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned document."
        )

    return extracted_pages
