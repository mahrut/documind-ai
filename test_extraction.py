# from app.services.document_processor 
# import extract_text_from_pdf

# result = extract_text_from_pdf("data/documents/research.pdf")

# for page in result:
#     print(f"\nPage: {page['page']}")
#     print(page['text'])


from app.services.document_processor import extract_text_from_pdf

result = extract_text_from_pdf("data/documents/research.pdf")

for page in result:
    print(f"\nPage: {page['page']}")
    print(page['text'])
