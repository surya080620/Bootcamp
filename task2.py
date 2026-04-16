from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import os


def load_pdfs(file_paths):
    documents = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    return documents


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def attach_metadata(chunks, source_type="unknown"):
    upload_date = datetime.now().strftime("%Y-%m-%d")

    for chunk in chunks:
        source_path = chunk.metadata.get("source", "")

        chunk.metadata["filename"] = os.path.basename(source_path)
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0)
        chunk.metadata["upload_date"] = upload_date
        chunk.metadata["source_type"] = source_type

    return chunks


def filter_chunks(chunks, **filters):
    filtered_chunks = []

    for chunk in chunks:
        match = True

        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                match = False
                break

        if match:
            filtered_chunks.append(chunk)

    return filtered_chunks


if __name__ == "__main__":
    pdf_files = ["sample1.pdf", "sample2.pdf"]

    docs = load_pdfs(pdf_files)
    chunks = split_into_chunks(docs)
    chunks = attach_metadata(chunks, source_type="research_paper")

    filtered = filter_chunks(chunks, page_number=0)

    print(f"Number of filtered chunks: {len(filtered)}")

    if filtered:
        print("\nSample chunk:\n")
        print(filtered[0].page_content[:300])
        print("\nMetadata:", filtered[0].metadata)