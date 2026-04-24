from fastapi import UploadFile
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import COLLECTION_NAME
from src.embeddings import get_embeddings
from src.vectorstores import get_qdrant_client

from qdrant_client.models import PointStruct
import uuid


async def ingest_pdf(file: UploadFile):
    content = await file.read()

    docs = []
    pdf = fitz.open(stream=content, filetype="pdf")

    try:
        for page_num in range(len(pdf)):
            text = pdf[page_num].get_text().strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_num, "source": file.filename}
                    )
                )
    finally:
        pdf.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    embeddings = get_embeddings(texts)

    client = get_qdrant_client()

    payloads = [
        {"text": c.page_content, **c.metadata}
        for c in chunks
    ]

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload=payloads[i]
        )
        for i in range(len(embeddings))
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return {"message": f"{len(points)} chunks uploaded"}