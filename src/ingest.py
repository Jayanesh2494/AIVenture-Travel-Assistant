from fastapi import UploadFile
from io import BytesIO

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import COLLECTION_NAME
from src.embeddings import get_embeddings
from src.vectorstores import get_qdrant_client

from qdrant_client.models import PointStruct
import uuid


async def ingest_pdf(file: UploadFile):
    print(f"📄 Processing file: {file.filename}")

    content = await file.read()

    # Load PDF from memory using pypdf
    reader = PdfReader(BytesIO(content))
    docs = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": i, "source": file.filename}
                )
            )

    print("Pages with text:", len(docs))

    if not docs:
        return {"message": "No readable text found in PDF (might be scanned/image PDF)"}

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print("Chunks created:", len(chunks))

    # Embeddings (make sure you're using MiniLM here)
    texts = [c.page_content for c in chunks]
    embeddings = get_embeddings(texts)
    print("Embeddings generated:", len(embeddings))

    # Qdrant
    client = get_qdrant_client()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": texts[i], **chunks[i].metadata}
        )
        for i in range(len(embeddings))
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    count = client.count(collection_name=COLLECTION_NAME)
    print("✅ Total vectors in DB:", count)

    return {"message": f"Uploaded {len(points)} chunks"}