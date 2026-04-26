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

    # Load PDF
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
        return {"message": "No readable text found in PDF"}

    # ✅ OPTIMIZED SPLITTING
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # 🔥 reduced
        chunk_overlap=50     # 🔥 reduced
    )

    chunks = splitter.split_documents(docs)
    print("Chunks created:", len(chunks))

    # ✅ SAFETY LIMIT (avoid overload)
    if len(chunks) > 200:
        chunks = chunks[:200]
        print("⚠️ Limited to 200 chunks")

    texts = [c.page_content for c in chunks]

    # Embeddings
    embeddings = get_embeddings(texts)
    print("Embeddings generated:", len(embeddings))

    # Qdrant client
    client = get_qdrant_client()

    # Prepare points
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": texts[i], **chunks[i].metadata}
        )
        for i in range(len(embeddings))
    ]

    # ✅ BATCH UPSERT (🔥 FIX TIMEOUT)
    batch_size = 20

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        print(f"⬆️ Uploading batch {i // batch_size + 1}")

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

    count = client.count(collection_name=COLLECTION_NAME)
    print("✅ Total vectors in DB:", count)

    return {"message": f"Uploaded {len(points)} chunks successfully"}