from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel

from src.ingest import ingest_pdf
from src.vectorstores import init_qdrant
from src.generator import generate_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Qdrant database...")
    init_qdrant()
    print("Database initialization complete.")
    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "AI Travel Assistant running 🚀"}


@app.post("/ask")
async def ask_question(req: QueryRequest):
    resp = generate_answer(req.query)
    return {"response": resp}


# ✅ ONLY ONE upload endpoint (FIXED)
@app.post("/upload")
async def upload_file(file: UploadFile = None):
    if not file:
        return {"message": "No file uploaded"}

    if not file.filename.endswith('.pdf'):
        return {"message": "Please upload a PDF file"}

    try:
        result = await ingest_pdf(file)
        return result
    except Exception as e:
        return {"message": f"Error processing file: {str(e)}"}