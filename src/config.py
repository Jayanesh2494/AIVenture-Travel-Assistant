import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "AIVenture")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_PROVIDER = "huggingface"
FASTAPI_URL = "http://localhost:8000"