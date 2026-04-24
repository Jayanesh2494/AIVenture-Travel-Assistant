from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME, MODEL_PROVIDER


def get_qdrant_client():
    if not QDRANT_HOST:
        raise ValueError("QDRANT_HOST is missing")

    return QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY
    )


def get_vector_size():
    """
    Dynamically set vector size based on embedding model
    """
    if MODEL_PROVIDER == "openai":
        return 1536   # text-embedding-3-small
    elif MODEL_PROVIDER == "huggingface":
        return 384    # all-MiniLM-L6-v2
    else:
        raise ValueError("Invalid MODEL_PROVIDER")


def init_qdrant():
    client = get_qdrant_client()

    try:
        collections = client.get_collections().collections
        existing = [col.name for col in collections]

        if COLLECTION_NAME not in existing:
            print(f"📦 Creating collection '{COLLECTION_NAME}'...")

            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=get_vector_size(),
                    distance=Distance.COSINE
                ),
            )
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' already exists")

    except Exception as e:
        print("❌ Qdrant initialization failed:", str(e))
        raise

    return client