from src.embeddings import get_embeddings
from src.config import COLLECTION_NAME
from src.vectorstores import get_qdrant_client


def retrieve_docs(query: str, top_k=5):
    client = get_qdrant_client()

    query_vector = get_embeddings([query])[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    texts = [point.payload.get("text", "") for point in results.points]
    print("Retrieved texts:", texts)
    print("Retrieved results:", results)
    client = get_qdrant_client()
    count = client.count(collection_name=COLLECTION_NAME)
    print("Total vectors in DB:", count)

    return [point.payload.get("text", "") for point in results.points]