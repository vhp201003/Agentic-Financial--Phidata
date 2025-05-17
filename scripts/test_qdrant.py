from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
results = client.scroll(collection_name="financial_docs", limit=4190)
companies = list(set([hit.payload.get("company", "") for hit in results[0]]))
print(companies, len(companies))