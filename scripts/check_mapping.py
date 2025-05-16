# scripts/check_mapping.py
from qdrant_client import QdrantClient
from utils.company_mapping import check_mapping_integrity
from config.env import QDRANT_HOST, QDRANT_PORT

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collection_name = "financial_docs"
    if check_mapping_integrity(client, collection_name):
        print("Mapping is consistent")
    else:
        print("Mapping inconsistencies found. Check logs for details.")

if __name__ == "__main__":
    main()