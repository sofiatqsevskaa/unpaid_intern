import chromadb
from chromadb.config import Settings
from typing import Dict, Any
import uuid
import os

PASSWORD = os.environ.get("PASSWORD")


class VectorStorage:

    _embedding_function = None

    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.environ.get("VECTOR_DB_PATH", "./vector_db")
        if VectorStorage._embedding_function is None:
            from chromadb.utils import embedding_functions
            VectorStorage._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

        self.embedding_function = VectorStorage._embedding_function

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

    def create_collection(self, user_id: str):
        name = f"user_{user_id}_docs"
        print("creating/getting collection:", name)
        return self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_function
        )

    def add_document(self, user_id: str, document_name: str, content: str, metadata: Dict[str, Any]):
        collection = self.create_collection(user_id)
        document_id = str(uuid.uuid4())
        chunks = self.text_splitter.split_text(content)
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{document_id}_{i}")
            docs.append(chunk)
            metas.append({
                "document_id": document_id,
                "document_name": document_name,
                "user_id": user_id,
                **metadata
            })
        collection.add(ids=ids, documents=docs, metadatas=metas)
        return {"document_id": document_id, "chunks_processed": len(chunks)}

    def query(self, user_id: str, query_text: str, top_k: int = 5):
        collection = self.create_collection(user_id)
        results = collection.query(query_texts=[query_text], n_results=top_k)
        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None
            })
        return output
