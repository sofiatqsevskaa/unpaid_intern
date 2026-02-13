import chromadb
from chromadb.config import Settings
from typing import Dict, Any, List
import uuid
import os
from logger import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger("vector_storage")


class VectorStorage:
    _embedding_function = None
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                  'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                  'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                  'to', 'from', 'in', 'on', 'at', 'by', 'with', 'without', 'after', 'before'}

    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.environ.get("VECTOR_DB_PATH", "./vector_db")

        logger.info(f"Initializing VectorStorage at path: {persist_directory}")

        if VectorStorage._embedding_function is None:
            logger.info(
                "Loading embedding model 'all-MiniLM-L6-v2' into memory...")
            from chromadb.utils import embedding_functions
            VectorStorage._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Embedding model loaded")

        self.embedding_function = VectorStorage._embedding_function
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def create_collection(self, user_id: str):
        name = f"user_{user_id}_docs"
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function
        )

    def document_exists(self, user_id: str, content: str) -> bool:
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        try:
            collection = self.create_collection(user_id)
            results = collection.get(
                where={"content_hash": content_hash},
                limit=1
            )
            if results and results["ids"]:
                logger.warning(
                    f"Duplicate detected"
                    f"(hash: {content_hash[:12]}..., "
                    f"existing doc_id: {results['metadatas'][0].get('document_id', 'unknown')})"
                )
                return True
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
        return False

    def add_document(self, user_id: str, document_name: str, content: str, metadata: Dict[str, Any]):
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        logger.info(
            f"Adding document '{document_name}' for user {user_id} | content_hash: {content_hash[:12]}...")

        if self.document_exists(user_id, content):
            logger.warning(
                f"Skipping '{document_name}' (duplicate)")
            return {"document_id": None, "chunks_processed": 0, "skipped": True, "reason": "duplicate"}

        collection = self.create_collection(user_id)
        document_id = str(uuid.uuid4())
        chunks = self.text_splitter.split_text(content)

        logger.info(
            f"Chunked '{document_name}' into {len(chunks)} chunks — vectorizing with all-MiniLM-L6-v2...")

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{document_id}_{i}")
            docs.append(chunk)
            metas.append({
                "document_id": document_id,
                "document_name": document_name,
                "user_id": user_id,
                "chunk_index": i,
                "content_hash": content_hash,
                "tags": ", ".join(metadata.get("tags") or []),
                "description": metadata.get("description") or "",
                "original_filename": metadata.get("original_filename") or document_name,
                "content_type": metadata.get("content_type") or "text/plain",
            })

        collection.add(ids=ids, documents=docs, metadatas=metas)
        logger.info(
            f"Vectorized and stored '{document_name}', {len(chunks)} chunks written")

        return {"document_id": document_id, "chunks_processed": len(chunks), "skipped": False}

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            collection = self.create_collection(user_id)
            all_items = collection.get(include=["metadatas"])
        except Exception as e:
            logger.error(
                f"Failed to list vector documents for user {user_id}: {e}")
            return []

        seen = {}
        for meta in all_items.get("metadatas") or []:
            doc_id = meta.get("document_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "document_name": meta.get("document_name"),
                    "original_filename": meta.get("original_filename"),
                    "content_hash": meta.get("content_hash"),
                }

        documents = list(seen.values())
        logger.info(
            f"Listed {len(documents)} unique documents from vector store for user {user_id}")
        return documents

    def query(self, user_id: str, query_text: str, top_k: int = 5):
        logger.info(
            f"Querying vector store for user {user_id}, query: '{query_text}', top_k: {top_k}")

        collection = self.create_collection(user_id)

        words = [word.lower() for word in query_text.split()
                 if word.lower() not in self.stop_words and len(word) > 2]

        phrases = []
        word_list = query_text.split()
        for i in range(len(word_list) - 1):
            phrase = f"{word_list[i]} {word_list[i+1]}".lower()
            if all(word.lower() not in self.stop_words for word in [word_list[i], word_list[i+1]]):
                phrases.append(phrase)

        search_terms = words + phrases
        logger.debug(f"search terms: {search_terms}")

        unique_results = {}

        for term in search_terms:
            logger.debug(f"Searching with term: '{term}'")

            try:
                results = collection.query(
                    query_texts=[term],
                    n_results=top_k
                )

                if results and results["documents"] and results["documents"][0]:
                    for i in range(len(results["documents"][0])):
                        content = results["documents"][0][i]
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {
                        }

                        doc_id = metadata.get('document_id', 'unknown')
                        chunk_index = metadata.get('chunk_index', i)
                        unique_key = f"{doc_id}_{chunk_index}"

                        distance = results["distances"][0][i] if results["distances"] else 1.0

                        if unique_key not in unique_results or distance < unique_results[unique_key]["distance"]:
                            unique_results[unique_key] = {
                                "content": content,
                                "metadata": metadata,
                                "distance": distance,
                                "term": term
                            }
            except Exception as e:
                logger.error(f"Error querying with term '{term}': {e}")
                continue
        output = list(unique_results.values())
        output.sort(key=lambda x: x["distance"])

        output = output[:top_k]

        logger.info(
            f"Vector query returned {len(output)} unique results (from {len(search_terms)} search terms)")
        return output
