from storage.vector_storage import VectorStorage
from storage.graph_storage import GraphStorage


class StorageRepository:
    def __init__(self):
        self.vector = VectorStorage()
        _ = self.vector.embedding_function

        self.graph = GraphStorage()

    def add_to_vector(self, user_id, document_name, content, metadata):
        return self.vector.add_document(
            user_id=user_id,
            document_name=document_name,
            content=content,
            metadata=metadata
        )

    def add_to_graph(self, user_id, document_name, content, metadata):
        return self.graph.add_document(
            user_id=user_id,
            document_name=document_name,
            content=content,
            metadata=metadata
        )

    def query_vector(self, user_id, query_text, top_k=5):
        return self.vector.query(user_id, query_text, top_k)

    def query_graph(self, user_id, query_text):
        return self.graph.query(user_id, query_text)

    def close(self):
        self.graph.close()
