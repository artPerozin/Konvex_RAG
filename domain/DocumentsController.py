import numpy as np
import faiss
from infra.database.DatabaseConnection import DatabaseConnection

class DocumentsController:
    def __init__(self, db_connection: DatabaseConnection, embedding_dim=384):
        self.db = db_connection
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.load_documents()

    def load_documents(self):
        self.db.cur.execute("SELECT content, embedding FROM documents")
        rows = self.db.cur.fetchall()
        self.documents = []
        embeddings = []
        for content, embedding in rows:
            self.documents.append(content)
            embeddings.append(np.array(embedding, dtype=np.float32))
        if embeddings:
            self.index.reset()
            self.index.add(np.vstack(embeddings).astype(np.float32))

    def add_document(self, content: str, embedding: np.ndarray):
        sql = "INSERT INTO documents (content, embedding) VALUES (%s, %s) RETURNING id"
        self.db.cur.execute(sql, (content, embedding.tolist()))
        id_ = self.db.cur.fetchone()[0]
        self.documents.append(content)
        self.index.add(np.array([embedding]))
        return id_

    def search(self, query_embedding, top_k: int):
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
