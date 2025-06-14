import pytest
import numpy as np
from unittest.mock import MagicMock

from domain.DocumentsController import DocumentsController

@pytest.fixture
def mock_db():
    mock_cursor = MagicMock()
    mock_connection = MagicMock()
    mock_connection.cur = mock_cursor
    return mock_connection

@pytest.fixture
def controller(mock_db):
    return DocumentsController(mock_db)

def test_load_documents_empty_db(mock_db):
    mock_db.cur.fetchall.return_value = []
    controller = DocumentsController(mock_db)
    assert controller.documents == []
    assert controller.index.ntotal == 0

def test_add_document(controller, mock_db):
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_db.cur.fetchone.return_value = [1]

    doc_id = controller.add_document("teste de conteúdo", mock_embedding)

    assert doc_id == 1
    assert controller.documents[-1] == "teste de conteúdo"
    assert controller.index.ntotal == 1

def test_search_returns_correct_documents(controller, mock_db):
    doc1 = np.ones(384, dtype=np.float32)
    doc2 = np.ones(384, dtype=np.float32) * 2

    mock_db.cur.fetchall.return_value = [
        ("conteúdo 1", doc1.tolist()),
        ("conteúdo 2", doc2.tolist())
    ]

    controller.load_documents()
    query = np.ones(384, dtype=np.float32)

    results = controller.search(query, top_k=1)

    assert len(results) == 1
    assert results[0] in controller.documents

def test_search_with_list_input(controller, mock_db):
    embedding = np.ones(384, dtype=np.float32)
    mock_db.cur.fetchall.return_value = [("conteúdo", embedding.tolist())]
    controller.load_documents()

    query_list = embedding.tolist()
    results = controller.search(query_list, top_k=1)

    assert isinstance(results, list)
    assert "conteúdo" in results
