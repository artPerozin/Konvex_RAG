from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

from DatabaseConnection import DatabaseConnection
from DocumentsController import DocumentsController

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DATABASE")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')
db_connection = DatabaseConnection(DB_USER, DB_PASSWORD, DB_NAME, DB_HOST, DB_PORT)
doc_store = DocumentsController(db_connection)

class DocumentIn(BaseModel):
    content: str

class DocumentOut(BaseModel):
    id: int
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 2

@app.get("/documents/", response_model=List[DocumentOut])
def get_all_documents():
    try:
        db_connection.cur.execute("SELECT id, content FROM documents")
        rows = db_connection.cur.fetchall()
        documents = [{"id": row[0], "content": row[1]} for row in rows]
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload_pdf/", response_model=List[DocumentOut])
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Somente arquivos PDF são permitidos.")

    contents = file.file.read()
    pdf = fitz.open(stream=contents, filetype="pdf")
    extracted_texts = []
    for page in pdf:
        text = page.get_text()
        if text.strip():
            extracted_texts.append(text.strip())
    pdf.close()

    if not extracted_texts:
        raise HTTPException(status_code=400, detail="O PDF não contém texto extraível.")

    result = []
    for chunk in extracted_texts:
        embedding = model.encode([chunk])[0].astype(np.float32)
        id_ = doc_store.add_document(chunk, embedding)
        result.append({"id": id_, "content": chunk})

    return result

@app.post("/ask/")
def ask_question(req: QueryRequest):
    if not doc_store.documents:
        raise HTTPException(status_code=400, detail="Nenhum documento carregado ainda.")

    query_embedding = model.encode([req.question])[0].astype(np.float32)
    context = doc_store.search(query_embedding, top_k=req.top_k)

    if not context:
        return JSONResponse(content={"answer": "Nenhum contexto relevante encontrado."})

    prompt = "Contexto:\n" + "\n".join(context) + f"\n\nPergunta: {req.question}\nResposta:"

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Responda com base apenas no contexto fornecido."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7,
    )

    return {"answer": response.choices[0].message.content.strip()}

