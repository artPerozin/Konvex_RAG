import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from infra.database.DatabaseConnection import DatabaseConnection
from domain.DocumentsController import DocumentsController

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY não definido no .env")

client = OpenAI(api_key=openai_api_key)

db = DatabaseConnection()
documents_controller = DocumentsController(db_connection=db, embedding_dim=384)

model = SentenceTransformer('all-MiniLM-L6-v2')

def recuperar_contexto(query: str, top_k: int = 3) -> list[str]:
    query_embedding = model.encode([query]).astype(np.float32)
    return documents_controller.search(query_embedding, top_k)

def answer_question(query: str) -> str:
    contexto = recuperar_contexto(query)
    prompt = f"Contexto:\n{chr(10).join(contexto)}\n\nPergunta: {query}\nResposta:"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Responda com base apenas no contexto fornecido."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    while True:
        pergunta = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        if pergunta.lower() in ("sair", "exit", "quit"):
            break
        resposta = answer_question(pergunta)
        print("\nResposta gerada:\n", resposta)
