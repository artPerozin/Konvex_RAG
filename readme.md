# IA_KONVEX_BACKEND

Backend em Python para um sistema de busca e resposta baseado em embeddings de documentos, usando **FastAPI**, **FAISS** e **OpenAI GPT**.

---

## 🚀 Visão Geral

Este projeto oferece uma API para:

- Armazenar documentos
- Indexá-los com FAISS
- Realizar buscas por similaridade com embeddings gerados por modelos de linguagem
- Responder perguntas usando o contexto recuperado via API do OpenAI GPT

---

## ✨ Funcionalidades

- 🔎 Indexação e busca rápida por similaridade usando **FAISS**
- 🧠 Geração de embeddings com `sentence-transformers`
- 🌐 API REST com **FastAPI** para gerenciar documentos e perguntas
- 🤖 Integração com **OpenAI GPT** para respostas contextuais
- 🔐 Uso de variáveis de ambiente para segurança e configuração

---

## 🛠 Tecnologias Utilizadas

- Python 3.11+
- FastAPI
- FAISS
- sentence-transformers
- OpenAI Python SDK (`openai`)
- python-dotenv
- requests

---

## 📋 Pré-requisitos

- Python 3.11 ou superior
- Virtualenv (recomendado)
- Conta e API Key da OpenAI

---

## ⚙️ Instalação

1. **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/IA_KONVEX_BACKEND.git
    cd IA_KONVEX_BACKEND
    ```

2. **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv

    # Windows
    .venv\Scripts\activate

    # Linux/macOS
    source .venv/bin/activate
    ```

3. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Como Rodar

Para iniciar o servidor FastAPI localmente:

```bash
uvicorn rag_api:app --reload
```

## 📚 Endpoints Principais

- `GET /documents/` — Lista os documentos indexados
- `POST /documents/` — Adiciona um novo documento (com embedding automático)
- `POST /ask/` — Envia uma pergunta e recebe a resposta gerada baseada nos documentos

---

## 🧩 Como Funciona a Busca

1. Os documentos são carregados e seus embeddings são gerados pelo modelo `sentence-transformers`.
2. Os embeddings são indexados com **FAISS** para busca rápida.
3. Ao fazer uma pergunta, o sistema gera o embedding da consulta.
4. Busca os documentos mais similares no índice FAISS.
5. Monta um prompt com esses documentos e envia para a API **OpenAI GPT** para gerar uma resposta contextualizada.

---

## 📁 Estrutura do Projeto

```
rag_api.py            # Arquivo principal da API FastAPI
DocumentsController.py # Gerencia documentos e busca FAISS
.env                  # Variáveis de ambiente
requirements.txt      # Dependências Python
README.md             # Este arquivo
```