from typing import Annotated
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form
import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
import logging
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasicCredentials
from pydantic import BaseModel
from chat import send_message
from embed import execute_embedding
from query import index_query

load_dotenv()

security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    user_name = os.getenv('BASIC_AUTH_USERNAME')
    user_password = os.getenv('BASIC_AUTH_PASSWORD')

    if not user_name or not user_password:
        raise HTTPException(
            status_code=500, detail="Basic Auth config not found")

    if not credentials.username or not credentials.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    correct_username = secrets.compare_digest(credentials.username, user_name)
    correct_password = secrets.compare_digest(
        credentials.password, user_password)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


app = FastAPI()


@app.get("/health")
def get_health():
    return {"status": "up"}


@app.post("/{project}/embed")
async def create_file(
    project: str,
    file: Annotated[bytes, File()],
    embed_model: str = Form("text-embed-ada-002"),
    buffer_size: int = Form(3),
    breakpoint_percentile_threshold: int = Form(85),
    embedding_dimension: int = Form(1536),
    credentials: HTTPBasicCredentials = Depends(verify_credentials)
):
    try:
        content: str = file.decode("utf-8")
        execute_embedding(content, project, embed_model,
                          buffer_size, breakpoint_percentile_threshold, embedding_dimension)

        return {"success": True}
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


class QueryRequest(BaseModel):
    query: str
    top_k: int
    embed_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536


@app.post("/{project}/query")
async def post_project_query(project: str, req: QueryRequest, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        query = req.query
        top_k = req.top_k
        embed_model = req.embed_model
        embedding_dimension = req.embedding_dimension

        if query is None or top_k is None:
            raise HTTPException(
                status_code=400, detail="Missing 'query' or 'top_k' request body params")

        chunks = index_query(project, query, top_k,
                             embed_model, embedding_dimension)

        return chunks
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    top_k: int = 3
    embed_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536


@app.post("/{project}/chat")
async def post_project_char(project: str, req: ChatRequest, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        return send_message(project, req.message, req.top_k, req.session_id, req.model, req.temperature, req.embed_model, req.embedding_dimension)
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
