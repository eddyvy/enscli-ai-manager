from typing import Annotated
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form
import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logging
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasicCredentials
from pydantic import BaseModel
from embed import execute_embedding
from query import index_query
from llama_index.embeddings.openai import OpenAIEmbeddingModelType

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    user_name = os.getenv('BASIC_AUTH_USERNAME')
    user_password = os.getenv('BASIC_AUTH_PASSWORD')

    correct_username = secrets.compare_digest(credentials.username, user_name)
    correct_password = secrets.compare_digest(credentials.password, user_password)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


load_dotenv()

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "up"}


@app.post("/{project}/embed")
async def create_file(
    project: str,
    file: Annotated[bytes, File()],
    model: str = Form(OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL),
):
    content: str = file.decode("utf-8")
    execute_embedding(content, project, model)

    return { "success": True }

class QueryRequest(BaseModel):
    query: str
    top_k: int
    model: str = OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL

@app.post("/{project}/query")
async def post_project_query(project: str, req: QueryRequest, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        query = req.query
        top_k = req.top_k
        model = req.model

        if query is None or top_k is None:
            raise HTTPException(status_code=400, detail="Missing 'query' or 'top_k' or 'model' request body params")

        chunks = index_query(project, query, top_k, model)

        return chunks
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    