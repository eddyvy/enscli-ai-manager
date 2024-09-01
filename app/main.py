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
):
    try:
        content: str = file.decode("utf-8")
        execute_embedding(content, project)

        return {"success": True}
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


class QueryRequest(BaseModel):
    query: str
    top_k: int


@app.post("/{project}/query")
async def post_project_query(project: str, req: QueryRequest, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        query = req.query
        top_k = req.top_k

        if query is None or top_k is None:
            raise HTTPException(
                status_code=400, detail="Missing 'query' or 'top_k' request body params")

        chunks = index_query(project, query, top_k)

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


@app.post("/{project}/chat")
async def post_project_char(project: str, req: ChatRequest, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        return send_message(project, req.message, req.top_k, req.session_id, req.model, req.temperature)
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
