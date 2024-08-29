import os
from typing import List

from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.astra_db import AstraDBVectorStore


def index_query(project_name: str, query: str, top_k: int, model: str) -> List[str]:
    # Astra DB config
    astra_endpoint = os.environ["ASTRA_DB_ENDPOINT"]
    astra_token = os.environ["ASTRA_DB_TOKEN"]

    if not astra_endpoint or not astra_token:
        raise ValueError("Astra DB config not found")

    embed_model = OpenAIEmbedding(
        model=model
    )

    # Astra DB vector store
    vector_store = AstraDBVectorStore(
        token=astra_token,
        api_endpoint=astra_endpoint,
        collection_name=project_name,
        # Dimensions: https://docs.datastax.com/en/astra-db-serverless/get-started/concepts.html
        embedding_dimension=1536,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    retriever = index.as_retriever(
        vector_store_query_mode="mmr",
        similarity_top_k=top_k,
        vector_store_kwargs={"mmr_prefetch_factor": 4}
    )
    nodes = retriever.retrieve(query)
    return [node.get_content() for node in nodes]
