import os
from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.astra_db import AstraDBVectorStore

from index_manager import IndexManager


def execute_embedding(content: str, project_name: str, embed_model_name: str, buffer_size: int, breakpoint_percentile_threshold: int, embedding_dimension: int) -> None:
    # Astra DB config
    astra_endpoint = os.environ["ASTRA_DB_ENDPOINT"]
    astra_token = os.environ["ASTRA_DB_TOKEN"]

    if not astra_endpoint or not astra_token:
        raise ValueError("Astra DB config not found")

    # Choose embedding model.
    embed_model = OpenAIEmbedding(
        model=embed_model_name,
    )

    # Astra DB vector store
    vector_store = AstraDBVectorStore(
        token=astra_token,
        api_endpoint=astra_endpoint,
        collection_name=project_name,
        # Dimensions: https://docs.datastax.com/en/astra-db-serverless/get-started/concepts.html
        embedding_dimension=embedding_dimension,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Process the content into nodes
    documents = [Document(text=content)]
    splitter = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # Create index and store it
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        verbose=True
    )

    IndexManager.instance().save_index(project_name, index)
