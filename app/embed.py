import os
from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.astra_db import AstraDBVectorStore

from index_manager import IndexManager


def execute_embedding(content: str, project_name: str, model: str) -> None:
    # Astra DB config
    astra_endpoint = os.environ["ASTRA_DB_ENDPOINT"]
    astra_token = os.environ["ASTRA_DB_TOKEN"]

    if not astra_endpoint or not astra_token:
        raise ValueError("Astra DB config not found")

    # Choose embedding model.
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

    # Process the content into nodes
    documents = [Document(text=content)]
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # Create index and store it
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    IndexManager.instance().save_index(project_name, index)
