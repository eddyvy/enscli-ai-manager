import os

from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.astra_db import AstraDBVectorStore


class IndexManager:
    __instance = None

    def __init__(self):
        if not IndexManager.__instance is None:
            raise Exception("There Can Be Only One IndexManager!!!")
        IndexManager.__instance = self

        self.__project_index = {}

    @staticmethod
    def instance():
        if IndexManager.__instance is None:
            IndexManager()
        return IndexManager.__instance

    def save_index(self, project_name: str, index: VectorStoreIndex):
        self.__project_index[project_name] = index

    def load_save_index(self, project_name: str, embed_model_name: str, embedding_dimension: int):
        # Astra DB config
        astra_endpoint = os.environ["ASTRA_DB_ENDPOINT"]
        astra_token = os.environ["ASTRA_DB_TOKEN"]

        if not astra_endpoint or not astra_token:
            raise ValueError("Astra DB config not found")

        embed_model = OpenAIEmbedding(
            model=embed_model_name
        )

        # Astra DB vector store
        vector_store = AstraDBVectorStore(
            token=astra_token,
            api_endpoint=astra_endpoint,
            collection_name=project_name,
            # Dimensions: https://docs.datastax.com/en/astra-db-serverless/get-started/concepts.html
            embedding_dimension=embedding_dimension,
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )

        self.save_index(project_name, index)

    def get_index(self, project_name: str, embed_model_name: str, embedding_dimension: int) -> VectorStoreIndex:
        if project_name not in self.__project_index:
            self.load_save_index(
                project_name, embed_model_name, embedding_dimension)
        return self.__project_index.get(project_name)
