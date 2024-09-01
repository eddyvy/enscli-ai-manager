from typing import List
from index_manager import IndexManager


def index_query(project_name: str, query: str, top_k: int, model: str) -> List[str]:
    index = IndexManager.instance().get_index(project_name, model)

    retriever = index.as_retriever(
        vector_store_query_mode="mmr",
        similarity_top_k=top_k,
        vector_store_kwargs={"mmr_prefetch_factor": 4}
    )
    nodes = retriever.retrieve(query)
    return [node.get_content() for node in nodes]
