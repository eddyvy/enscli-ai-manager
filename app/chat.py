from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate

from chat_manager import ChatManager
from index_manager import IndexManager


custom_prompt = """
You are a chatbot, able to have normal interactions, as well as talk about the provided context.
Here are the relevant documents for the context:

```markdown
{context_str}
```

Instruction: Use the previous chat history, or the context above, to interact and help the user.
"""


def send_message(
    project_name: str,
    message: str,
    top_k: int,
    session_id: str,
    model: str,
    temperature: float,
    embed_model: str,
    embedding_dimension: int
) -> str:
    llm = OpenAI(model=model, temperature=temperature)
    index: VectorStoreIndex = IndexManager.instance().get_index(
        project_name, embed_model, embedding_dimension)

    chat_memory = ChatManager.instance().get_chat_memory(session_id)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=chat_memory,
        context_prompt=custom_prompt,
        # context_prompt=custom_prompt,
        similarity_top_k=top_k,
        vector_store_query_mode="mmr",
        vector_store_kwargs={"mmr_prefetch_factor": 4},
        verbose=False
    )

    return chat_engine.chat(message)
