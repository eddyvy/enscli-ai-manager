from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer


class ChatManager:
    __instance = None

    def __init__(self):
        if not ChatManager.__instance is None:
            raise Exception("There Can Be Only One ChatManager!!!")
        ChatManager.__instance = self

        self.__chat_store = SimpleChatStore()

    @staticmethod
    def instance():
        if ChatManager.__instance is None:
            ChatManager()
        return ChatManager.__instance

    def get_chat_memory(self, session_id):
        return ChatMemoryBuffer.from_defaults(
            token_limit=4000,
            chat_store=self.__chat_store,
            chat_store_key=session_id,
        )
