# 벡터저장소 클래스

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class SajuVectorStore:
    def __init__(self, model_name="gemma"):
        self.embeddings = OllamaEmbeddings(model=model_name) #

    def create_and_save(self, chunks, save_path):
        vector_db = FAISS.from_documents(chunks, self.embeddings) #
        vector_db.save_local(save_path) #
        return vector_db