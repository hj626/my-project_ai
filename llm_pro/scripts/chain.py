# LLM 연결 클래스

# scripts/chain.py 예시
class SajuLLMChain:
    def __init__(self, vector_db, model_type="gemma"):
        if model_type == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        else:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model="gemma")
        
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 3})