# scripts/chain.py
import langchain

class SajuLLMChain:
    def __init__(self, vector_db, model_type="gemma"):
        if model_type == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        else:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model="gemma")
        
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    def get_response(self, question):
        # 최신 경로인 langchain.chains.retrieval 대신 공식 라이브러리 내장 함수 사용
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        from langchain_core.prompts import ChatPromptTemplate

        # AI가 참고할 기본 지침(프롬프트) 설정
        prompt = ChatPromptTemplate.from_template("""
        당신은 사주 전문가입니다. 아래 제공된 문맥(Context)만을 사용하여 성격과 특징을 분석하세요.
        
        <context>
        {context}
        </context>
        
        질문: {input}
        """)

        # 문서 결합 체인과 리트리벌 체인 생성
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

        # 결과 반환
        response = retrieval_chain.invoke({"input": question})
        return {"result": response["answer"]}