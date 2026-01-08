# 메인 실행 로직

import os
from dotenv import load_dotenv
from scripts.preprocess import SajuPreprocessor
from scripts.embed import SajuVectorStore
from scripts.chain import SajuLLMChain

def main():
    # 0. 환경 변수(.env) 로드
    load_dotenv()

    # 1. 데이터 가공 로직 (Preprocess)
    # data 폴더의 사주 텍스트를 읽어 조각냅니다.
    preprocessor = SajuPreprocessor("./data/saju_data.txt")
    chunks = preprocessor.process()
    print(f"✅ 데이터 분할 완료: {len(chunks)}개 조각")

    # 2. 지식 창고 생성 로직 (Vector Store)
    # Ollama의 Gemma를 사용하여 FAISS DB를 만듭니다.
    vs = SajuVectorStore(model_name="gemma")
    vector_db = vs.create_and_save(chunks, "vector_db/saju_index")
    print("✅ 벡터 지식 창고 생성 및 저장 완료")

    # 3. 질문 답변 로직 (Chain)
    # 여기에서 모델을 "gemma" 혹은 "gemini"로 선택할 수 있습니다.
    # 우선 실습을 위해 gemma를 먼저 사용합니다.
    saju_ai = SajuLLMChain(vector_db, model_type="gemma") 
    
    # 4. 테스트 질문 실행
    question = "갑목(甲木) 성격의 장점과 리더십에 대해 설명해줘."
    print(f"\n질문: {question}")
    
    result = saju_ai.get_response(question)
    
    print("\n" + "="*50)
    print("[사주 AI 전문가의 답변]")
    print(result['result'])
    print("="*50)

if __name__ == "__main__":
    main()