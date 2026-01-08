import os
import re
from dotenv import load_dotenv
from scripts.preprocess import SajuPreprocessor
from scripts.embed import SajuVectorStore
from scripts.chain import SajuLLMChain
from scripts.manse import SajuCalculator # 만세력 클래스 임포트 확인

def main():
    load_dotenv()

    # 1. 데이터 가공 및 2. 벡터 DB 로드
    preprocessor = SajuPreprocessor("./data/saju_data.txt")
    chunks = preprocessor.process()
    vs = SajuVectorStore(model_name="gemma")
    vector_db = vs.create_and_save(chunks, "vector_db/saju_index")

    # 3. AI 체인 생성
    saju_ai = SajuLLMChain(vector_db, model_type="gemma") 
    
    # 4. 실시간 사용자 입력 (생년월일 받기)
    print("\n" + "*"*50)
    user_input = input("분석할 생년월일을 입력하세요 (예: 1987년 6월 26일): ")
    
    # 숫자만 추출 (예: [1987, 6, 26])
    nums = re.findall(r'\d+', user_input)
    
    if len(nums) >= 3:
        year, month, day = int(nums[0]), int(nums[1]), int(nums[2])
        
        # 5. 생년월일 분석 (manse.py 활용)
        calc = SajuCalculator()
        analysis = calc.analyze_five_elements(year, month, day)
        counts = analysis['counts']
        main_element = analysis['main_element']
        
        print(f"\n✨ 분석 결과: {year}년 {month}월 {day}일생은 '{main_element}'의 기운을 타고났습니다.")
        print(f"📊 오행 구성: 목({counts['목']}), 화({counts['화']}), 토({counts['토']}), 금({counts['금']}), 수({counts['수']})")

        # 6. AI에게 던질 종합 분석 질문 구성
        question = f"""
        이 사용자의 일간은 '{main_element}'이며, 오행 구성은 목:{counts['목']}, 화:{counts['화']}, 토:{counts['토']}, 금:{counts['금']}, 수:{counts['수']}입니다.
        이 구성의 특징과 성격 장단점을 'saju_data.txt' 내용을 참고해서 종합 분석해줘.
        """
        
        print(f"\n[조회 중] 사주 데이터를 바탕으로 종합 분석을 생성하고 있습니다...")
        result = saju_ai.get_response(question) #
        
        print("\n" + "="*50)
        print(f"[{main_element} 중심 - 사주 오행 종합 분석 결과]")
        print(result['result']) # AI의 답변 출력
        print("="*50)
    else:
        print("❌ 생년월일 형식이 올바르지 않습니다. (예: 1987 06 26)")

if __name__ == "__main__":
    main()