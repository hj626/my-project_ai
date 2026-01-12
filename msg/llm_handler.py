import requests
from langchain_ollama import OllamaLLM


class MovieCriticLLM:
    """Ollama의 Gemma를 사용하여 영화 리뷰를 전문적으로 분석하는 클래스"""
    
    def __init__(self, model_name="gemma2:2b"):
        self.url = "http://localhost:11434/api/generate"
        self.model_name = model_name

    def analyze_review(self, review_text, sentiment_score):
        """딥러닝 결과와 원문을 LLM에게 전달하여 비평 작성"""
        sentiment = "긍정적" if sentiment_score > 0.5 else "부정적"
        
        prompt = f"""
        너는 전문 영화 비평가야. 아래 영화 리뷰에 대한 딥러닝 분석 결과가 '{sentiment}'으로 나왔어.
        리뷰 원문과 이 결과를 바탕으로, 이 영화가 왜 이런 평가를 받았을지 전문적인 비평을 한국어로 3문장 이내로 작성해줘.
        
        리뷰 원문: {review_text[:300]}...
        분석 결과: {sentiment} (확률: {sentiment_score:.2f})
        """
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.url, json=payload)
            return response.json().get('response', "분석 실패")
        except:
            return "Ollama 서버가 응답하지 않습니다."