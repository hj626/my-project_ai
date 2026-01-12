import gradio as gr
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

from llm_handler import MovieCriticLLM

# 모델 및 토크나이저 로드
model = tf.keras.models.load_model("imdb_model.keras")
tokenizer = joblib.load("imdb_tokenizer.pkl")
llm = MovieCriticLLM()

def predict(text):
    
    if not text.strip():
        return "입력 없음", "리뷰를 입력해 주세요."
    
    
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    
    # 리뷰 감정 에측 및 확률 계산
    pred = int(model.predict(padded)[0][0] > 0.5)
    
    # 확률값(0~1 사이)을 직접 가져옵니다.
    prob = float(model.predict(padded)[0][0])
    
    # 함수는 (텍스트, 점수) 두 개만 받으므로 prob만 넘겨주는 게 가장 정확합니다.
    explanation = llm.analyze_review(text, prob)
    return ("긍정 😊" if pred == 1 else "부정 ☹️"), explanation


# --- 화면 구성 시작 ---
with gr.Blocks(title="IMDB 감정 분석 딥러닝 + LLM") as demo:
    gr.Markdown("# 🎬 IMDB 감정 분석 딥러닝 + LLM")
    
    with gr.Row():
        # 왼쪽: 입력창 및 결과창 (비중 3) / 인터페이스였던 부분이야.
        with gr.Column(scale=3):
            input_text = gr.Textbox(lines=5, label="영화 리뷰 입력", placeholder="여기에 영어로 리뷰를 써보세요.")
            
            with gr.Row():
                clear_btn = gr.Button("지우기")
                submit_btn = gr.Button("분석하기", variant="primary")
            
            output_label = gr.Textbox(label="감정 분류 결과")
            output_explanation = gr.Textbox(label="Gemma LLM 설명", lines=15) # 3배 크게 설정
           
            
        # 오른쪽: 쉬운 영어 예시 버튼
        with gr.Column(scale=1):
            gr.Markdown("### 📝 클릭해서 입력하기 (Example)")
            # 중학교 수준의 쉬운 영어 문장들
            btn1 = gr.Button("Very good movie!")
            btn2 = gr.Button("It was so boring.")
            btn3 = gr.Button("The actors were great.")

    # 버튼 기능 연결
    submit_btn.click(fn=predict, inputs=input_text, outputs=[output_label, output_explanation])
    clear_btn.click(lambda: "", None, input_text)
    
    # 예시 버튼을 누르면 해당 영어가 입력창에 들어갑니다.
    btn1.click(lambda: "This movie is very good! I love it.", None, input_text)
    btn2.click(lambda: "It was so boring. I want my money back.", None, input_text)
    btn3.click(lambda: "The actors were great. The story was beautiful.", None, input_text)

demo.launch()



# # 인터페이스 설정
# gr.Interface(
#     fn=predict,
#     inputs=gr.Textbox(lines=5, label="영화 리뷰 입력"),
#     outputs=[
#         gr.Textbox(label="감정 분류 결과"),
#         gr.Textbox(label="Gemma LLM 설명", lines=15)
#     ],
#     title="IMDB 감정 분석 딥러닝 + LLM",
#     submit_btn="분석하기",
#     clear_btn="지우기"
# ).launch()
