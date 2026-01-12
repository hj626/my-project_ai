import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from llm_handler import MovieCriticLLM

# 모델 및 토크나이저 로드
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('imdb_model.h5')
    tokenizer = joblib.load('imdb_tokenizer.pkl')
    return model, tokenizer

model, tokenizer = load_resources()
llm = MovieCriticLLM()

st.title("🎬 영화 리뷰 AI 감정 분석기")
st.write("리뷰를 입력하면 딥러닝이 감정을 분석하고, Gemma AI가 비평을 작성합니다.")

user_review = st.text_area("영어 리뷰를 입력하세요 (IMDB 데이터 기준):", "The cinematography was great, but the plot was boring.")

if st.button("분석 시작"):
    # 1. 딥러닝 예측
    seq = tokenizer.texts_to_sequences([user_review])
    pad = pad_sequences(seq, maxlen=200)
    prediction = model.predict(pad)[0][0]
    
    # 2. 결과 출력
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("긍정 확률", f"{prediction:.2%}")
    with col2:
        sentiment_label = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
        st.subheader(f"결과: {sentiment_label}")

    # 3. LLM 비평 생성
    with st.spinner("Gemma AI가 비평을 작성 중입니다..."):
        critic_view = llm.analyze_review(user_review, prediction)
        st.write("🤖 **AI 비평가 Gemma의 한마디:**")
        st.info(critic_view)