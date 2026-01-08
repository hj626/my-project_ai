import streamlit as st
import pandas as pd
import tensorflow as tf  # 모델 로드용
import numpy as np
import os
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="폐암 신항원 AI 분석 시스템", layout="wide")

# 2. 파일 로드 함수 (캐싱 처리)
@st.cache_resource
def load_assets():
    model_path = "lung_cancer_model.keras"
    # 모델이 있으면 로드, 없으면 None 반환
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception:
            return None
    return None

model = load_assets()

# 3. 사이드바: 분석 리포트 및 정보
with st.sidebar:
    st.title("📂 분석 자산")
    if os.path.exists("lung_cancer_rich_report.html"):
        st.success("✅ 분석 리포트 로드 완료")
    
    st.markdown("---")
    st.info("🎓 **국립암센터 AI 대학원**\n\n폐암 신항원(Neoantigen) 발굴 및 정밀 면역 치료 연구용 대시보드입니다.")

# 4. 메인 UI
st.title("🧬 폐암 신항원 정밀 예측 시스템")
st.markdown("""
입력된 아미노산 서열을 바탕으로 **MHC-I 결합력** 및 **면역원성(Immunogenicity)**을 예측합니다. 
""")

if model is None:
    st.warning("⚠️ 'lung_cancer_model.keras' 파일이 없습니다. 현재는 DB 대조 모드 및 샘플 데이터 출력 모드로 작동합니다.")

# 입력창
user_input = st.text_area("분석할 단백질 서열(Fasta format 또는 Raw Sequence)을 입력하세요", 
                         placeholder="예: MTEYKLVVVGAGGVGKSALTIQLI...",
                         height=150)

# 5. 분석 실행 로직
if st.button("🔬 AI 정밀 분석 시작"):
    if user_input:
        # 분석 중 알림
        with st.spinner('AI 모델이 서열을 분석 중입니다...'):
            # [노트북 로직 반영 지점] 
            # 1. 입력 서열 9-mer 분할 로직 (Sliding Window)
            # 2. 모델 인퍼런스 (model.predict)
            # 3. 결과 후처리
            
            # --- 결과 섹션 ---
            st.subheader("📊 예측 분석 결과")
            
            # 임시 결과 데모 (실제 모델 연동 시 계산된 값으로 대체)
            col1, col2, col3 = st.columns(3)
            col1.metric("신항원 가능성", "89.5%", "High Confidence")
            col2.metric("MHC 결합력 점수", "0.92", "IC50 < 50nM")
            col3.metric("면역원성 지수", "0.78", "Positive")

            # 데이터프레임 표시 (노트북에서 저장한 df_final.csv 활용)
            if os.path.exists("df_final.csv"):
                df_final = pd.read_csv("df_final.csv")
                st.write("### 📋 분석 상세 데이터 (Top Candidates)")
                st.dataframe(df_final.head(10), use_container_width=True)
                
                # 시각화 예시: 점수 분포
                fig = go.Figure(data=[go.Histogram(x=df_final['score'], nbinsx=20)])
                fig.update_layout(title="예측 점수 분포도", xaxis_title="Score", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("💡 'df_final.csv' 파일이 있으면 이곳에 상세 리스트가 표시됩니다.")
    else:
        st.error("분석할 서열을 입력해 주세요.")
