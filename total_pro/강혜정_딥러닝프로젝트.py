import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import plotly.graph_objects as go

# --- [1] 페이지 설정 및 자산 로드 ---
st.set_page_config(page_title="폐암 신항원 AI 분석 시스템", page_icon="🧬", layout="wide")

@st.cache_resource
def load_assets():
    model_path = "lung_cancer_model.keras"
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception:
            return None
    return None

model = load_assets()

# --- [2] 사이드바: 전문 정보 및 검증 서열 ---
with st.sidebar:
    st.title("📂 분석 자산 및 참조")
    
    if os.path.exists("lung_cancer_rich_report.html"):
        st.success("✅ 전문 분석 리포트 준비 완료")
    
    st.markdown("---")
    st.subheader("🧬 검증된 신항원 사례")
    st.caption("실제 환자 데이터 기반 Positive 서열")
    positive_examples = {
        "LLDFVRFMG": "폐선암",
        "SLLMWITQV": "비소세포폐암",
        "KVLEYVIKV": "전이성 폐암"
    }
    for seq, desc in positive_examples.items():
        st.code(seq)
        st.caption(f"유형: {desc}")

    st.markdown("---")
    st.info("🎓 **국립암센터 AI 대학원**\n\n본 시스템은 폐암 신항원(Neoantigen) 발굴 및 맞춤형 면역 치료 연구를 위해 개발되었습니다.")

# --- [3] 메인 화면 레이아웃 ---
st.title("🛡️ AI 기반 환자 맞춤형 신항원 정밀 분석")
st.markdown("입력된 아미노산 서열을 분석하여 **MHC-I 결합력**과 **암 백신 후보 가능성**을 예측합니다.")
st.write("---")


st.write("---")
with st.expander("❓ 이 시스템의 분석 원리가 궁금하신가요? (국립암센터 AI 대학원 연구 가이드)"):
    st.markdown("### 🧬 AI가 신항원을 찾아내는 방법")
    
    col_a, col_b = st.columns([1, 1.5])
    
    with col_a:
        # 비유를 통한 설명
        st.info("""
        **1. MHC: 세포의 '검문소 쟁반'**
        세포는 내부의 단백질 조각을 MHC라는 '쟁반'에 담아 세포 표면에 내놓습니다. 
        면역세포는 이 쟁반을 검사하여 정상인지 암인지 판단합니다.
        """)
    
    with col_b:
        st.success(f"""
        **2. 소수성 잔기의 역할 (핵심 원리)**
        암세포 조각에 **류신(L)**이나 **발린(V)** 같은 **소수성 잔기**가 많을수록, 
        이 '쟁반'에 더 단단하고 안정적으로 고정됩니다. 
        안정적으로 고정될수록 면역세포에게 더 잘 노출되어 공격 대상이 될 확률이 높아집니다.
        """)

    st.write("---")
    st.markdown(f"""
    **3. AI 모델의 학습 내용**
    본 시스템의 AI는 수만 개의 암세포 서열 데이터를 통해 어떤 서열이 MHC 쟁반에 잘 올라가는지, 
    그리고 면역세포가 '낯선 적'으로 인식할 확률이 높은 패턴은 무엇인지 학습했습니다. 
    이를 통해 **새로운 암 백신 연구에 적합한 후보군을 정밀하게 추천**합니다.
    """)


if model is None:
    st.warning("⚠️ 'lung_cancer_model.keras' 모델 파일을 찾을 수 없습니다. 현재는 데모 모드로 작동합니다.")

# 서열 입력창 (긴 서열 입력 시 자동 분할 처리 로직 포함 예정)
user_input = st.text_area("분석할 아미노산 서열을 입력하세요 (9글자 권장)", 
                         placeholder="예: LLDFVRFMG (9글자 입력 시 정밀 분석 가능)",
                         height=100).upper().replace(" ", "").strip()

# --- [4] 분석 로직 ---
if st.button("🔬 AI 정밀 분석 시작", type="primary"):
    if not user_input:
        st.error("분석할 서열을 입력해 주세요.")
    elif len(user_input) < 9:
        st.warning("⚠️ 서열이 너무 짧습니다. 최소 9글자 이상 입력해주세요.")
    else:
        with st.spinner('국립암센터 AI 모델이 서열 패턴을 분석 중입니다...'):
            # 전처리 (9-mer 추출 - 첫 번째 9글자 기준 예시)
            target_seq = user_input[:9]
            
            # 원-핫 인코딩 (ACDEFGHIKLMNPQRSTVWY)
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
            matrix = np.zeros((9, 20))
            for i, aa in enumerate(target_seq):
                if aa in aa_to_int:
                    matrix[i, aa_to_int[aa]] = 1
            
            # AI 예측 (모델이 있을 경우 실행)
            if model:
                prediction = model.predict(np.array([matrix]), verbose=0)
                prob = float(prediction[0][0]) * 100
            else:
                prob = 89.5  # 모델 없을 시 데모용 점수

            # --- 결과 표시 ---
            st.subheader(f"📊 분석 결과: {target_seq}")
            
            res_col1, res_col2 = st.columns([1, 1.2])
            
            with res_col1:
                # 게이지 차트
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "신항원 적합도 (%)", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444" if prob >= 70 else "#3b82f6"},
                        'steps' : [
                            {'range': [0, 40], 'color': "#f3f4f6"},
                            {'range': [40, 70], 'color': "#fef3c7"},
                            {'range': [70, 100], 'color': "#fee2e2"}]
                    }
                ))
                fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.markdown("### 🧬 물리화학적 특성")
                # 소수성 분석
                hydro_score = sum(1 for aa in target_seq if aa in 'AILMFPWV')
                m1, m2 = st.columns(2)
                m1.metric("소수성 지수", f"{hydro_score}/9")
                m2.metric("결합 안정성", "높음" if 'L' in target_seq or 'V' in target_seq else "보통")
                
                # 위치별 기여도 시각화
                aa_props = pd.DataFrame({
                    'Position': [f"P{i+1}" for i in range(9)],
                    'Is_Hydro': [1 if aa in 'AILMFPWV' else 0 for aa in target_seq]
                })
                st.write("**📍 위치별 결합 기여도 (소수성)**")
                st.bar_chart(aa_props.set_index('Position'), height=200)

            # --- 쉬운 해설 섹션 ---
            st.write("---")
            exp_col1, exp_col2 = st.columns(2)
            
            with exp_col1:
                st.subheader("📖 AI 분석 해설")
                if prob >= 70:
                    st.success(f"**[최종 판정: 적합]**\n\n이 서열은 암세포의 '지문'일 가능성이 매우 높습니다. 면역 세포가 이 서열을 인지하여 암세포를 공격하도록 유도하는 **맞춤형 암 백신** 설계의 유력한 후보입니다.")
                elif prob >= 40:
                    st.warning(f"**[최종 판정: 보류]**\n\n면역 반응 가능성이 존재하나, MHC 결합 안정성이 다소 낮을 수 있습니다. 추가적인 실험적 검증이 권장됩니다.")
                else:
                    st.error(f"**[최종 판정: 부적합]**\n\n정상 세포의 서열과 유사하여 면역 시스템이 적으로 간주하지 않을 확률이 높습니다.")

            with exp_col2:
                st.subheader("👨‍👩‍👧‍👦 환자를 위한 안내")
                st.info(f"**\"오직 환자분만을 위한 정밀 타격\"**\n\n입력하신 서열은 암세포가 입은 '특이한 무늬 옷'과 같습니다. AI는 면역 세포가 이 무늬를 얼마나 잘 찾아낼 수 있는지 점수화(현재 {prob:.1f}%)하여 최적의 치료법을 설계하는 데 도움을 줍니다.")

# --- [5] 하단 참조 데이터 ---
if os.path.exists("df_final.csv"):
    with st.expander("📋 전체 분석 후보군 리스트 확인"):
        df_all = pd.read_csv("df_final.csv")
        st.dataframe(df_all, use_container_width=True)
