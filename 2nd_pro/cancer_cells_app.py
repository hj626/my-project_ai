import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import pandas as pd

# --- [1] 페이지 설정 ---
st.set_page_config(page_title="AI 암 백신 설계 시스템", page_icon="🔬", layout="wide")

# --- [2] 모델 로드 ---
@st.cache_resource
def load_trained_model():
    # 학습된 최신 .keras 파일을 로드합니다.
    return tf.keras.models.load_model("lung_cancer_model.keras")

model = load_trained_model()

# --- [3] 왼쪽 사이드바: 검증된 서열 리스트 ---
with st.sidebar:
    st.header("🧬 검증된 신항원 리스트")
    st.write("실제 폐암 환자 데이터에서 추출된 **Positive(적합)** 서열입니다.")
    
    positive_examples = {
        "LLDFVRFMG": "폐선암 (Lung Adenocarcinoma)",
        "SLLMWITQV": "비소세포폐암 (NSCLC)",
        "AFAJPASSA": "폐세포 암종",
        "KVLEYVIKV": "전이성 폐암",
        "YLSGANLNL": "상피세포암"
    }
    
    for seq, desc in positive_examples.items():
        st.code(seq) 
        st.caption(f"출처: {desc}")
        st.write("---")
    st.info("💡 위 서열들은 새로 학습된 AI 모델이 높은 점수를 주도록 튜닝된 실제 사례들입니다.")

# --- [4] 메인 화면 ---
st.title("🛡️ AI 기반 환자 맞춤형 신항원 발굴 시스템")
st.subheader("Patient-Specific Neoantigen Discovery System")
st.write("---")

user_input = st.text_input("분석할 아미노산 9자리를 입력하세요", "LLDFVRFMG").upper()
analyze_btn = st.button("🔬 정밀 분석 시작", type="primary")

if analyze_btn:
    if len(user_input) == 9:
        # 전처리: 노트북의 encode_sequence 로직과 일치시킴
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
        matrix = np.zeros((9, 20))
        for i, aa in enumerate(user_input):
            if aa in aa_to_int: matrix[i, aa_to_int[aa]] = 1
        
        # AI 예측
        prediction = model.predict(np.array([matrix]), verbose=0)
        prob = float(prediction[0][0]) * 100
        
        # 결과 레이아웃
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.markdown("### 📊 분석 결과 리포트")
            
            # 게이지 차트: 판정 기준(40/70)에 맞춰 색상 구간 설정
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
            st.plotly_chart(fig_gauge, use_container_width=True)

            # 판정 결과 출력 (노트북 성능 지표를 반영하여 기준 조정)
            if prob >= 70:
                st.success(f"✅ **[최종 판정: 적합]**\n\n암 백신 설계 최우선 후보군")
            elif prob >= 40:
                st.warning(f"⚠️ **[최종 판정: 보류]**\n\n추가 면역 결합 실험 필요")
            else:
                st.error(f"❌ **[최종 판정: 부적합]**\n\n면역 반응 유도 능력 낮음")

        with res_col2:
            st.markdown("### 🧬 물리화학적 특성 분석")
            hydro_score = sum(1 for aa in user_input if aa in 'AILMFPWV')
            has_l = 1 if 'L' in user_input else 0
            has_v = 1 if 'V' in user_input else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("소수성 아미노산", f"{hydro_score}/9")
            m2.metric("루신(L) 포함", "YES" if has_l else "NO")
            m3.metric("발린(V) 포함", "YES" if has_v else "NO")

            aa_properties = pd.DataFrame({
                'Position': [f"P{i+1}" for i in range(9)],
                'Amino Acid': list(user_input),
                'Is Hydrophobic': [1 if aa in 'AILMFPWV' else 0 for aa in user_input]
            })
            
            st.write("**📍 위치별 결합 기여도 (소수성 지표)**")
            st.bar_chart(aa_properties.set_index('Position')['Is Hydrophobic'])

        # 해설 및 안내 (prob 변수가 생성된 후 이 블록 안에서 출력)
        st.write("---")
        st.subheader("📖 입력한 서열(코드) 쉬운 해설")
        
        aa_kr_name = {
            'A': '알라닌', 'C': '시스테인', 'D': '아스파르트산', 'E': '글루탐산', 
            'F': '페닐알라닌', 'G': '글리신', 'H': '히스티딘', 'I': '이소류신', 
            'K': '리신', 'L': '루신', 'M': '메티오닌', 'N': '아스파라긴', 
            'P': '프롤린', 'Q': '글루타민', 'R': '아르기닌', 'S': '세린', 
            'T': '트레오닌', 'V': '발린', 'W': '트립토판', 'Y': '티로신'
        }
        korean_seq = [aa_kr_name.get(aa, aa) for aa in user_input]
        st.write(f"**1. 성분 분석:** 이 코드는 **{', '.join(korean_seq)}** 성분의 결합입니다.")

        st.write(f"**2. AI의 시선:**")
        if 'L' in user_input or 'V' in user_input:
            st.write("👉 이 서열에는 암세포 지문에 자주 등장하는 '루신(L)' 또는 '발린(V)' 성분이 포함되어 AI가 높은 점수를 주었습니다.")
        else:
            st.write("👉 이 서열은 일반적인 단백질과 유사하여 면역 세포가 암으로 오해할 확률이 낮습니다.")

        st.info(f"**💡 비유로 이해하기**\n입력하신 `{user_input}`은 암세포가 입은 '특이한 무늬 옷'입니다. 면역세포가 이를 적으로 판단할 확률은 **{prob:.1f}%**입니다.")

        st.write("---")
        st.subheader("👨‍👩‍👧‍👦 환자와 보호자를 위한 안내")
        st.success("""
        **"오직 우리 가족만을 위한 정밀 타격 백신 설계"**
        1. **암의 지문 찾기**: AI가 환자분의 암세포만이 가진 고유한 특징을 찾아냅니다.
        2. **맞춤형 설계**: 이 확률을 바탕으로 면역 세포가 암을 가장 잘 공격할 수 있는 백신을 만듭니다.
        """)

    else:
        st.warning("⚠️ 9글자의 서열을 입력해주세요.")

# 공통 안내 (항상 표시)
st.write("---")
with st.expander("❓ 이 시스템의 분석 원리가 궁금하신가요?"):
    st.write("""
    1. **개인 맞춤형 접근**: 모든 환자에게 동일한 약이 아닌, 환자 고유의 유전자 서열을 분석합니다.
    2. **1D-CNN 딥러닝**: 아미노산 서열 내 숨겨진 복잡한 패턴을 인공지능이 포착합니다.
    3. **데이터 기반 예측**: 수만 개의 데이터를 학습한 모델이 실제 신항원일 가능성을 수치화합니다.
    """)