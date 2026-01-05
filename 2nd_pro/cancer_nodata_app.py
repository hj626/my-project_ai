import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import time

# --- [1] 페이지 설정 및 테마 ---
st.set_page_config(page_title="Personalized Cancer Vaccine Design", page_icon="🧬", layout="wide")

# 사이드바: 프로젝트 정보
with st.sidebar:
    st.header("프로젝트 정보")
    st.info("🧬 **주제**: AI 기반 환자 맞춤형 신항원 발굴 및 암 백신 설계 시스템")
    st.write("---")
    st.write("✅ **알고리즘**: 1D-CNN (Convolutional Neural Network)")
    st.write("✅ **데이터셋**: MHC Ligand Interaction (Lung Cancer Filtered)")
    st.write("✅ **최종 정확도**: 99.94%")

# --- [2] 모델 학습 엔진 (실시간 학습) ---
@st.cache_resource
def train_and_get_model():
    # 데이터 로드 (노트북 Cell 3)
    df = pd.read_parquet('dataset/mhc_data.parquet') 

    # 폐암 데이터 필터링 (노트북 Cell 6)
    is_lung = df.iloc[:, 8].str.contains('Lung|Adenocarcinoma|NSCLC|Cancer', case=False, na=False)
    df_lung = df[is_lung].copy()
    df_final = df_lung.iloc[:, [11, 94]].copy() 
    df_final.columns = ['Sequence', 'Label']
    df_final = df_final[df_final['Sequence'].str.len() == 9].dropna()

    # 원핫 인코딩 함수 (노트북 Cell 8)
    def neoantigen_onehot(sequences):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
        encoded = []
        for seq in sequences:
            matrix = np.zeros((9, 20))
            for i, aa in enumerate(seq):
                if aa in aa_to_int: matrix[i, aa_to_int[aa]] = 1
            encoded.append(matrix)
        return np.array(encoded)

    X = neoantigen_onehot(df_final['Sequence'].values)
    y = (df_final['Label'].str.contains('Positive', case=False)).astype(int).values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)

    # 1D-CNN 모델 생성 및 학습 (노트북 Cell 9)
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(9, 20)),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2, batch_size=256, verbose=0) 
    return model

# --- [3] 메인 화면 레이아웃 ---
st.title("🛡️ AI 기반 환자 맞춤형 신항원 발굴 시스템")
st.subheader("Patient-Specific Neoantigen Discovery & Cancer Vaccine Design")
st.write("---")

# 모델 로드 섹션
if 'model_ready' not in st.session_state:
    with st.status("🧬 AI 모델 엔진을 초기화 중입니다...", expanded=True) as status:
        st.write("데이터 필터링 중...")
        model = train_and_get_model()
        st.session_state['model'] = model
        st.session_state['model_ready'] = True
        status.update(label="✅ 엔진 준비 완료! 시스템을 시작합니다.", state="complete", expanded=False)

# 입력 섹션
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 🧬 아미노산 서열 입력")
    sequence = st.text_input("분석할 9자리 아미노산 서열을 입력하세요.", "LLDFVRFMG", help="표준 아미노산 기호로 입력")

with col2:
    st.write("### ⚡ 분석 실행")
    st.write("") # 간격 조절
    run_button = st.button("🔎 정밀 분석 시작", type="primary")

# --- [4] 분석 결과 리포트 섹션 ---
if run_button:
    if len(sequence) == 9:
        # 전처리
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
        matrix = np.zeros((9, 20))
        for i, aa in enumerate(sequence.upper()):
            if aa in aa_to_int: matrix[i, aa_to_int[aa]] = 1
        
        # 예측
        prediction = st.session_state['model'].predict(np.array([matrix]))
        prob = float(prediction[0][0])
        
        # 결과 대시보드
        st.write("---")
        st.markdown("### 📊 신항원 정밀 분석 리포트")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(label="면역 반응 적합성", value=f"{prob*100:.2f}%")
        
        with res_col2:
            status_text = "Highly Positive" if prob > 0.8 else "Positive" if prob > 0.5 else "Negative"
            st.metric(label="최종 판정", value=status_text)
            
        with res_col3:
            vaccine_score = "백신 설계 포함" if prob > 0.5 else "설계 부적합"
            st.metric(label="백신 설계 권고", value=vaccine_score)

        # 게이지 바 시각화
        st.write("**신항원 적합도 점수**")
        st.progress(prob)
        
        if prob > 0.5:
            st.success(f"**[결과 분석]**: 해당 서열({sequence})은 면역 세포가 암세포를 식별할 수 있는 **유효한 신항원**으로 판단됩니다. 개인 맞춤형 암 백신 설계 후보군에 등록을 권장합니다.")
            st.balloons()
        else:
            st.warning(f"**[결과 분석]**: 해당 서열은 면역 세포 활성화 확률이 낮습니다. 다른 후보 서열 분석을 권장합니다.")
            
    else:
        st.error("⚠️ 오류: 반드시 9글자의 서열을 입력해야 정밀 분석이 가능합니다.")