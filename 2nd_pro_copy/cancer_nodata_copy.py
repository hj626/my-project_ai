"""
============================================================
🔬 AI 기반 환자 맞춤형 신항원 발굴 시스템 (실시간 학습 버전)
============================================================

[시스템 목적]
데모 및 교육용으로, 데이터 로드부터 모델 학습까지 전 과정을
실시간으로 보여주는 버전입니다.

[특징]
- 학습 과정 가시화
- 빠른 데모 (Epoch 수 최소화)
- 전체 파이프라인 이해에 최적

[주의사항]
실제 배포용은 cancer_cells_app_FINAL.py를 사용하세요.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import time

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="AI 암 백신 시스템 (실시간 학습)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff9e6;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .process-step {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 사이드바: 시스템 정보
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## 🤖 시스템 정보")
    
    st.markdown("""
    <div class="info-box">
        <h4>⚡ 실시간 학습 버전</h4>
        <p>데이터 로드 → 전처리 → 학습까지<br>전체 과정을 실시간으로 보여줍니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 데이터셋 정보")
    st.info("""
    **출처**: IEDB MHC Ligand  
    **전체 데이터**: 약 46만 건  
    **폐암 필터링**: 9-mer 펩타이드  
    **클래스**: Positive/Negative
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 학습 파라미터")
    st.success("""
    **모델**: 1D-CNN  
    **Epochs**: 5회 (빠른 데모용)  
    **Batch Size**: 256  
    **Optimizer**: Adam  
    **Loss**: Binary Crossentropy
    """)
    
    st.markdown("---")
    
    st.markdown("### 📌 이 버전의 장점")
    st.warning("""
    ✅ 전체 파이프라인 이해  
    ✅ 학습 과정 시각화  
    ✅ 교육 및 데모용 최적  
    
    ⚠️ 실제 서비스는  
    케라스 파일 사용 권장
    """)
    
    st.markdown("---")
    
    with st.expander("💡 사용 팁"):
        st.markdown("""
        1. 처음 실행 시 자동으로 모델 학습
        2. 학습 완료 후 서열 입력
        3. 분석 버튼 클릭
        4. 결과 확인 및 다운로드
        
        **새로고침 시**: 재학습 필요
        """)

# ============================================================
# 메인: 타이틀
# ============================================================
st.markdown('<h1 class="main-header">🧬 AI 신항원 발굴 시스템 (실시간 학습)</h1>', 
            unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666;">Real-Time Training & Prediction Pipeline</p>', 
            unsafe_allow_html=True)

st.markdown("---")

# 프로세스 설명
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="process-step">
        <h3 style="text-align: center;">1️⃣</h3>
        <p style="text-align: center;"><b>데이터 로드</b><br>46만건 필터링</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="process-step">
        <h3 style="text-align: center;">2️⃣</h3>
        <p style="text-align: center;"><b>전처리</b><br>원핫 인코딩</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="process-step">
        <h3 style="text-align: center;">3️⃣</h3>
        <p style="text-align: center;"><b>모델 학습</b><br>1D-CNN 훈련</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="process-step">
        <h3 style="text-align: center;">4️⃣</h3>
        <p style="text-align: center;"><b>예측</b><br>신항원 분석</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 모델 학습 엔진 (캐싱)
# ============================================================
@st.cache_resource(show_spinner=False)
def train_model_pipeline():
    """전체 파이프라인: 데이터 로드 → 전처리 → 학습"""
    
    steps_log = []
    
    # STEP 1: 데이터 로드
    steps_log.append("📂 STEP 1: 데이터 파일 로드 중...")
    try:
        df = pd.read_parquet('dataset/mhc_data.parquet')
        steps_log.append(f"✅ 전체 데이터 로드 완료: {len(df):,}건")
    except Exception as e:
        steps_log.append(f"❌ 오류: {str(e)}")
        return None, steps_log, None
    
    # STEP 2: 폐암 데이터 필터링
    steps_log.append("🔍 STEP 2: 폐암 관련 데이터 필터링 중...")
    is_lung = df.iloc[:, 8].str.contains('Lung|Adenocarcinoma|NSCLC|Cancer', case=False, na=False)
    df_lung = df[is_lung].copy()
    steps_log.append(f"✅ 폐암 데이터 추출: {len(df_lung):,}건")
    
    # STEP 3: 9-mer 펩타이드 추출
    steps_log.append("🧬 STEP 3: 9-mer 펩타이드 서열 정제 중...")
    df_final = df_lung.iloc[:, [11, 94]].copy()
    df_final.columns = ['Sequence', 'Label']
    df_final['Sequence'] = df_final['Sequence'].astype(str)
    df_final = df_final[df_final['Sequence'].str.len() == 9].dropna()
    steps_log.append(f"✅ 최종 데이터: {len(df_final):,}건")
    
    # STEP 4: 원핫 인코딩
    steps_log.append("🔧 STEP 4: 데이터 전처리 (원핫 인코딩) 중...")
    
    def neoantigen_onehot(sequences):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
        encoded = []
        for seq in sequences:
            matrix = np.zeros((9, 20))
            for i, aa in enumerate(seq):
                if aa in aa_to_int:
                    matrix[i, aa_to_int[aa]] = 1
            encoded.append(matrix)
        return np.array(encoded)
    
    X = neoantigen_onehot(df_final['Sequence'].values)
    y = (df_final['Label'].str.contains('Positive', case=False)).astype(int).values
    steps_log.append(f"✅ 전처리 완료: {X.shape}")
    
    # STEP 5: 데이터 분할
    steps_log.append("✂️ STEP 5: 학습/테스트 데이터 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    steps_log.append(f"✅ 학습 데이터: {len(X_train):,}개 | 테스트: {len(X_test):,}개")
    
    # STEP 6: 모델 구축
    steps_log.append("🤖 STEP 6: 1D-CNN 모델 구축 중...")
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(9, 20)),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    steps_log.append("✅ 모델 구축 완료")
    
    # STEP 7: 모델 학습
    steps_log.append("🚀 STEP 7: 모델 학습 시작 (5 epochs, 빠른 데모용)...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=256, 
                       validation_data=(X_test, y_test), verbose=0)
    
    final_acc = history.history['val_accuracy'][-1]
    steps_log.append(f"✅ 학습 완료! 최종 검증 정확도: {final_acc*100:.2f}%")
    
    # 통계 정보
    stats = {
        'total_data': len(df),
        'lung_data': len(df_lung),
        'final_data': len(df_final),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'accuracy': final_acc * 100,
        'positive_ratio': (y.sum() / len(y)) * 100
    }
    
    return model, steps_log, stats

# ============================================================
# 모델 초기화 섹션
# ============================================================
st.markdown("## 🔄 AI 모델 초기화")

if 'model_ready' not in st.session_state:
    with st.status("🧬 AI 엔진을 준비하는 중입니다...", expanded=True) as status:
        
        # 프로그레스 바
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 로그 영역
        log_container = st.container()
        
        # 학습 실행
        model, logs, stats = train_model_pipeline()
        
        # 로그 출력
        with log_container:
            for i, log in enumerate(logs):
                st.text(log)
                progress = min(100, int((i + 1) / len(logs) * 100))
                progress_bar.progress(progress)
                time.sleep(0.1)
        
        if model is not None:
            st.session_state['model'] = model
            st.session_state['model_ready'] = True
            st.session_state['stats'] = stats
            status.update(label="✅ AI 엔진 준비 완료!", state="complete", expanded=False)
            st.balloons()
        else:
            st.error("❌ 모델 초기화 실패. 데이터 파일을 확인하세요.")
            st.stop()
else:
    st.success("✅ AI 모델이 이미 준비되어 있습니다!")

# 통계 정보 표시
if 'stats' in st.session_state:
    st.markdown("### 📊 데이터셋 통계")
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    with stat_col1:
        st.metric("전체 데이터", f"{st.session_state['stats']['total_data']:,}건")
    
    with stat_col2:
        st.metric("폐암 데이터", f"{st.session_state['stats']['lung_data']:,}건")
    
    with stat_col3:
        st.metric("학습 데이터", f"{st.session_state['stats']['train_size']:,}건")
    
    with stat_col4:
        st.metric("검증 정확도", f"{st.session_state['stats']['accuracy']:.2f}%")
    
    with stat_col5:
        st.metric("Positive 비율", f"{st.session_state['stats']['positive_ratio']:.1f}%")

st.markdown("---")

# ============================================================
# 입력 섹션
# ============================================================
st.markdown("## 🧬 신항원 서열 분석")

input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    st.markdown("""
    <div class="info-box">
        <b>💡 입력 가이드</b><br>
        • 9자리 아미노산 서열 입력<br>
        • 표준 20종 아미노산 사용 (A~Y)<br>
        • 환자 유전자 검사에서 발견된 돌연변이 서열 사용
    </div>
    """, unsafe_allow_html=True)
    
    sequence = st.text_input(
        "아미노산 서열 (9자리)",
        value="LLDFVRFMG",
        max_chars=9
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔬 정밀 분석 시작", type="primary", use_container_width=True)
    
    st.markdown("**예시 서열:**")
    example_col1, example_col2 = st.columns(2)
    with example_col1:
        if st.button("예시 1", use_container_width=True):
            sequence = "KLLMVLMLA"
            st.rerun()
    with example_col2:
        if st.button("예시 2", use_container_width=True):
            sequence = "FLNQTDETL"
            st.rerun()

# ============================================================
# 분석 실행
# ============================================================
if analyze_btn:
    if len(sequence) != 9:
        st.error("⚠️ 정확히 9글자를 입력해주세요.")
    else:
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequence_upper = sequence.upper()
        invalid = [c for c in sequence_upper if c not in amino_acids]
        
        if invalid:
            st.error(f"⚠️ 잘못된 아미노산: {', '.join(invalid)}")
        else:
            # 예측 수행
            with st.spinner('🔬 분석 중...'):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)
                
                # 원핫 인코딩
                aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
                matrix = np.zeros((9, 20))
                for i, aa in enumerate(sequence_upper):
                    if aa in aa_to_int:
                        matrix[i, aa_to_int[aa]] = 1
                
                # 예측
                pred = st.session_state['model'].predict(np.array([matrix]), verbose=0)
                prob = float(pred[0][0])
            
            st.markdown("---")
            
            # ============================================================
            # 결과 리포트
            # ============================================================
            st.markdown("## 📊 분석 결과 리포트")
            
            # 메트릭
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            with res_col1:
                st.metric("입력 서열", sequence_upper)
            
            with res_col2:
                st.metric("면역원성 점수", f"{prob*100:.2f}%",
                         delta=f"{(prob-0.5)*100:+.1f}%p" if prob > 0.5 else None)
            
            with res_col3:
                if prob > 0.8:
                    status = "🔴 최우선"
                elif prob > 0.5:
                    status = "🟠 추천"
                else:
                    status = "⚪ 부적합"
                st.metric("판정", status)
            
            with res_col4:
                vaccine = "설계 포함" if prob > 0.5 else "재검토"
                st.metric("백신 설계", vaccine)
            
            st.markdown("---")
            
            # 게이지 차트
            st.markdown("### 📈 면역원성 시각화")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "면역원성 점수 (%)", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "lightyellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # 결과 해석
            st.markdown("### 🔬 결과 해석")
            
            if prob > 0.8:
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ 강력 추천 신항원</h4>
                    <p><b>{sequence_upper}</b>는 면역원성이 매우 높습니다 ({prob*100:.2f}%).</p>
                    <ul>
                        <li>백신 설계 최우선 후보</li>
                        <li>강력한 T세포 반응 예상</li>
                        <li>펩타이드 또는 mRNA 백신 제조 권장</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                
            elif prob > 0.5:
                st.markdown(f"""
                <div class="info-box">
                    <h4>✅ 추천 신항원</h4>
                    <p><b>{sequence_upper}</b>는 면역원성이 있습니다 ({prob*100:.2f}%).</p>
                    <ul>
                        <li>백신 후보 목록에 포함</li>
                        <li>다른 고순위 신항원과 병합 사용 권장</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ 백신 부적합</h4>
                    <p><b>{sequence_upper}</b>는 면역원성이 낮습니다 ({prob*100:.2f}%).</p>
                    <ul>
                        <li>다른 돌연변이 서열 탐색 권장</li>
                        <li>HLA 타입 재확인 필요</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # 다운로드
            report_df = pd.DataFrame({
                '서열': [sequence_upper],
                '면역원성': [f"{prob*100:.2f}%"],
                '판정': [status],
                '권고': ['백신 포함' if prob > 0.5 else '재검토']
            })
            
            csv = report_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 결과 다운로드 (CSV)",
                data=csv,
                file_name=f"analysis_{sequence_upper}.csv",
                mime="text/csv"
            )

# ============================================================
# 푸터
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p><b>🧬 AI 기반 신항원 발굴 시스템 (실시간 학습 버전)</b></p>
    <p>Real-Time Training Pipeline for Educational & Demo Purpose</p>
    <p style="font-size: 0.85rem;">⚠️ 빠른 데모를 위해 Epoch 수를 줄였습니다. 실제 배포는 케라스 파일 버전을 사용하세요.</p>
</div>
""", unsafe_allow_html=True)