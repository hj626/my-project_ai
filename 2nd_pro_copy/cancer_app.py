"""
============================================================
🔬 AI 기반 환자 맞춤형 신항원 발굴 및 암 백신 설계 시스템
============================================================

[시스템 목적]
폐암 환자의 유전자 변이로부터 생성된 펩타이드 서열을 분석하여,
면역세포가 암세포를 효과적으로 인식할 수 있는 "신항원" 후보를
AI 모델로 예측하고, 맞춤형 암 백신 설계를 지원합니다.

[사용 방법]
1. 환자의 9자리 아미노산 서열 입력
2. AI 분석 시작 버튼 클릭
3. 면역원성 점수 및 백신 설계 권고 확인

[개발 정보]
- 모델: 1D-CNN (Convolutional Neural Network)
- 학습 데이터: 46만 건의 MHC-펩타이드 상호작용 데이터
- 정확도: 99.94%
- AUC: 0.9998
============================================================
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import time

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="AI 암 백신 설계 시스템",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .info-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 사이드바: 프로젝트 정보
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dna-helix.png", width=80)
    st.markdown("## 📊 프로젝트 정보")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 프로젝트 목표</h4>
        <p>폐암 환자 맞춤형 암 백신 설계를 위한 신항원 발굴 시스템 구축</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔬 기술 스펙")
    st.info("""
    **알고리즘**: 1D-CNN  
    **학습 데이터**: 462,017건  
    **정확도**: 99.94%  
    **AUC Score**: 0.9998
    """)
    
    st.markdown("---")
    
    st.markdown("### 📌 왜 폐암인가?")
    st.warning("""
    🇰🇷 **국내 현황**
    - 암 발생률 2위
    - 5년 생존율 35.4%
    - 유전자 변이가 다양함
    
    → **맞춤 치료 필수!**
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 신항원이란?")
    st.success("""
    암세포의 돌연변이로 생긴  
    **새로운 단백질 조각**
    
    면역세포가 이를 인식하면  
    암세포만 골라서 공격 가능!
    """)
    
    st.markdown("---")
    
    st.markdown("### 👨‍⚕️ 활용 시나리오")
    with st.expander("자세히 보기"):
        st.markdown("""
        1️⃣ **유전자 검사**  
        환자의 종양 조직에서 DNA 분석
        
        2️⃣ **돌연변이 발견**  
        정상 세포와 다른 부분 찾기
        
        3️⃣ **AI 분석** ← 여기!  
        어떤 조각이 백신 후보인지 예측
        
        4️⃣ **백신 제조**  
        선정된 신항원으로 맞춤 백신
        
        5️⃣ **환자 투여**  
        면역세포가 암세포 공격!
        """)

# ============================================================
# 메인: 타이틀 및 소개
# ============================================================
st.markdown('<h1 class="main-header">🛡️ AI 기반 환자 맞춤형 신항원 발굴 시스템</h1>', 
            unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Patient-Specific Neoantigen Discovery & Cancer Vaccine Design</p>', 
            unsafe_allow_html=True)

st.markdown("---")

# 프로젝트 설명 섹션
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #1f77b4;">🎯</h2>
        <h4>문제점</h4>
        <p>암세포는 계속 변이해서<br>면역세포가 못 찾음</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #2ca02c;">🤖</h2>
        <h4>AI 역할</h4>
        <p>환자의 유전자 분석해서<br>최적 신항원 발굴</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #ff7f0e;">💉</h2>
        <h4>최종 결과</h4>
        <p>맞춤형 암 백신<br>설계 가이드 제공</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 모델 로드
# ============================================================
@st.cache_resource
def load_trained_model():
    """학습된 AI 모델 로드 (캐싱으로 1회만 실행)"""
    try:
        model = tf.keras.models.load_model("lung_cancer_model.keras")
        return model, True
    except Exception as e:
        return None, False

with st.spinner('🧬 AI 모델 엔진을 불러오는 중...'):
    model, model_loaded = load_trained_model()

if not model_loaded:
    st.error("⚠️ 모델 파일을 찾을 수 없습니다. 'lung_cancer_model.keras' 파일이 같은 폴더에 있는지 확인하세요.")
    st.stop()

st.success("✅ AI 모델 준비 완료! 분석을 시작할 수 있습니다.")

# ============================================================
# 입력 섹션
# ============================================================
st.markdown("## 🧬 신항원 후보 서열 입력")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <div class="info-box">
        <b>💡 입력 가이드</b><br>
        • 9자리 아미노산 서열을 입력하세요 (예: LLDFVRFMG)<br>
        • 표준 아미노산 기호 사용: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y<br>
        • 실제 환자의 유전자 검사에서 발견된 돌연변이 서열을 입력합니다
    </div>
    """, unsafe_allow_html=True)
    
    sequence = st.text_input(
        "아미노산 서열 (9자리)",
        value="LLDFVRFMG",
        max_chars=9,
        help="펩타이드 서열을 대문자로 입력하세요"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🔬 정밀 분석 시작", type="primary", use_container_width=True)
    
    # 예시 버튼들
    st.markdown("**빠른 예시:**")
    if st.button("예시 1", use_container_width=True):
        sequence = "KLLMVLMLA"
        st.rerun()
    if st.button("예시 2", use_container_width=True):
        sequence = "FLNQTDETL"
        st.rerun()

# ============================================================
# 분석 실행
# ============================================================
if analyze_button:
    if len(sequence) != 9:
        st.error("⚠️ 오류: 정확히 9글자의 아미노산 서열을 입력해주세요.")
    else:
        # 입력 검증
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequence_upper = sequence.upper()
        invalid_chars = [c for c in sequence_upper if c not in amino_acids]
        
        if invalid_chars:
            st.error(f"⚠️ 오류: 잘못된 아미노산 기호가 포함되어 있습니다: {', '.join(invalid_chars)}")
        else:
            # 프로그레스 바
            with st.spinner('🔬 AI가 서열을 분석하는 중...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # 원핫 인코딩
                aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
                matrix = np.zeros((9, 20))
                for i, aa in enumerate(sequence_upper):
                    if aa in aa_to_int:
                        matrix[i, aa_to_int[aa]] = 1
                
                # 예측
                prediction = model.predict(np.array([matrix]), verbose=0)
                prob = float(prediction[0][0])
            
            st.markdown("---")
            
            # ============================================================
            # 결과 대시보드
            # ============================================================
            st.markdown("## 📊 신항원 정밀 분석 리포트")
            
            # 메트릭 카드
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    label="입력 서열",
                    value=sequence_upper,
                    help="분석 대상 펩타이드"
                )
            
            with metric_col2:
                st.metric(
                    label="면역원성 점수",
                    value=f"{prob*100:.2f}%",
                    delta=f"{(prob-0.5)*100:+.2f}%p" if prob > 0.5 else None,
                    help="면역세포 반응 확률"
                )
            
            with metric_col3:
                if prob > 0.8:
                    status = "🔴 최우선"
                    status_color = "red"
                elif prob > 0.5:
                    status = "🟠 추천"
                    status_color = "orange"
                else:
                    status = "⚪ 부적합"
                    status_color = "gray"
                
                st.metric(
                    label="최종 판정",
                    value=status,
                    help="백신 후보 적합성"
                )
            
            with metric_col4:
                rank_text = "Top 10%" if prob > 0.9 else "Top 30%" if prob > 0.7 else "하위"
                st.metric(
                    label="예상 순위",
                    value=rank_text,
                    help="전체 후보 중 예상 순위"
                )
            
            st.markdown("---")
            
            # 게이지 차트
            st.markdown("### 📈 면역원성 시각화")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "면역원성 점수", 'font': {'size': 24}},
                delta={'reference': 50, 'suffix': "%p"},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#ffcccc'},
                        {'range': [50, 80], 'color': '#ffffcc'},
                        {'range': [80, 100], 'color': '#ccffcc'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # 결과 해석
            st.markdown("### 🔬 결과 해석 및 권고사항")
            
            if prob > 0.8:
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ 강력 추천 신항원 후보</h4>
                    <p><b>서열 {sequence_upper}</b>는 면역원성이 매우 높은 것으로 분석되었습니다 
                    (확률: <b>{prob*100:.2f}%</b>).</p>
                    
                    <h5>📋 백신 설계 권고사항:</h5>
                    <ul>
                        <li><b>우선순위</b>: 최우선 백신 후보로 등록</li>
                        <li><b>합성 방법</b>: 펩타이드 합성 또는 mRNA 백신</li>
                        <li><b>면역 보조제</b>: Adjuvant와 함께 사용 권장</li>
                        <li><b>예상 효과</b>: 강력한 T세포 반응 유도 가능</li>
                    </ul>
                    
                    <h5>💉 투여 계획 제안:</h5>
                    <ul>
                        <li>1회차: 진단 직후 (0주)</li>
                        <li>2회차: 3주 후</li>
                        <li>3회차: 6주 후</li>
                        <li>부스터: 3개월 후</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                
            elif prob > 0.5:
                st.markdown(f"""
                <div class="info-box">
                    <h4>✅ 추천 신항원 후보</h4>
                    <p><b>서열 {sequence_upper}</b>는 면역원성이 있는 것으로 분석되었습니다 
                    (확률: <b>{prob*100:.2f}%</b>).</p>
                    
                    <h5>📋 백신 설계 권고사항:</h5>
                    <ul>
                        <li><b>우선순위</b>: 백신 후보 목록에 포함</li>
                        <li><b>추가 검증</b>: in vitro 면역 반응 테스트 권장</li>
                        <li><b>병합 전략</b>: 다른 고순위 신항원과 함께 사용</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ 백신 후보 부적합</h4>
                    <p><b>서열 {sequence_upper}</b>는 면역원성이 낮은 것으로 분석되었습니다 
                    (확률: <b>{prob*100:.2f}%</b>).</p>
                    
                    <h5>📋 권고사항:</h5>
                    <ul>
                        <li>다른 돌연변이 부위의 신항원 후보를 탐색하세요</li>
                        <li>환자의 전체 변이 프로파일을 재검토하세요</li>
                        <li>HLA 타입과의 결합 친화도를 추가 확인하세요</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 추가 정보
            with st.expander("📚 상세 분석 정보"):
                st.markdown(f"""
                **입력 서열**: `{sequence_upper}`
                
                **서열 특성 분석**:
                - 길이: 9-mer (MHC Class I 최적 길이)
                - 첫 아미노산: {sequence_upper[0]}
                - 마지막 아미노산: {sequence_upper[-1]}
                - 소수성 아미노산: {sum(1 for aa in sequence_upper if aa in 'AILMFPWV')}개
                
                **AI 모델 정보**:
                - 알고리즘: 1D Convolutional Neural Network
                - 학습 데이터: 462,017건의 폐암 관련 MHC-펩타이드
                - 모델 정확도: 99.94%
                - AUC Score: 0.9998
                
                **예측 신뢰도**: {prob*100:.2f}%
                """)
            
            # 다운로드 버튼
            report_data = {
                '서열': [sequence_upper],
                '면역원성 점수': [f"{prob*100:.2f}%"],
                '판정': [status],
                '권고사항': ['백신 설계 포함' if prob > 0.5 else '재분석 필요']
            }
            df_report = pd.DataFrame(report_data)
            
            csv = df_report.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 분석 결과 다운로드 (CSV)",
                data=csv,
                file_name=f"neoantigen_analysis_{sequence_upper}.csv",
                mime="text/csv"
            )

# ============================================================
# 푸터
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><b>🔬 AI 기반 환자 맞춤형 신항원 발굴 시스템</b></p>
    <p>Powered by 1D-CNN Deep Learning | Trained on 462K+ Data Points</p>
    <p style="font-size: 0.9rem;">⚠️ 본 시스템은 연구 목적으로 제작되었으며, 실제 임상 사용 시 의료진의 검토가 필요합니다.</p>
</div>
""", unsafe_allow_html=True)