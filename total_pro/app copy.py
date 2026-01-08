import streamlit as st
import pandas as pd
import gzip
from Bio import SeqIO
import os
import plotly.graph_objects as go
import time

# 1. 화면 설정
st.set_page_config(
    page_title="폐암 신항원 정밀 예측기", 
    page_icon="🧬",
    layout="wide"
)

# 2. 로직: 정상 서열 데이터베이스 구축 (캐싱 처리)
@st.cache_resource
def build_normal_db(fasta_path):
    if not os.path.exists(fasta_path):
        return None
    
    normal_set = set()
    try:
        with gzip.open(fasta_path, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                if len(seq) >= 9:
                    for i in range(len(seq) - 8):
                        normal_set.add(seq[i:i+9])
        return normal_set
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None

# 3. 사이드바 설정 (cancer_app 스타일 적용)
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    file_name = st.text_input("정상 서열 DB 파일명", "uniprotkb_Proteome_UP000005640_2026_01_06.fasta.gz")
    
    st.markdown("---")
    st.info("""
    **🎯 분석 가이드**
    1. 정상 단백질 DB를 로드합니다.
    2. 입력된 서열을 9-mer로 쪼갭니다.
    3. DB에 없는 서열을 '신항원'으로 분류합니다.
    """)
    
    st.markdown("### 📌 폐암 신항원 연구")
    st.warning("국립암센터 AI 대학원 관련 주제: 폐암의 유전자 변이 다양성을 고려한 맞춤형 백신 설계 연구")

# 4. 메인 화면 타이틀
st.title("🛡️ AI 기반 폐암 신항원 발굴 시스템")
st.caption("Lung Cancer Neoantigen Discovery System")

# 데이터 로드 체크
if os.path.exists(file_name):
    with st.spinner("🧬 정상 서열 사전을 구축 중입니다..."):
        normal_db = build_normal_db(file_name)
    if normal_db:
        st.success(f"✅ {len(normal_db):,}개의 정상 서열 로드 완료")
else:
    st.error(f"❌ '{file_name}' 파일을 찾을 수 없습니다.")
    st.stop()

# 5. 입력 섹션
st.markdown("---")
st.markdown("## 🧬 분석 대상 서열 입력")
user_input = st.text_area("아미노산 서열 입력 (예: MTEYKLVVVG...)", height=150)

if st.button("🔬 정밀 분석 시작", type="primary"):
    if not user_input or len(user_input) < 9:
        st.error("⚠️ 9글자 이상의 서열을 입력해주세요.")
    else:
        with st.spinner('🔬 DB 대조 분석 중...'):
            # 분석 로직
            res = []
            user_input = user_input.upper().strip()
            for i in range(len(user_input) - 8):
                sub = user_input[i:i+9]
                # DB에 없으면 신항원(Label 1), 있으면 정상(Label 0)
                is_neo = 1 if sub not in normal_db else 0
                res.append({
                    "9-mer Sequence": sub, 
                    "Label": is_neo, 
                    "Status": "Neoantigen" if is_neo == 1 else "Normal",
                    "Hydrophobic": sum(1 for aa in sub if aa in 'AILMFPWV')
                })
            
            df = pd.DataFrame(res)
            neo_count = len(df[df['Label'] == 1])
            
            # 6. 결과 리포트 (cancer_app 스타일)
            st.markdown("---")
            st.markdown("## 📊 분석 결과 리포트")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("총 분석 서열", f"{len(df)} 개")
            c2.metric("신항원 후보", f"{neo_count} 개")
            c3.metric("분석 상태", "완료")

            # 게이지 차트 (첫 번째 발견된 신항원 기준 또는 전체 비율)
            if neo_count > 0:
                neo_ratio = (neo_count / len(df)) * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=neo_ratio,
                    title={'text': "전체 서열 내 신항원 밀도 (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "red"},
                           'steps': [
                               {'range': [0, 30], 'color': "lightgray"},
                               {'range': [30, 70], 'color': "orange"},
                               {'range': [70, 100], 'color': "red"}]}
                ))
                st.plotly_chart(fig, use_container_width=True)

            # 결과 테이블
            st.subheader("📋 상세 분석 목록")
            
            def highlight_neo(val):
                if val == "Neoantigen":
                    return 'background-color: #ff4b4b; color: white; font-weight: bold'
                return ''

            # 최신 Pandas 대응을 위해 map 사용 (또는 applymap)
            st.dataframe(df.style.map(highlight_neo, subset=['Status']), use_container_width=True)

            # 7. 환자용 설명 섹션
            st.markdown("---")
            with st.expander("📖 환자·보호자를 위한 쉬운 설명"):
                st.info(f"""import streamlit as st
import pandas as pd
import tensorflow as tf  # 모델 로드용
import numpy as np
import os
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="폐암 신항원 AI 분석 시스템", layout="wide")

# 2. 파일 로드 함수 (노트북 결과물 활용)
@st.cache_resource
def load_assets():
    model_path = "lung_cancer_model.keras"
    # 모델이 있으면 로드, 없으면 None 반환
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_assets()

# 3. 사이드바: 분석 리포트 연결
with st.sidebar:
    st.title("📂 분석 자산")
    if os.path.exists("lung_cancer_rich_report.html"):
        st.success("✅ 데이터 리포트 준비 완료")
        # 리포트 다운로드 버튼
        with open("lung_cancer_rich_report.html", "rb") as f:
            st.download_button("📊 상세 분석 리포트 다운로드", f, file_name="Report.html")
    
    st.markdown("---")
    st.info("국립암센터 AI 대학원 관련 폐암 연구 프로젝트")

# 4. 메인 UI
st.title("🧬 폐암 신항원 정밀 예측 시스템")

if model is None:
    st.warning("⚠️ 'lung_cancer_model.keras' 파일이 없습니다. 현재는 DB 대조 모드로만 작동합니다.")

user_input = st.text_area("분석할 서열을 입력하세요", height=150)

if st.button("🔬 AI 정밀 분석 시작"):
    # [노트북 로직 반영] 
    # 1. 입력 서열 9-mer 분할
    # 2. AI 모델 예측 (예시 코드)
    # 3. 결과 시각화
    
    st.balloons()
    st.subheader("📊 예측 분석 결과")
    
    # 임시 결과 데모 (실제 구현 시 모델 인퍼런스 코드 삽입)
    col1, col2 = st.columns(2)
    col1.metric("신항원 가능성", "89.5%", "+2.3%")
    col2.metric("MHC 결합력 점수", "0.92/1.0")

    # 데이터프레임 표시 (노트북에서 만든 df_final의 구조 활용)
    if os.path.exists("df_final.csv"):
        df_sample = pd.read_csv("df_final.csv").head(5)
        st.write("📋 참조 데이터 (df_final.csv):")
        st.dataframe(df_sample)
                **1. 이 분석은 무엇인가요?** 환자님의 암세포 서열을 정상인의 단백질 지도와 비교했습니다.
                
                **2. 결과의 의미** 빨간색으로 표시된 **{neo_count}개의 서열**은 정상 세포에는 없는 암세포만의 특징입니다. 
                이 부분이 우리 몸의 면역세포가 암을 공격하게 만드는 '표적'이 될 수 있습니다.
                
                **3. 다음 단계** 이 후보들을 바탕으로 실제 백신 제조가 가능한지 의료진이 검토하게 됩니다.
                """)
