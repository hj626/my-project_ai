import streamlit as st
import numpy as np
import joblib
import pandas as pd

from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="인스탁스 판매량 예측",
    layout="centered"
)

# 모델불러오기
@st.cache_resource
def load_model():
    try:
        model = joblib.load("instax_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"모델 파일을 찾을 수 없습니다.: {e}")
        st.stop()
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()
    
model, scaler = load_model()


st.title("머신러닝 기반 인스탁스 판매량 예측 시스템") #제목
st.write("할인 금액과 판매월을 입력하면 예상 판매량(qty)을 예측합니다.") #설명

# 사이드바에 정보 표시
with st.sidebar:
    st.header("모델 정보")
    st.write("알고리즘: ")
    st.write("입력 변수 : 할인급액/판매월 ")
    st.write("출력: 예상 판매량")
    st.divider()
    st.write("제작자: 강혜정")
    st.write("제작일: 2025.12.29")
    st.divider()
    
    
st.subheader(" 예측 정보 입력")

# 입력 풀 생성
col1, col2 = st.columns(2)

with col1:
    discount = st.number_input(
        "할인금액 ",
        min_value=0.0,
        max_value=1000000.0,
        value=0.0,
        step=1000.0,
        help="할인 금액을 인도네시아 루피아(IDR)로 입력하세요"
        )

with col2:
    month = st.number_input(
        "판매 월 (1~12)",
        min_value=1,
        max_value=12,
        value=datetime.now().month, #현재월을 기본값으로 설정하는것
        help="판매 예상 월을 선택하세요 (1~12)"
    )
    
st.divider()


# 에측버튼
if st.button("판매량 예측", type="primary", use_container_width=True):
    try:
        # 입력 데이터 준비
        input_data = np.array([[discount, month]])
        
        # 스케일링
        input_scaled = scaler.transform(input_data)
        
        # 예측
        prediction = model.predict(input_scaled)[0]
        
        # 결과표시
        st.success(f"예상 판매량: 약{int(prediction)} 개")
        
        
        #  # 신뢰도 표시 (예측값 기준)
        # if prediction >= 3:
        #     confidence = "높음 ⭐⭐⭐"
        # elif prediction >= 2:
        #     confidence = "중간 ⭐⭐"
        # else:
        #     confidence = "낮음 ⭐"
        
        # st.info(f"예측 신뢰도: {confidence}")
        
        
        # 추가 정보 표시
        with st.expander("입력정보 확인"):
            
            st.write(f"- 할인금액: {int(discount)}")
            st.write(f"- 판매 월: {month}월")
            st.write(f"- 예측 판매량: {prediction:.2f} -> {int(round(prediction))} 개")
        
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        st.write("입력 값을 확인하시고 다시 시도해주세요.")



#   # 6-4. 예측 결과 표시
#         st.success(f"### 🎯 예상 판매량: **약 {int(round(prediction))} 개**")
        
#         # 신뢰도 표시 (예측값 기준)
#         if prediction >= 3:
#             confidence = "높음 ⭐⭐⭐"
#         elif prediction >= 2:
#             confidence = "중간 ⭐⭐"
#         else:
#             confidence = "낮음 ⭐"
        
#         st.info(f"예측 신뢰도: {confidence}")
        
#         # 6-5. 입력 정보 확인 (펼침 가능한 영역)
#         with st.expander("📊 입력 정보 및 예측 상세"):
#             # 입력값을 테이블 형태로 표시
#             input_df = pd.DataFrame({
#                 '항목': ['할인금액', '판매월', '예측 판매량 (정확값)', '예측 판매량 (반올림)'],
#                 '값': [
#                     f"{int(discount):,} IDR",
#                     f"{month}월",
#                     f"{prediction:.2f} 개",
#                     f"{int(round(prediction))} 개"
#                 ]
#             })
#             st.table(input_df)
            
#             # 해석 가이드
#             st.markdown("""
#             **해석 가이드:**
#             - 할인이 클수록 판매량이 증가할 가능성이 있습니다
#             - 월별 계절성(예: 연말, 여름방학)이 판매에 영향을 줍니다
#             - 예측값은 과거 판매 패턴을 기반으로 합니다
#             """)
        
#         # 6-6. 추가 인사이트
#         st.markdown("---")
#         st.markdown("### 💡 예측 인사이트")
        
#         # 할인 효과 분석
#         if discount > 0:
#             st.write(f"✅ 할인 금액 {int(discount):,} IDR 적용")
#             st.write("💰 할인이 적용되어 판매량 증가가 예상됩니다")
#         else:
#             st.write("ℹ️ 할인 미적용")
#             st.write("📈 할인을 적용하면 판매량이 더 증가할 수 있습니다")
        
#         # 월별 인사이트
#         if month in [5, 6, 7, 8]:
#             st.write("🌞 여름 시즌으로 카메라 수요가 높을 수 있습니다")
#         elif month in [11, 12]:
#             st.write("🎄 연말 시즌으로 선물 수요가 높을 수 있습니다")
#         else:
#             st.write(f"📅 {month}월은 평균적인 판매 시즌입니다")
        
#     except Exception as e:
#         # 오류 발생 시 사용자에게 친절한 메시지 표시
#         st.error(f"⚠️ 예측 중 오류가 발생했습니다: {e}")
#         st.write("입력값을 확인하고 다시 시도해주세요.")
#         st.write("문제가 계속되면 관리자에게 문의하세요.")

# # ============================================
# # 7. 푸터 - 추가 정보
# # ============================================
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: gray; font-size: 0.9em;'>
#     <p>본 시스템은 머신러닝 기반 판매량 예측 서비스입니다</p>
#     <p>예측 결과는 참고용이며, 실제 판매량과 차이가 있을 수 있습니다</p>
# </div>
# """, unsafe_allow_html=True)