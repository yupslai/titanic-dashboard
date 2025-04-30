import streamlit as st
import pandas as pd

from utils.data_loader import load_data, prepare_prediction_data
from components.sidebar import show_sidebar, apply_filters, show_prediction_inputs
from components.charts import (
    plot_survival_by_category, plot_age_distribution,
    plot_correlation_matrix, plot_feature_importance,
    plot_survival_prediction, plot_fare_vs_age,
    plot_survival_metrics
)
from models.survival_predictor import SurvivalPredictor

# 페이지 설정
st.set_page_config(
    page_title="타이타닉 생존자 분석",
    page_icon="🚢",
    layout="wide"
)

# 데이터 로드
@st.cache_data
def get_data():
    return load_data()

# 메인 함수
def main():
    # 데이터 로드
    df = get_data()
    
    # 사이드바 표시 및 필터 적용
    filters = show_sidebar()
    filtered_df = apply_filters(df, filters)
    
    # 분석 유형에 따른 화면 표시
    if filters['analysis_type'] == "데이터 개요":
        show_data_overview(filtered_df)
    elif filters['analysis_type'] == "생존율 분석":
        show_survival_analysis(filtered_df)
    elif filters['analysis_type'] == "상관관계 분석":
        show_correlation_analysis(filtered_df)
    else:  # "생존 예측"
        show_survival_prediction(df)

def show_data_overview(df):
    st.title("📊 데이터 개요")
    
    # 기본 통계
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 승객 수", len(df))
    with col2:
        survival_rate = (df['Survived'].mean() * 100).round(2)
        st.metric("전체 생존율", f"{survival_rate}%")
    with col3:
        st.metric("평균 나이", f"{df['Age'].mean():.1f}세")
    
    # 데이터 미리보기
    st.subheader("데이터 미리보기")
    st.dataframe(df.head())
    
    # 기본 분포 차트
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_survival_by_category(df, 'Sex', '성별 생존율'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_survival_by_category(df, 'Pclass', '객실 등급별 생존율'), use_container_width=True)

def show_survival_analysis(df):
    st.title("🔍 생존율 분석")
    
    # 다양한 범주별 생존율
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_survival_by_category(df, 'Age_Group', '연령대별 생존율'), use_container_width=True)
        st.plotly_chart(plot_survival_by_category(df, 'Embarked', '승선 항구별 생존율'), use_container_width=True)
    
    with col2:
        st.plotly_chart(plot_survival_by_category(df, 'Fare_Group', '요금 그룹별 생존율'), use_container_width=True)
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)
    
    # 요금과 나이 관계
    st.plotly_chart(plot_fare_vs_age(df), use_container_width=True)

def show_correlation_analysis(df):
    st.title("🔗 상관관계 분석")
    
    # 분석할 특성 선택
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    selected_features = st.multiselect(
        "분석할 특성 선택",
        options=numeric_cols,
        default=['Survived', 'Pclass', 'Age', 'Fare', 'Family_Size']
    )
    
    if selected_features:
        st.plotly_chart(plot_correlation_matrix(df, selected_features), use_container_width=True)
    else:
        st.warning("분석할 특성을 선택해주세요.")

def show_survival_prediction(df):
    st.title("🔮 생존 예측")
    
    # 예측 모델 초기화 및 학습
    if 'model' not in st.session_state:
        with st.spinner("모델 학습 중..."):
            model = SurvivalPredictor()
            X = prepare_prediction_data(df)
            y = df['Survived']
            metrics = model.train(X, y)
            st.session_state.model = model
            st.session_state.metrics = metrics
    
    # 모델 성능 지표 표시
    st.subheader("모델 성능")
    st.plotly_chart(plot_survival_metrics(st.session_state.metrics), use_container_width=True)
    
    # 특성 중요도 표시
    st.subheader("특성 중요도")
    importance_df = st.session_state.model.get_feature_importance()
    st.plotly_chart(plot_feature_importance(importance_df), use_container_width=True)
    
    # 생존 예측
    st.subheader("생존 예측하기")
    
    # 입력 폼을 사이드바가 아닌 메인 화면에 표시
    col1, col2 = st.columns(2)
    
    with col1:
        # 기본 정보 입력
        pclass = st.selectbox("승객 등급", options=[1, 2, 3])
        sex = st.selectbox("성별", options=['male', 'female'])
        age = st.number_input("나이", min_value=0, max_value=100, value=30)
    
    with col2:
        # 가족 정보 입력
        sibsp = st.number_input("형제/배우자 수", min_value=0, max_value=10, value=0)
        parch = st.number_input("부모/자녀 수", min_value=0, max_value=10, value=0)
    
        # 승선 정보 입력
        fare = st.number_input("요금", min_value=0.0, max_value=600.0, value=32.2)
        embarked = st.selectbox("승선 항구", options=['C', 'Q', 'S'])
    
    # 입력값 반환
    inputs = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
        'Family_Size': sibsp + parch + 1,
        'Is_Alone': 1 if (sibsp + parch == 0) else 0
    }
    
    # 예측 버튼을 메인 화면에 표시
    if st.button("예측하기"):
        # 입력 데이터 준비
        input_df = pd.DataFrame([inputs])
        X = prepare_prediction_data(input_df)
        
        try:
            # 예측 수행
            survival_prob = st.session_state.model.predict(X)[0]
            survival_status = "생존" if survival_prob > 0.5 else "사망"
            
            # 결과 표시
            st.success(f"생존 확률: {survival_prob:.1%} ({survival_status})")
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
            st.info("데이터 형식을 확인하고 다시 시도해 주세요.")

if __name__ == "__main__":
    main() 