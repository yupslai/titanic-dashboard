import streamlit as st

def show_sidebar():
    """사이드바 UI를 구성합니다."""
    with st.sidebar:
        st.title('타이타닉 생존자 분석')
        
        # 분석 섹션 선택
        analysis_type = st.radio(
            "분석 유형 선택",
            ["데이터 개요", "생존율 분석", "상관관계 분석", "생존 예측"]
        )
        
        # 필터링 옵션
        st.subheader("데이터 필터링")
        
        # 승객 등급 필터
        pclass = st.multiselect(
            "승객 등급",
            options=[1, 2, 3],
            default=[1, 2, 3]
        )
        
        # 성별 필터
        sex = st.multiselect(
            "성별",
            options=['male', 'female'],
            default=['male', 'female']
        )
        
        # 나이 범위 필터
        age_range = st.slider(
            "나이 범위",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )
        
        # 승선 항구 필터
        embarked = st.multiselect(
            "승선 항구",
            options=['C', 'Q', 'S'],
            default=['C', 'Q', 'S']
        )
        
        # 요금 범위 필터
        fare_range = st.slider(
            "요금 범위",
            min_value=0,
            max_value=600,
            value=(0, 600)
        )
        
        # 가족 크기 필터
        family_size = st.slider(
            "가족 크기",
            min_value=1,
            max_value=11,
            value=(1, 11)
        )
        
    # 필터링 조건 반환
    filters = {
        'analysis_type': analysis_type,
        'pclass': pclass,
        'sex': sex,
        'age_range': age_range,
        'embarked': embarked,
        'fare_range': fare_range,
        'family_size': family_size
    }
    
    return filters

def apply_filters(df, filters):
    """선택된 필터를 데이터프레임에 적용합니다."""
    mask = (
        (df['Pclass'].isin(filters['pclass'])) &
        (df['Sex'].isin(filters['sex'])) &
        (df['Age'].between(*filters['age_range'])) &
        (df['Embarked'].isin(filters['embarked'])) &
        (df['Fare'].between(*filters['fare_range'])) &
        (df['Family_Size'].between(*filters['family_size']))
    )
    return df[mask]

def show_prediction_inputs():
    """생존 예측을 위한 입력 폼을 표시합니다."""
    st.sidebar.subheader("생존 예측")
    
    # 기본 정보 입력
    pclass = st.sidebar.selectbox("승객 등급", options=[1, 2, 3])
    sex = st.sidebar.selectbox("성별", options=['male', 'female'])
    age = st.sidebar.number_input("나이", min_value=0, max_value=100, value=30)
    
    # 가족 정보 입력
    sibsp = st.sidebar.number_input("형제/배우자 수", min_value=0, max_value=10, value=0)
    parch = st.sidebar.number_input("부모/자녀 수", min_value=0, max_value=10, value=0)
    
    # 승선 정보 입력
    fare = st.sidebar.number_input("요금", min_value=0.0, max_value=600.0, value=32.2)
    embarked = st.sidebar.selectbox("승선 항구", options=['C', 'Q', 'S'])
    
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
    
    return inputs 