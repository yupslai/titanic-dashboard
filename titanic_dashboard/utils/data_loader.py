import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """타이타닉 데이터를 로드하고 기본적인 전처리를 수행합니다."""
    data_path = Path(__file__).parent.parent / 'data' / 'titanic.csv'
    df = pd.read_csv(data_path)
    return preprocess_data(df)

def preprocess_data(df):
    """데이터 전처리를 수행합니다."""
    # 복사본 생성
    df = df.copy()
    
    # 나이 결측치 처리
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # 객실 등급별 요금 결측치 처리
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Cabin 정보에서 Deck 추출
    df['Deck'] = df['Cabin'].str.extract('([A-Z])', expand=False)
    
    # 승선 항구 최빈값으로 결측치 처리
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 나이 그룹 생성
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 12, 20, 30, 40, 50, 60, np.inf],
        labels=['0-12세', '13-20세', '20-29세', '30-39세', '40-49세', '50-59세', '60세 이상']
    )
    
    # 요금 그룹 생성
    df['Fare_Group'] = pd.qcut(
        df['Fare'],
        q=4,
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    # 가족 크기 계산
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    
    return df

def get_feature_importance_data(model, feature_names):
    """모델의 특성 중요도를 반환합니다."""
    importance = model.feature_importances_
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

def prepare_prediction_data(df):
    """예측을 위한 데이터를 준비합니다."""
    # 수치형 특성
    numeric_features = ['Age', 'Fare', 'Family_Size']
    
    # 범주형 특성
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Is_Alone']
    
    # 가능한 모든 값 정의 (학습 및 예측 데이터의 일관성을 위해)
    category_values = {
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female'],
        'Embarked': ['C', 'Q', 'S'],
        'Is_Alone': [0, 1]
    }
    
    # 범주형 특성에 대한 더미 변수 생성
    dummy_dfs = []
    for feature in categorical_features:
        # 결측값 먼저 처리
        if feature in df.columns:
            df[feature] = df[feature].fillna(df[feature].mode()[0] if len(df) > 1 else category_values[feature][0])
            
            # 모든 가능한 값에 대한 더미 변수 생성
            dummy = pd.get_dummies(df[feature], prefix=feature)
            
            # 누락된 범주 추가
            for val in category_values[feature]:
                col_name = f"{feature}_{val}"
                if col_name not in dummy.columns:
                    dummy[col_name] = 0
                    
            dummy_dfs.append(dummy)
    
    # 수치형 특성과 범주형 특성 결합
    numeric_df = df[numeric_features].copy() if all(f in df.columns for f in numeric_features) else pd.DataFrame()
    
    # 누락된 수치형 특성 처리
    for feature in numeric_features:
        if feature not in numeric_df.columns:
            numeric_df[feature] = 0
    
    # 최종 결합
    df_final = pd.concat([numeric_df] + dummy_dfs, axis=1)
    
    return df_final 