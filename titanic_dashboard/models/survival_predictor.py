import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

class SurvivalPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
    def prepare_features(self, X):
        """특성 데이터 전처리"""
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        return X_scaled
        
    def train(self, X, y):
        """모델 학습"""
        self.feature_names = X.columns.tolist()
        X_scaled = self.prepare_features(X)
        
        # 학습/검증 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 모델 학습
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        y_pred = self.model.predict(X_val)
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        return self.metrics
    
    def predict(self, X):
        """생존 확률 예측"""
        if self.feature_names is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 입력 데이터에 필요한 모든 특성이 있는지 확인하고 처리
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            # 누락된 특성에 대해 0으로 채우기
            for feature in missing_features:
                X[feature] = 0
                
        # 학습에 사용된 특성만 선택하고 순서 맞추기
        X_selected = X[self.feature_names]
        X_scaled = self.prepare_features(X_selected)
        
        # 생존 확률 예측
        survival_prob = self.model.predict_proba(X_scaled)[:, 1]
        return survival_prob
    
    def get_feature_importance(self):
        """특성 중요도 반환"""
        if self.feature_names is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def get_metrics(self):
        """모델 성능 지표 반환"""
        if not self.metrics:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.metrics 