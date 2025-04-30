# titanic-dashboard
# 타이타닉 생존자 분석 대시보드

## 온라인 데모
**[타이타닉 대시보드 라이브 데모](https://titanic-dashboard-haxzaqgywet5aai2dentbb.streamlit.app/)** - 지금 바로 브라우저에서 사용해 보세요!

## 프로젝트 소개
이 프로젝트는 역사적인 타이타닉호 침몰 사고의 승객 데이터를 분석하고 시각화하는 대시보드입니다. 사용자는 승객의 다양한 특성(성별, 나이, 객실 등급 등)에 따른 생존율을 탐색하고, 머신러닝 모델을 통해 생존 가능성을 예측할 수 있습니다.

### 데이터셋
타이타닉 데이터셋은 다음과 같은 정보를 포함하고 있습니다:
- **PassengerId**: 승객 고유 식별 번호
- **Survived**: 생존 여부 (0 = 사망, 1 = 생존)
- **Pclass**: 객실 등급 (1 = 1등석, 2 = 2등석, 3 = 3등석)
- **Name**: 승객 이름
- **Sex**: 성별
- **Age**: 나이
- **SibSp**: 함께 탑승한 형제자매/배우자 수
- **Parch**: 함께 탑승한 부모/자녀 수
- **Ticket**: 티켓 번호
- **Fare**: 요금
- **Cabin**: 객실 번호
- **Embarked**: 승선 항구 (C = Cherbourg, Q = Queenstown, S = Southampton)

## 주요 기능
- **데이터 개요**: 기본 통계와 인구 통계학적 분포 시각화
- **생존율 분석**: 다양한 특성에 따른 생존율 비교
- **상관관계 분석**: 특성 간 상관관계 히트맵
- **생존 예측**: 머신러닝 기반 개인별 생존 확률 예측

## 기술 스택
- **프론트엔드**: Streamlit
- **데이터 처리**: Pandas, NumPy
- **시각화**: Plotly, Matplotlib, Seaborn
- **머신러닝**: Scikit-learn (RandomForest)
- **배포**: Streamlit Cloud, GitHub

## 설치 방법
1. 저장소 클론
```bash
git clone https://github.com/yupslai/titanic-dashboard.git
cd titanic-dashboard
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

## 실행 방법
```bash
# 중요: 앱은 titanic_dashboard 폴더 안에 있습니다
streamlit run titanic_dashboard/app.py
```

## 프로젝트 구조
titanic-dashboard/ └── titanic_dashboard/ ├── app.py # 메인 Streamlit 앱 ├── data/ │ └── titanic.csv # 원본 데이터 ├── models/ │ └── survival_predictor.py # ML 모델 ├── components/ │ ├── sidebar.py # UI 컴포넌트 │ └── charts.py # 시각화 함수 ├── utils/ │ └── data_loader.py # 데이터 처리 └── requirements.txt # 패키지 목록
