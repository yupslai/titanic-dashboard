# 타이타닉 생존자 분석 대시보드

## 프로젝트 개요
이 프로젝트는 타이타닉 호 승객 데이터를 분석하고 시각화하는 대시보드를 제공합니다. 
사용자는 다양한 요인별 생존율을 분석하고, 머신러닝 모델을 통해 생존 예측을 수행할 수 있습니다.

## 주요 기능
- 승객 데이터 기반 생존율 분석
- 다양한 시각화 차트 제공
- 머신러닝 기반 생존 예측
- 인터랙티브 필터링

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
```
titanic-dashboard/
└── titanic_dashboard/
    ├── app.py                  # 메인 Streamlit 앱
    ├── data/
    │   └── titanic.csv        # 원본 데이터
    ├── models/
    │   └── survival_predictor.py  # ML 모델
    ├── components/
    │   ├── sidebar.py         # UI 컴포넌트
    │   └── charts.py          # 시각화 함수
    ├── utils/
    │   └── data_loader.py     # 데이터 처리
    └── requirements.txt       # 패키지 목록
```

## 기여 방법
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 라이선스
MIT License 