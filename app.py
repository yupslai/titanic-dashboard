"""
타이타닉 대시보드 앱 진입점
실제 앱은 titanic_dashboard 폴더 안에 있습니다.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # 앱 경로 설정
    app_path = os.path.join("titanic_dashboard", "app.py")
    
    # 앱 실행
    print(f"타이타닉 대시보드를 실행합니다: {app_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path]) 