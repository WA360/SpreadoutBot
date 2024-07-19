# 베이스 이미지로 Python 3.12 사용
FROM python:3.12

# 작업 디렉터리를 설정
WORKDIR /app

# 필요 패키지 목록 복사
COPY requirements.txt /app/

# pip 업그레이드
RUN pip install --upgrade pip

# 패키지 설치
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

RUN pip install pymupdf

RUN pip install -U python-dotenv

RUN pip install flask_mysqldb

RUN pip install anthropic

# 소스 코드 복사
COPY . /app/

# 포트를 노출
EXPOSE 3100

# Django 서버 실행 명령
CMD ["python", "app.py"]