# service1/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY config/ config/
COPY ../common/ common/

CMD ["python", "-m", "app.routes"]
