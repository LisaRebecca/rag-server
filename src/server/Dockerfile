FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi[all] requests httpx aioredis redis passlib python-jose psutil prometheus-client

EXPOSE 8000 8801 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]