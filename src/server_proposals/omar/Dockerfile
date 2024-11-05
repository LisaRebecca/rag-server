FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi[all] requests httpx aioredis redis passlib python-jose

EXPOSE 8000 8801 8002

# CMD [ "uvicorn", "Main_FastAPI:app", "--host", "0.0.0.0", "--port", "80", "--ssl-keyfile", "./server.key", "--ssl-certfile", "./server.crt"]
CMD [ "uvicorn", "Main_FastAPI:app", "--host", "0.0.0.0", "--port", "8000"]