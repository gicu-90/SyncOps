FROM python:3.10-slim

RUN apt-get update && apt-get install -y cmake build-essential && \
    pip install llama-cpp-python fastapi uvicorn

WORKDIR /app

COPY server.py /app/server.py
COPY models/ /app/models/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "11434"]
