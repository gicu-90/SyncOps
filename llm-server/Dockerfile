FROM python:3.11-slim

WORKDIR /app
COPY server.py /app/server.py

RUN apt-get update && apt-get install -y cmake build-essential && \
    pip install llama-cpp-python fastapi uvicorn

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "11434", "--reload"]
