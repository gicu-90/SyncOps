FROM python:3.11-slim

WORKDIR /app
COPY server.py /app/server.py
COPY finetune.py /app/finetune.py

RUN apt-get update && apt-get install -y cmake build-essential && \
    pip install torch transformers accelerate peft bitsandbytes sentencepiece datasets fastapi uvicorn

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "11434", "--reload"]
