FROM python:3.11-slim

WORKDIR /app

COPY main.py /app/main.py

RUN pip install fastapi uvicorn requests

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
