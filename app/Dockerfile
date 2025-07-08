# app/Dockerfile
FROM python:3.10-slim

RUN pip install requests GitPython PyGithub

COPY main.py /app/main.py
CMD ["python", "/app/main.py"]
