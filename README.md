# 🧠 SyncOps: FastAPI-Powered LLM Inference Server

This project provides a minimal LLM API server using **FastAPI** and **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)**.
It wraps a local quantized GGUF model (TinyLlama 1.1B) and serves it over HTTP with a `/generate` endpoint.

To run this, you need to manually download the TinyLlama GGUF model from Hugging Face:
[📦 Download TinyLlama GGUF Q4_K_M (637MB)](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
After downloading, place it here: SyncOps/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

Promt example:
curl -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{
  "prompt": "<|system|>\nYou are a helpful assistant.\n<|user|>\nWrite a Python function that adds two numbers.\n<|assistant|>",
  "temperature": 0.7,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "max_tokens": 100
}'

| Container     | Purpose                                               |
| ------------- | ----------------------------------------------------- |
| `app/`        | FastAPI app, orchestrates prompts and agents          |
| `llm-server/` | Loads base model + LoRA adapters, responds to prompts |
| `tuner/`      | Fine-tunes LoRA adapters from datasets                |
| Shared Volume | `/models/` for storing & loading LoRA adapters        |
