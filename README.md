# 🔁 SyncOps — AI-Powered API Gateway with LoRA Fine-Tuned LLM

SyncOps is a modular microservice system that connects a FastAPI-based app with a LoRA fine-tuned TinyLlama model server. Everything runs in Docker, and setup is one command away.

---

## 📦 What's Inside

### `app/`
A FastAPI service acting as the API gateway. It:
- Serves an `/` health check that pings the LLM server
- Accepts prompts via `/chat` and forwards them to the LLM server

### `llm-server/`
A FastAPI-based server hosting:
- A pre-trained TinyLlama model (via Hugging Face)
- A fine-tuning script (`finetune.py`) using PEFT + LoRA
- Endpoints to ping or generate completions

### `models/`
Volume-mounted storage for:
- Pretrained models
- Finetuned adapters
- Datasets for training

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/gicu-90/SyncOps.git
cd SyncOps
```

### 2. Install Docker (if not installed)

Run the following command to install the latest Docker on Ubuntu:

```bash
curl -fsSL https://get.docker.com | sh
```

You may need to log out and log back in or run Docker commands with `sudo`.

### 3. Build and run containers with Docker Compose CLI

```bash
docker compose up --build
```

This will build and start the app and LLM server containers.

---

## 🧠 How It Works

- The app (`app/main.py`) exposes `/chat`, forwarding prompts to the LLM server. 
- The LLM server (`llm-server/server.py`) loads TinyLlama from local cache and runs text generation.
- The `finetune.py` script lets you fine-tune TinyLlama with LoRA on custom or HF datasets.

---

## 🧪 Example Usage

```bash
curl -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{
  "prompt": "<|system|>\nYou are a helpful assistant.\n<|user|>\n Say something in french.\n<|assistant|>",
  "temperature": 0.7,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "max_tokens": 100
}'
```

---

## 🔧 Fine-Tuning

To fine-tune the model:

```bash
docker exec -it <llm-container-name> python /app/finetune.py
```

Fine-tuned models will be saved to `/models/finetuned`.

---

## 📁 Project Structure

```text
.
├── app/                  # API gateway (FastAPI)
├── llm-server/           # TinyLlama model server + fine-tuning
├── models/               # Pretrained + fine-tuned models, datasets
├── docker-compose.yml    # Service orchestration
└── README.md             # You are here
```

---

## 🤝 Contributing

PRs are welcome! If you’d like to add agents, adapters, or more features, feel free to fork and contribute.

---

## 📜 License

MIT License — use, modify, or deploy freely.

---

## 🧠 Credits

- [TinyLlama](https://huggingface.co/TinyLlama) for the base model
- Hugging Face Transformers + Datasets
- PEFT & LoRA for efficient fine-tuning