from fastapi import FastAPI, Request
from llama_cpp import Llama

MODEL_PATH = "/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # from mounted volume

llm = Llama(model_path=MODEL_PATH, n_ctx=512)

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "LLM server is alive!"}

@app.post("/generate")
async def generate(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "")
    result = llm(prompt, max_tokens=128)
    return {"response": result["choices"][0]["text"]}
