from fastapi import FastAPI, Request
from llama_cpp import Llama

app = FastAPI()

MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    output = llm(prompt, max_tokens=100)
    return {"response": output["choices"][0]["text"]}
