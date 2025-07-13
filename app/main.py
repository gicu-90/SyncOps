from fastapi import FastAPI, Request
import requests

app = FastAPI()

@app.get("/")
def read_root():
    # Check if LLM server is available
    try:
        response = requests.get("http://llm-server:11434/ping")
        return {"llm_response": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "")
    resp = requests.post("http://llm-server:11434/generate", json={"prompt": prompt})
    return resp.json()
