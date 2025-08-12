from fastapi import FastAPI, Request
import requests
import time

app = FastAPI()

@app.get("/")
def read_root():
    try:
        response = requests.get("http://llm-server:11434/ping")
        return {"llm_response": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "")

    start_time = time.perf_counter()
    resp = requests.post("http://llm-server:11434/generate", json={"prompt": prompt})
    elapsed_time = time.perf_counter() - start_time

    response_json = resp.json()
    # Convert elapsed time to minutes
    response_json["client_time_taken_minutes"] = round(elapsed_time / 60, 3)

    return response_json
