from fastapi import FastAPI, Request, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any

app = FastAPI(title="TinyLlama API", description="API for TinyLlama-1.1B-Chat model")

# Load model and tokenizer (do this once at startup)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@app.on_event("startup")
async def load_model():
    global tokenizer, model
    
    try:
        # Use local cache if available
        model_path = "/models/pretrained"
        
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cache_dir=model_path
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cache_dir=model_path
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded on {device}. Cache location: {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


@app.get("/ping")
async def ping() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "LLM server is alive!"}

@app.post("/generate")
async def generate(request: Request) -> Dict[str, str]:
    """
    Generate text based on the provided prompt
    
    Parameters:
    - prompt: The input text to generate from
    - max_new_tokens: Maximum number of tokens to generate (default: 50)
    - temperature: Sampling temperature (default: 0.7)
    """
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 50)
        temperature = data.get("temperature", 0.7)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Tokenize and generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response if it's included
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
            
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))