from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from typing import Dict

app = FastAPI(title="TinyLlama API", description="API for 4-bit Quantized TinyLlama-1.1B-Chat model")

# Global tokenizer and model
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model

    try:
        # Choose whether to load a fine-tuned LoRA adapter or just base 4-bit model
        use_finetuned_model = True

        model_cache_dir = "/models/pretrained"
        adapter_path = "/models/finetuned"  # only used if use_finetuned_model=True

        # BitsAndBytes 4-bit configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        base_model_id = "unsloth/tinyllama-bnb-4bit"

        print(f"Loading 4-bit base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=model_cache_dir)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            cache_dir=model_cache_dir,
            quantization_config=bnb_config,
            device_map="auto"
        )

        if use_finetuned_model:
            print(f"Applying LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model

        model.config.use_cache = False  # sometimes needed for compatibility

        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/ping")
async def ping() -> Dict[str, str]:
    return {"status": "LLM server is alive!"}


@app.post("/generate")
async def generate(request: Request) -> Dict[str, str]:
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 50)
        temperature = data.get("temperature", 0.7)

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].strip()

        return {"response": decoded}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
