import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# ---------------------------
# Config
# ---------------------------
base_model_id = "unsloth/tinyllama-bnb-4bit"
cache_dir = "/models/pretrained"
output_dir = "/models/finetuned"
dataset_path = "/models/hub/datasets--Abirate--english_quotes/snapshots/7b544c4920a8be268b48b403c188acf0a462051b/quotes.jsonl"

per_device_train_batch_size = 2
gradient_accumulation_steps = 4
num_train_epochs = 3
learning_rate = 2e-5
max_length = 256

# ---------------------------
# BitsAndBytes config with GPU check
# ---------------------------
try:
    import bitsandbytes as bnb
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    print("✅ bitsandbytes loaded with GPU quantization support.")
except Exception as e:
    bnb_config = None
    print("⚠️ bitsandbytes GPU quantization unavailable. Running without 4-bit.")

# ---------------------------
# Load tokenizer and model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    cache_dir=cache_dir,
    quantization_config=bnb_config if bnb_config else None,
    device_map="auto"
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# ---------------------------
# LoRA config
# ---------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------
# Dataset & tokenization
# ---------------------------
dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize_fn(example):
    question_about_quote = f"Who said: \"{example['quote']}\"?"
    answer_about_quote = example["author"]
    question_about_author = f"What is a famous quote by {example['author']}?"
    answer_about_author = f"\"{example['quote']}\""

    text = (
        f"<|user|>\n{question_about_quote}\n<|assistant|>\n{answer_about_quote}\n\n"
        f"<|user|>\n{question_about_author}\n<|assistant|>\n{answer_about_author}"
    )

    tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Remove original columns so Trainer gets only tokenized tensors
tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=False,
    remove_columns=dataset.column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------------------
# Training
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    logging_dir="/logs",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(),
    learning_rate=learning_rate,
    warmup_steps=100,
    optim="adamw_torch",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ---------------------------
# Save LoRA adapters only
# ---------------------------
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ LoRA fine-tuning completed. Adapters saved at {output_dir}")
