import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# ID of the base model from Hugging Face Hub or local cache
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Local directory to cache pretrained models and tokenizer files
cache_dir = "/models/pretrained"

# Directory where the fine-tuned model and tokenizer will be saved
output_dir = "/models/finetuned"

# Load tokenizer and base causal language model (AutoModelForCausalLM)
# tokenizer: responsible for converting text to tokens and vice versa
# model: the pretrained language model to be fine-tuned
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

# LoRA (Low-Rank Adaptation) configuration parameters for fine-tuning:
# r: LoRA rank, controls the size of the low-rank adaptation matrices
# lora_alpha: scaling factor for LoRA layers
# lora_dropout: dropout rate applied inside LoRA layers to regularize training
# bias: type of bias handling ("none" means no bias adjustment)
# task_type: task-specific type, here it's CAUSAL_LM for language modeling
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply the LoRA configuration to the base model, creating a LoRA-adapted model
model = get_peft_model(model, lora_config)

# Print the count and percentage of parameters that will be trained (LoRA params)
model.print_trainable_parameters()

# Flag to control which dataset to load:
# True: load a custom JSON dataset from local path "/data/train.jsonl"
# False: load a public dataset "Abirate/english_quotes" from Hugging Face Hub (limited to first 1000 samples)
is_json_from_custom_sourfcese = False
if is_json_from_custom_sourfcese:
    dataset = load_dataset("json", data_files="/data/train.jsonl", split="train")
else:
    dataset = load_dataset("Abirate/english_quotes", split="train[:1000]")

# Function to tokenize each example in the dataset
# example: a dictionary with key "quote" containing the text to tokenize
# Returns tokenized input with padding and truncation to max length 128 tokens
def tokenize(example):
    return tokenizer(example["quote"], padding="max_length", truncation=True, max_length=128)

# Apply the tokenization function to the entire dataset in batches for efficiency
tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments specifying how training is configured:
training_args = TrainingArguments(
    output_dir=output_dir,                  # Where to save checkpoints and final model
    per_device_train_batch_size=2,          # Batch size per device (GPU/CPU)
    gradient_accumulation_steps=4,          # Accumulate gradients over 4 steps before optimizer step (effectively batch size 8)
    num_train_epochs=3,                      # Number of full passes through the dataset
    logging_dir="/logs",                     # Directory to save training logs
    save_strategy="epoch",                   # Save checkpoint after every epoch
    logging_steps=10,                        # Log training info every 10 steps
    report_to="none",                        # Disable reporting to external tools (e.g., WandB)
    fp16=torch.cuda.is_available(),         # Use mixed precision (fp16) if a GPU with CUDA is available
)

# Initialize the Trainer object, which handles training loop, evaluation, etc.
trainer = Trainer(
    model=model,                             # The LoRA-adapted model to train
    args=training_args,                      # Training configuration
    train_dataset=tokenized_dataset,        # Dataset to train on (tokenized)
    tokenizer=tokenizer,                     # Tokenizer used (for padding & decoding)
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    # Data collator batches data and applies masking; mlm=False means causal LM (not masked LM)
)

# Start the training process (runs for num_train_epochs)
trainer.train()

# Save the fine-tuned LoRA model and tokenizer to output_dir for later use/deployment
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Fine-tuned LoRA model saved at {output_dir}")
