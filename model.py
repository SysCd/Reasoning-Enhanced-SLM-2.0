import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

# Paths
MODEL_PATH = "../Models/mistral-7b-instruct"
DATA_PATH = "reasoning_dataset.jsonl"
OUTPUT_DIR = "./mistral-7b-devops-lora"

# 1. Dataset helpers
def pick_field(ex, candidates, name):
    for k in candidates:
        if k in ex and ex[k] not in (None, ""):
            return ex[k]
    raise KeyError(f"No {name} field found in example. Keys: {list(ex.keys())}")

def format_example(ex):
    # Case 1: OpenAI-style chat format: {"messages": [{"role": "...", "content": "..."}]}
    if "messages" in ex:
        msgs = ex["messages"]

        system_msgs = [m["content"] for m in msgs if m.get("role") == "system"]
        user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
        assistant_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]

        system = (
            system_msgs[-1]
            if system_msgs
            else "You are a terse, highly technical DevOps & infra assistant. "
                 "Answer directly, prefer commands, avoid fluff."
        )
        user = user_msgs[-1] if user_msgs else ""
        assistant = assistant_msgs[-1] if assistant_msgs else ""

        return (
            "<s>[INST] "
            + system
            + "\n\nUser: "
            + user
            + " [/INST]\n"
            + assistant
            + "</s>"
        )

    # Case 2: legacy flat format with prompt/response-style keys
    prompt = pick_field(ex, ["prompt", "instruction", "input", "question"], "prompt")
    response = pick_field(ex, ["response", "output", "answer"], "response")

    return (
        "<s>[INST] You are a terse, highly technical DevOps & infra assistant. "
        "Answer directly, prefer commands, avoid fluff.\n\n"
        f"User: {prompt} [/INST]\n"
        f"{response}</s>"
    )

# 2. Load raw dataset
print("Loading dataset from", DATA_PATH)
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Map to 'text' that TRL will use
dataset = dataset.map(lambda ex: {"text": format_example(ex)})

# 3. Model on MPS (load on CPU fp16, then move)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,          # fp16; bf16 not supported on MPS
    low_cpu_mem_usage=False,      # avoid meta/bf16 path
)

model.to(device)

# 4. LoRA config (no bitsandbytes, pure PEFT)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Training config (standard TrainingArguments)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=200,
)

# 6. Trainer (no tokenizer arg for this TRL version)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
)


# 7. Train + save
trainer.train()
trainer.save_model(OUTPUT_DIR)

print("✅ Finished training. LoRA adapter saved to", OUTPUT_DIR)
