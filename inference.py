import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_PATH = "../Models/mistral-7b-instruct"
FINETUNED_PATH = "./mistral-7b-devops-lora"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
).to(device)

prompt = "<s>[INST] You are a terse, highly technical DevOps & infra assistant. Answer directly.\n\nUser: Explain why a /16 VPC is usually split into multiple /24 subnets. [/INST]\n"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
