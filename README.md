# Reasoning-Enhanced SLM 2.0

A Cognitive-Architecture Approach to Fine-Tuning Small Language Models  
Using First-Principles Reasoning, Custom Blueprint Generation, and LoRA Training  
(Model: Mistral-7B-Instruct, Local Training on Apple Silicon MPS)

---

## OVERVIEW

Reasoning-Enhanced SLM 2.0 is an applied research project exploring how to
modify the _reasoning behavior_ of an open-weight LLM using:

• A structured reasoning architecture  
• First-principles decomposition  
• Blueprint-driven dataset generation  
• Parameter-efficient LoRA fine-tuning  
• Local training on Apple Silicon (MPS, fp16)

Unlike most fine-tuning projects that simply teach a model new content,
this project teaches the model _how to think_ in a specific domain style:
terse, accurate, technical, DevOps-oriented reasoning.

---

## SECTION 1 — REASONING ARCHITECTURE (THE CORE INNOVATION)

This project does not rely on unstructured scraping or manual prompt-writing.
Instead, it introduces a **Reasoning Blueprint System** — a method for converting
domain knowledge into structured reasoning patterns that LLMs can learn reliably.

The system is visually represented in three diagram groups included below.

==============================

1. # First Principles Thinking (Physics Example)

This diagram shows how a complex domain (physics) can be reduced into:

• Final Principles  
• Explanatory principles  
• Core logic principles  
• Compressed reasoning blocks

This demonstrates a universal method for converting high-level knowledge  
into stable, reusable reasoning logic for machine learning datasets.

### First Principles Reasoning Architecture

![First Principles Thinking for AI](diagrams/first_principles_thinking_ai.jpeg)

# ============================== 2. Physics Systems Cognition

A hierarchical decomposition of physics into conceptual layers:

• Classical Physics → Evolution 1  
• Modern Physics → Evolution 2  
• Theoretical Physics → Evolution 3  
• Applied & Interdisciplinary Physics → Evolution 4

This method shows how any domain can be structured into a curriculum-like  
hierarchy suitable for progressive training or dataset generation.

### Physics Systems Cognition

This diagram represents a hierarchical system-level decomposition of the entire domain of physics:
from classical foundations → modern physics → theoretical frameworks → interdisciplinary extensions.

It demonstrates how complex knowledge can be structured into conceptual layers suitable for model
training, curriculum-style dataset construction, and reasoning architecture design.

![Physics Systems Cognition](diagrams/physics_systems_cognition.jpeg)

# ============================== 3. Data Generation Blueprint Architecture

This diagram illustrates the full data pipeline used in the project:

1. Create a domain-specific reasoning framework
2. Feed the architecture to an LLM
3. LLM converts it into a structured blueprint
4. Blueprint undergoes pattern-matching expansion
5. Output transforms into a supervised fine-tuning dataset

This system allows scalable, structured reasoning data creation  
without manually writing thousands of samples.

### Custom Data Generation for Model Fine-Tuning

This diagram illustrates the full reasoning-to-dataset pipeline:
framework → blueprint → pattern expansion → JSONL dataset for supervised training.

![Custom Data Generation for Model Fine-Tuning](diagrams/custom_data_generation_finetuning.jpeg)

---

## SECTION 2 — TRAINING PIPELINE

1. Load model (Mistral-7B-Instruct) on CPU in fp16
2. Move model to MPS manually (bfloat16 unsupported on Apple Silicon)
3. Load tokenizer (no fast version; uses SentencePiece)
4. Inject LoRA adapters via PEFT
5. Use TRL’s SFTTrainer to learn behavior from custom dataset
6. Save LoRA weights as lightweight adapters

Training runs fully on a MacBook with Apple M4/M3/M2 chips.

---

## SECTION 3 — DATASET FORMAT

The dataset is stored in `reasoning_dataset.jsonl`, using an OpenAI-style
message format:

{
"messages": [
{"role": "system", "content": "..."},
{"role": "user", "content": "..."},
{"role": "assistant", "content": "..."}
]
}

During preprocessing, each example is transformed into the Mistral
instruction-tuning format:

<s>[INST] system text + user question [/INST] assistant answer </s>

This preserves conversational alignment while teaching precise reasoning steps.

---

## SECTION 4 — LOADED MODEL AND TRAINING OUTPUT

Model: Mistral-7B-Instruct-v0.3  
Adapter: LoRA (r=16, alpha=32, dropout=0.05)  
Device: Apple MPS (fp16)

Output is stored in:

mistral-7b-devops-lora/
│
├── adapter_config.json  
├── adapter_model.bin  
└── tokenizer/ (copied for convenient inference)

Use PEFT to merge or apply adapters during inference.

---

## SECTION 5 — EXAMPLE POST-FINETUNE OUTPUT

Prompt:
Explain why a /16 VPC is usually split into multiple /24 subnets.

Model Output (LoRA-tuned):
To reduce broadcast domain size, minimize ARP noise, and segment workloads cleanly.
Smaller /24 blocks isolate services (public, private, DB, mgmt), simplify routing,
and support scalable multi-AZ layouts.

This demonstrates the intended terse, precise DevOps reasoning style.

---

## SECTION 6 — WHY THIS PROJECT MATTERS

Most fine-tuning projects only adjust _content_.  
This project adjusts _cognition_.

It demonstrates:

• First-principles reasoning compression  
• System-level decomposition of knowledge  
• Automated blueprint-to-dataset generation  
• Local LoRA-based behavioral alignment  
• Architecture-level thinking for ML systems

This is closer to _applied alignment research_ than typical ML coursework.

---

## SECTION 7 — FUTURE WORK

• Add DPO for preference-based refinement  
• Add reward modeling for multi-step reasoning  
• Expand blueprint generator into an automated framework  
• Evaluate reasoning drift, consistency, and stability  
• Fine-tune multiple domains (physics, CS, finance, DevOps)

---

## LICENSE

MIT License
