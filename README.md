# Reasoning-Enhanced SLM 2.0

A Cognitive-Architecture Approach to Fine-Tuning Small Language Models  
Using First-Principles Reasoning, Custom Blueprint Generation, and LoRA Training  
(Model: Mistral-7B-Instruct, Local Training on Apple Silicon MPS)

---

## OVERVIEW

Reasoning-Enhanced SLM 2.0 is an applied research project exploring how to
modify the reasoning behavior of an open-weight LLM using:

- A structured reasoning architecture
- First-principles decomposition
- Blueprint-driven dataset generation
- Parameter-efficient LoRA fine-tuning
- Local training on Apple Silicon (MPS, fp16)

Unlike most fine-tuning projects that simply teach a model new content,
this project teaches the model how to think in a specific domain style:
terse, accurate, technical, DevOps-oriented reasoning.

---

## SECTION 1 — REASONING ARCHITECTURE (CORE INNOVATION)

This project introduces a **Reasoning Blueprint System** — a method for converting
domain knowledge into structured reasoning patterns that LLMs can learn reliably.

The system is visually represented in three diagram groups.

---

### 1. First Principles Thinking (Physics Example)

This diagram shows how a complex domain (physics) can be reduced into:

- Final principles
- Explanatory principles
- Core logic principles
- Compressed reasoning blocks

This provides a universal method for turning high-level knowledge into reusable
reasoning logic for machine learning datasets.

#### First Principles Reasoning Architecture

![First Principles Thinking for AI](Diagrams/first_principles_thinking_ai.jpeg)

---

### 2. Physics Systems Cognition

A hierarchical decomposition of physics into conceptual layers:

- Classical Physics → Evolution 1
- Modern Physics → Evolution 2
- Theoretical Physics → Evolution 3
- Applied & Interdisciplinary Physics → Evolution 4

This turns a complex domain into a structured curriculum suitable for model training.

#### Physics Systems Cognition Diagram

This diagram demonstrates how domain knowledge can be layered to support curriculum-style dataset construction and reasoning architecture design.

![Physics Systems Cognition](Diagrams/physics_systems_cognition.jpeg)

---

### 3. Data Generation Blueprint Architecture

This diagram illustrates the full data pipeline used in this project:

1. Create a domain-specific reasoning framework
2. Feed the architecture to an LLM
3. The LLM converts it into a structured blueprint
4. The blueprint undergoes pattern expansion
5. The output becomes a supervised fine-tuning dataset

This system allows scalable production of structured reasoning data.

#### Custom Data Generation for Model Fine-Tuning

![Custom Data Generation for Model Fine-Tuning](Diagrams/custom_data_generation_finetuning.jpeg)

---

## SECTION 2 — TRAINING PIPELINE

Training steps:

1. Load Mistral-7B-Instruct on CPU in fp16
2. Move model to MPS manually (bfloat16 unsupported on Apple Silicon)
3. Load tokenizer (SentencePiece-based, non-fast)
4. Inject LoRA adapters using PEFT
5. Train using TRL's SFTTrainer
6. Save LoRA adapter weights

Training runs fully on Apple Silicon (M4/M3/M2) using MPS acceleration.

---

## SECTION 3 — DATASET FORMAT

Dataset stored in `reasoning_dataset.jsonl`, using an OpenAI-style message format:

{
"messages": [
{"role": "system", "content": "..."},
{"role": "user", "content": "..."},
{"role": "assistant", "content": "..."}
]
}

---

## SECTION 4 — LOADED MODEL AND TRAINING OUTPUT

- Model: Mistral-7B-Instruct-v0.3
- Adapter: LoRA (r=16, alpha=32, dropout=0.05)
- Device: Apple MPS (fp16)

Output directory:

mistral-7b-devops-lora/
│
├── adapter_config.json  
├── adapter_model.bin  
└── tokenizer/

Adapters can be merged or applied at inference time using PEFT.

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

Most fine-tuning projects adjust **content**.  
This project adjusts **cognition**.

It demonstrates:

- First-principles reasoning compression
- System-level decomposition of knowledge
- Automated blueprint-to-dataset generation
- Local LoRA-based behavioral alignment
- Architecture-level thinking for ML systems

This aligns more with _applied alignment research_ than standard ML fine-tuning.

---

## SECTION 7 — FUTURE WORK

- Add DPO for preference-based refinement
- Add reward modeling for multi-step reasoning
- Expand blueprint generator into an automated system
- Evaluate reasoning drift and alignment stability
- Fine-tune additional domains (physics, CS, finance, DevOps)

---

## SECTION 8 — POST-FINETUNE VISUAL EVIDENCE

### 1. Structured DevOps Reasoning

The model demonstrates concise, principle-driven explanation patterns learned from the dataset, avoiding the verbosity typical of base models.

![Structured DevOps Reasoning](Diagrams/output_devops_reasoning.jpeg)

### 2. Deductive Reasoning Behavior (Physics)

The model applies first-principles decomposition without explicit chain-of-thought prompting, showing that the _structure_ of reasoning has been internalized.

![Deductive Reasoning Behavior](Diagrams/output_deductive_physics.jpeg)

### 3. Multi-Step Compression (Economics)

The model demonstrates the "compressed logic" architecture taught during training, breaking complex social science topics into atomic constraints and goals.

![Multi-Step Compression](Diagrams/output_economics_compression.jpeg)

---

## SECTION 9 — RAG KNOWLEDGE LAYER (RETRIEVAL-AUGMENTED GENERATION)

Reasoning-Enhanced SLM 2.0 includes a Retrieval-Augmented Generation (RAG) layer to provide **grounded, updatable domain knowledge** alongside the fine-tuned reasoning behavior.

Fine-tuning shapes _how_ the model reasons.  
RAG supplies _what_ the model knows.

This separation allows the model to behave like a “well-read professor” without embedding factual knowledge into weights.

![RAG](Diagrams/rag.jpeg)

---

### 9.1 High-Level Architecture

The RAG system operates in two phases.

#### Ingestion (offline / one-time)

1. Download research papers (arXiv)
2. Store documents locally
3. Convert PDFs to text
4. Chunk text into semantically coherent passages
5. Generate embeddings using **OpenAI (`text-embedding-3-small`)**
6. Store vectors and metadata in **Qdrant**

#### Query-time (online)

1. Embed user question (same embedding model)
2. Perform semantic search over Qdrant
3. Retrieve top-k relevant chunks
4. Assemble prompt with retrieved context
5. Generate answer using **Mistral-7B-Instruct (LoRA-fine-tuned, local runtime)**

---

### 9.2 RAG Components

- **Embeddings**  
  Model: `text-embedding-3-small` (OpenAI)  
  Converts text into 1536-dimensional semantic vectors.

- **Memory Store**  
  Qdrant (local vector database).  
  Persists embeddings and payload metadata (file path, chunk index, text).

- **Retrieval**  
  Dense semantic search using cosine similarity over Qdrant vectors.

- **Generation**  
  Mistral-7B-Instruct with LoRA adapters applies domain-specific reasoning to retrieved context.

---

### 9.3 Initial Knowledge Base

The current RAG knowledge base is bootstrapped with **three arXiv research papers**, stored locally as text files.

rag/data/docs/
├── 2512.11254v2.txt
├── 2512.17898v1.txt
└── 2512.17901v1.txt

Each document is split into overlapping chunks.  
Each chunk is embedded and stored as an independent retrievable memory unit.

---

### 9.4 Why RAG Is Included

RAG addresses limitations of fine-tuning alone:

- **Updatability:** add or remove papers without retraining
- **Grounding:** answers are supported by retrieved text
- **Scalability:** knowledge grows via storage, not parameter updates

The system combines:

- **Behavioral alignment** (LoRA fine-tuning)
- **Externalized knowledge memory** (RAG)

---

### 9.5 Current Status

- Qdrant running locally
- Research papers ingested and embedded
- Query-time retrieval integrated with generation pipeline

Planned extensions:

- Larger document corpus
- Section-aware chunking
- Hybrid retrieval (BM25 + vectors)
- Reranking for improved precision

---

## LICENSE

MIT License
