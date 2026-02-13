# Domain-Specific Assistant via LLM Fine-Tuning  
### Fine-tuning TinyLlama for Healthcare Question Answering

## Project Overview

This project explores **parameter-efficient fine-tuning (PEFT)** of a compact Large Language Model, **TinyLlama-1.1B-Chat**, for **English-language healthcare question-answering (QA)**.  
The objective is to develop a **resource-efficient, domain-specialized medical assistant** while systematically studying how different training strategies affect performance.

Three fine-tuning experiments (**Exp_001, Exp_002, Exp_003**) are conducted **in parallel**, sharing the same data pipeline and LoRA configuration but varying **training schedules and hyperparameters**.

---

## 1. Dataset Identification & Curation

### Datasets Used (English-only)
- **MedQuAD**  
  Source: KaggleHub (`pythonafroz/medquad-medical-question-answer-for-ai-research`)  
  Initial size: 16,412 QA pairs

- **AfrimedQA v2**  
  Source: Hugging Face (`intronhealth/afrimedqa_v2`)  
  Initial size: 15,275 QA pairs

### Unified Dataset Construction
Both datasets were converted to a shared schema:
- `question` → `instruction`
- `answer` / `answer_rationale` → `response`

### Cleaning & Filtering
- Removed empty or whitespace-only instructions and responses
- Final unified dataset size: **18,122 high-quality samples**

### Dataset Split (seed = 42)
| Split | Samples |
|------|--------|
| Train | 14,497 |
| Validation | 1,812 |
| Test | 1,813 |

---

## 2. Prompt Formatting & Tokenization

### Instruction-Tuning Format
