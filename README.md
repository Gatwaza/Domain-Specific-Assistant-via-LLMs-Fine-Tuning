# Fine-Tuning TinyLlama for Medical Interpreter Question-Answering

## Project Overview

This project fine-tunes **TinyLlama/TinyLlama-1.1B-Chat-v1.0** to build a domain-specific **Medical Interpreter QA assistant**.

Using **LoRA (PEFT)**, 4-bit quantization, and optimized preprocessing, I demonstrate how parameter-efficient fine-tuning can significantly improve domain performance under limited GPU resources.

Four controlled experiments were conducted to evaluate:

* Training steps
* Learning rate
* Tokenization strategy
* Label shifting alignment

The final & best of all experiment achieved:

* **+64.02% ROUGE-2**
* **+223.28% BLEU**

# Dataset

| Dataset      | Source      | Initial Size |
| ------------ | ----------- | ------------ |
| MedQuAD      | Kaggle      | ~16,412      |
| AfrimedQA_v2 | HuggingFace | ~15,275      |

After cleaning and merging:

* **Unified dataset:** 18,122 QA pairs
* **Split:** 80% Train / 10% Validation / 10% Test
* **Training samples:** 14,497
* **Seed:** 42

### Prompt Format

```
### Instruction:
<question>

### Response:
<answer>
```


# Base Model

* **Model:** TinyLlama-1.1B-Chat-v1.0
* 1.1B parameters
* LLaMA-based causal LM
* 4-bit compatible
* Suitable for Colab GPUs


# LoRA Configuration (Shared Across All Experiments)

| Parameter            | Value                          |
| -------------------- | ------------------------------ |
| r                    | 8                              |
| lora_alpha           | 16                             |
| lora_dropout         | 0.05                           |
| target_modules       | ["q_proj", "k_proj", "v_proj"] |
| bias                 | "none"                         |
| task_type            | CAUSAL_LM                      |
| Trainable Parameters | **0.57%**                      |


# Training Setup (Shared)

* 4-bit NF4 quantization
* bfloat16 compute
* Gradient checkpointing
* Logging / saving every 50 steps
* Resume from checkpoints enabled
* Effective batch size: 32


# Experiment Configurations

| Experiment | Steps | LR   | Batch | Key Focus                 |
| ---------- | ----- | ---- | ----- | ------------------------- |
| Exp_001    | 908   | 5e-5 | 32    | Baseline LoRA adaptation  |
| Exp_002    | 400   | 1e-4 | 32    | Reduced steps impact      |
| Exp_003    | 400   | 2e-4 | 32    | Higher learning rate      |
| Exp_004    | 400   | 2e-4 | 32    | Label shifting correction |


# Results Summary

## Exp_001 – Baseline

* ROUGE-2: **+46.79%**
* BLEU: **+114.61%**

Strong initial gains; minor drop in strict F1 token overlap.


## Exp_002 – Reduced Steps

Performance declined across most metrics.

Shows insufficient training steps limit adaptation.


## Exp_003 – Higher LR

* ROUGE-2: +0.21%
* BLEU: +3.36%

Minimal impact; LR alone not decisive.


## Exp_004 – Label Shifting Fix (Breakthrough)

| Metric       | Improvement  |
| ------------ | ------------ |
| ROUGE-1      | +16.15%      |
| ROUGE-2      | **+64.02%**  |
| ROUGE-L      | +20.87%      |
| BLEU         | **+223.28%** |
| BERTScore F1 | +0.75%       |

Correct label alignment significantly improved fluency and n-gram overlap.


# Evaluation Metrics

* ROUGE-1 / ROUGE-2 / ROUGE-L
* BLEU
* F1 Score
* BERTScore
* Flesch-Kincaid
* SMOG Index
* Validation Loss & Perplexity


# Deployment

All LoRA adapters are available in the Hugging Face Project Space:Jeanrobert/tinyllama-medqa-gradio-demo-exp002.

Adapters can be loaded into the base TinyLlama model for inference without retraining.


# Google Colab Setup Guide

Follow this order for smooth execution:

### 1 Configure Runtime (Manual)

* Runtime → Change runtime type
* Enable **GPU**
* Use **High-RAM** if available


### 2 Run `library_name` (First Code Cell)

Installs:

* accelerate
* bitsandbytes
* datetime
* datasets
* evaluate
* glob
* google.colab
* gradio
* huggingface_hub
* json
* kagglehub
* matplotlib
* nltk
* numpy
* os
* pandas
* peft
* pickle
* re
* rouge_score
* seaborn
* string
* subprocess
* sys
* textstat
* textwrap
* time
* torch
* transformers

Restart runtime after installation.


### 3 Configure Hugging Face Token

Add `HF_TOKEN` via Colab Secrets (Keys icon in left panel).


### 4 Run `hf_login`

Required before:

* Pushing models
* Accessing gated datasets
* Deploying Gradio spaces


### 5 Mount Google Drive

Run `mount_drive_code` before:

* Saving checkpoints
* Loading datasets
* Storing results

Colab sessions are temporary — Drive ensures persistence.


### Resume Logic

Training resumes automatically:

```python
trainer.train(resume_from_checkpoint=latest_checkpoint)
```

Prevents loss of progress during long runs.


# Key Takeaways

1. LoRA enables fine-tuning with only **0.57% trainable parameters**.
2. 4-bit quantization allows training on limited hardware.
3. Training steps and LR alone were not decisive.
4. **Label shifting correction was the critical factor.**
5. Exp_004 achieved the strongest performance gains.

# Conclusion
This project shows that LoRA-based fine-tuning of TinyLlama can significantly improve medical QA performance with minimal trainable parameters and limited hardware. The largest gains came from correcting label shifting, proving that proper target alignment is more impactful than simply increasing training steps or learning rate.
