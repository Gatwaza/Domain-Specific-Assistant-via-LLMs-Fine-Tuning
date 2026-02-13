# Fine-tuning TinyLlama for Healthcare Question-Answering

## Project Goal
The aim of this project is to fine-tune **TinyLlama/TinyLlama-1.1B-Chat-v1.0** to perform **Healthcare Question-Answering (QA)** tasks. Using datasets like MedQuAD and AfrimedQA_v2, we leverage **LoRA + PEFT** techniques for **parameter-efficient domain adaptation**. By experimenting with different hyperparameters, tokenization strategies, optimizers, and memory optimization techniques, we aim to develop a **high-quality, reproducible healthcare QA model** suitable for deployment on limited-resource hardware.


## Dataset Sources and Curation

| Dataset | Source | Initial Size | Cleaned Size | QA Format | Role in Project |
|---------|--------|--------------|--------------|-----------|----------------|
| MedQuAD | Kaggle (`pythonafroz/medquad-medical-question-answer-for-ai-research`) | ~14,000 | ~13,800 | `### Instruction:\n{question}\n\n### Response:\n{answer}` | Provides a large, diverse set of medical questions and answers to teach the model general medical QA. |
| AfrimedQA_v2 | HuggingFace (`intronhealth/afrimedqa_v2`) | ~1,200 | ~1,100 | `### Instruction:\n{question}\n\n### Response:\n{answer_rationale}` | Supplements MedQuAD with African-focused medical questions to improve cultural and regional relevance. |

**Processing Steps:**
1. Remove empty questions or answers to ensure high-quality training data.  
2. Merge datasets for broader coverage.  
3. **Train/Validation/Test split:** 80:10:10, random seed = 42.  
   - **Purpose:** Ensures reproducibility and proper evaluation of generalization.


## Data Preprocessing and Tokenization

| Experiment | Tokenization Approach | Padding | Max Length | Role in Project |
|------------|--------------------|--------|------------|----------------|
| Exp_001 | Dynamic padding (`padding=True`) | `tokenizer.pad_token = tokenizer.eos_token` | 512 | Reduces memory usage by padding sequences only to batch maximum; allows longer sequences without wasting memory. |
| Exp_002 | Max-length padding (`padding="max_length"`) | `tokenizer.pad_token = tokenizer.eos_token` | 512 | Fixes sequence length for batch uniformity; may simplify some optimization but increases memory usage. |
| Exp_003 | Max-length padding with explicit `[PAD]` token | Added `[PAD]` token if absent | 512 | Ensures consistent padding behavior across GPUs and experiments; critical for reproducibility. |

**Prompt formatting for all experiments:**
```text
### Instruction:
<question>

### Response:
<answer>
````

* **Purpose:** Standardized instruction-response format improves model understanding and downstream QA generation consistency.


## Base Model Selection

* Model: **TinyLlama/TinyLlama-1.1B-Chat-v1.0**
* **Why this model:**

  * **Size:** ~1.1B parameters → light enough to fine-tune on GPUs like Colab.
  * **Architecture:** LLaMA-based causal LM → well-suited for instruction-based QA.
  * **Pre-training:** General text corpus → provides a solid starting point before domain adaptation.
  * **Efficiency:** Compatible with 4-bit quantization and bf16 → reduces memory footprint while maintaining performance.

**Role:** Serves as the frozen backbone for LoRA adaptation; allows quick experimentation without full model retraining.


## Fine-tuning Strategy: LoRA & PEFT

| Parameter      | Value                                                                   | Purpose                                                                                                  |
| -------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| r              | 8                                                                       | Low-rank adaptation dimension; controls capacity of LoRA layers.                                         |
| lora_alpha     | 16                                                                      | Scales LoRA weight updates to match base model magnitude.                                                |
| target_modules | ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] | Specifies which layers to adapt; focuses training on attention and feed-forward modules relevant for QA. |
| lora_dropout   | 0.05                                                                    | Regularizes LoRA updates to reduce overfitting.                                                          |
| bias           | "none"                                                                  | No additional bias parameters; keeps adaptation lightweight.                                             |
| task_type      | TaskType.CAUSAL_LM                                                      | Indicates causal LM fine-tuning (next-token prediction) suitable for generative QA.                      |

**Role in Project:** Enables **parameter-efficient fine-tuning**, training only ~1–5% of total model parameters, drastically reducing memory and compute requirements.


## Memory-Efficient Techniques

| Experiment | Quantization                 | Compute Dtype | Gradient Checkpointing | Purpose                                                                                                              |
| ---------- | ---------------------------- | ------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Exp_001    | 4-bit (`BitsAndBytesConfig`) | bfloat16      | False                  | Reduces memory usage; allows fitting larger batches. Falls back to full precision if 4-bit fails.                    |
| Exp_002    | Attempted 4-bit              | bfloat16      | False                  | Tests feasibility of lightweight fine-tuning with small GPU memory; handles fallback cases gracefully.               |
| Exp_003    | 4-bit                        | bfloat16      | True                   | Saves memory during training via gradient checkpointing; allows larger effective batch size and prevents OOM errors. |

**Role:** Ensures experiments can run on limited GPU memory without sacrificing model quality.


## Optimizer and Scheduler Differences

| Experiment | Optimizer        | Scheduler | Warmup Steps | Role                                                                                                   |
| ---------- | ---------------- | --------- | ------------ | ------------------------------------------------------------------------------------------------------ |
| Exp_001    | adamw_torch      | None      | 0            | Standard optimizer; stable for step-based training.                                                    |
| Exp_002    | paged_adamw_8bit | cosine    | 20           | Tests optimizer with lower memory usage and cosine LR scheduling; warms up LR to stabilize training.   |
| Exp_003    | adamw_torch      | None      | 0            | Standard optimizer; combined with gradient checkpointing to maintain stability for resumable training. |


## Experiment Configurations

| Experiment | Max Steps | Learning Rate | Batch Size (per device) | Grad Accum | Effective Batch | bf16           | fp16  | Checkpoint/Resume                  | Purpose/Notes                                                                                     |
| ---------- | --------- | ------------- | ----------------------- | ---------- | --------------- | -------------- | ----- | ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| Exp_001    | 908       | 5e-5          | 4                       | 8          | 32              | True           | False | Auto-resume from latest checkpoint | Step-based (~2 epochs); validates baseline LoRA adaptation and dynamic padding.                   |
| Exp_002    | 400       | 1e-4          | 4                       | 8          | 32              | True (if CUDA) | False | Auto-resume with try/catch         | Explores optimizer LR effect, fixed padding, and scheduler behavior.                              |
| Exp_003    | 400       | 2e-4          | 8                       | 4          | 32              | True           | False | Auto-resume enabled                | Full resumable training, gradient checkpointing; validates reproducibility and memory efficiency. |


## Trainer Setup

All experiments use **HuggingFace `Trainer`**:

* Model: LoRA-adapted TinyLlama.
* TrainingArguments: includes batch size, gradient accumulation, LR, bf16, save/eval/log steps.
* Datasets: `train_dataset`, `eval_dataset`.
* Data collator: `DataCollatorForLanguageModeling(tokenizer, mlm=False)`.

**Role:** Handles all training loops, evaluation, checkpointing, and logging automatically, making experiments reproducible.


## Training Execution

* **Resume logic:**

```python
checkpoints = [dir for dir in os.listdir(OUTPUT_DIR) if dir.startswith("checkpoint")]
latest_checkpoint = sorted(checkpoints)[-1] if checkpoints else None
trainer.train(resume_from_checkpoint=latest_checkpoint)
```

* Ensures long-running experiments can **resume after interruptions**, critical for Colab or limited-resource GPUs.

* **Metrics tracked:**

  * Wall-clock training time
  * Peak GPU memory usage (via `torch.cuda.max_memory_allocated()`)


## Evaluation

* Validation loss is computed using `trainer.evaluate()` → `eval_loss`.
* Baseline model metrics for comparison:

| Metric          | TinyLlama-1.1B-Chat-v1.0 (Pre-fine-tune) |
| --------------- | ---------------------------------------- |
| Validation Loss | ~2.15                                    |
| Perplexity      | ~8.6                                     |

**Role:** Provides a **quantitative measure of model improvement** after each fine-tuning experiment.


## Saving LoRA Adapters

* LoRA adapters are saved to:

```text
<OUTPUT_DIR>/lora_adapters
```

* Allows **loading adapters into base model for inference** without retraining the full model.
* Ensures **persistent storage** via Google Drive for reproducibility and deployment.


## Contribution of Each Experiment

| Experiment | Contribution                                                                                                                                                |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Exp_001    | Baseline step-based fine-tuning with dynamic tokenization; establishes foundational LoRA adaptation for healthcare QA.                                      |
| Exp_002    | Tests higher learning rate, alternative optimizer (paged_adamw_8bit), fixed-length tokenization, and LR scheduler; informs impact of hyperparameter tuning. |
| Exp_003    | Full resumable training with gradient checkpointing; validates memory-efficient training, reproducibility, and checkpointing logic.                         |


**Conclusion:**
These experiments collectively explore **how tokenization, batch size, optimizer, scheduler, memory optimization, and LoRA configuration affect model performance**, building a **robust and reproducible healthcare QA TinyLlama model**.