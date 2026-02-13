# Healthcare Interpreter Chatbot for Refugee Settings

## Overview

Language barriers remain a critical challenge in hospitals and healthcare centers serving refugees and displaced populations. This project aims to design and implement an **AI-powered multilingual healthcare interpreter chatbot** that facilitates communication between patients and healthcare professionals using **pretrained Transformer models**.

The system leverages **Hugging Face multilingual models**, implemented and fine-tuned using **TensorFlow**, to enable interpretation and conversational support for **Sub-Saharan African languages** in clinical contexts. The chatbot is designed as an assistive tool to improve access, safety, and quality of healthcare delivery in humanitarian and low-resource environments.

This repository is under active development.

---

## Background & Motivation

With over **30 million forcibly displaced people in Africa** and more than **2,000 languages spoken across the continent**, language barriers significantly hinder access to timely and effective healthcare. Refugees and displaced populations frequently encounter healthcare providers who do not speak their native languages, leading to miscommunication, misdiagnosis, delayed treatment, and reduced quality of care.

In many humanitarian and low-resource healthcare settings, professional medical interpreters are unavailable or insufficient. This project explores the use of **multilingual interpretation and conversational models powered chatbot systems** as assistive interpretation tools to bridge communication gaps between patients and healthcare professionals while acknowledging the ethical and clinical limitations of automated systems.

---

## Project Objectives

* Break language barriers in hospital and healthcare environments for refugees
* Support interpretation between patients and healthcare professionals
* Enable multilingual conversational interaction in clinical contexts
* Focus on low-resource Sub-Saharan African languages
* Provide an intuitive user interface suitable for healthcare settings
* Evaluate system performance using standard NLP metrics and qualitative testing

---

## Key Features

* **Multilingual Transformer Models**

  * Pretrained models sourced from Hugging Face
  * Fine-tuned using TensorFlow
* **Healthcare-Oriented Chatbot**

  * Context-aware medical conversations
  * Interpretation between patient and clinician languages
* **User Interface (UI)**

  * Chat-based interaction
  * Designed for usability in clinical environments
* **Robust Evaluation Framework**

  * Automatic and human-centered evaluation
* **Scalable & Modular Architecture**

  * Supports future language, domain, and modality expansion

---

## Supported Languages (Initial Scope)

* Selected Sub-Saharan African languages (e.g., Kinyarwanda, Swahili, Amharic – subject to dataset availability)
* English and/or French as pivot languages

> Language coverage will depend on availability

---

## Technology Stack

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow
* **NLP Models:** Hugging Face Transformers
* **Evaluation Metrics:** BLEU, F1-score, Perplexity, Qualitative Human Evaluation
* **Frontend/UI:** Web-based chatbot interface

---

## Model Selection & Architecture

### Multilingual Model Strategy

This project adopts a **two-stage multilingual architecture** to balance translation quality and conversational reasoning:

1. **Translation / Interpretation Layer**

   * Models such as **NLLB-200**, **mBART**, or equivalent multilingual translation models
   * Optimized for low-resource and African languages

2. **Conversation & Reasoning Layer**

   * Instruction-tuned multilingual models such as **Flan-T5**, **BLOOMZ**, or similar
   * Handles dialogue flow, intent understanding, and response generation

### High-Level Pipeline

```
Patient Language → Translation Model → Pivot Language (EN/FR)
Pivot Language → Conversational Model → Response
Response → Translation Model → Patient Language
```

This modular design allows independent evaluation of translation quality and conversational performance.

---

## Domain-Specific Assistant & Fine-Tuning Strategy

### Objective

In addition to multilingual interpretation, this project explicitly focuses on building a **domain-specific healthcare assistant** by fine-tuning a **Large Language Model (LLM)**. The fine-tuned model is expected to understand user queries and generate **relevant, accurate, and context-aware responses** within the healthcare domain, while handling out-of-domain queries safely and appropriately.

The assistant is approached as a **generative question–answering (QA) system**, capable of producing free-text responses rather than selecting from predefined answers.

---

### Model Selection for Efficient Fine-Tuning

A modern generative LLM will be selected from **Hugging Face** with careful consideration of:

* Compatibility with **Google Colab free GPU resources**
* Parameter efficiency
* Strong baseline language understanding

Recommended candidate models include:

* **Gemma** (lightweight, modern, instruction-friendly)
* **TinyLLaMA** (resource-efficient and suitable for experimentation)
* Other small-to-medium-scale open LLMs that balance capability with practical training constraints

The implementation will primarily use **TensorFlow**, while leveraging the Hugging Face ecosystem and modern fine-tuning utilities.

---

### Parameter-Efficient Fine-Tuning (PEFT)

To enable fine-tuning on limited hardware, this project employs **parameter-efficient fine-tuning techniques**, specifically **LoRA (Low-Rank Adaptation)** via the `peft` library. This approach allows effective customization of large models without updating all parameters, making it a foundational technique for modern LLM adaptation.

---

### Dataset Collection & Preparation

Fine-tuning requires a curated dataset of **instruction–response (question–answer) pairs** aligned with the **multilingual healthcare interpretation domain**.

#### Potential Dataset Sources

* **Hugging Face Datasets Hub** ([https://huggingface.co/datasets](https://huggingface.co/datasets))
* **Kaggle Datasets** ([https://kaggle.com/datasets](https://kaggle.com/datasets))
* **UCI Machine Learning Repository** ([https://archive.ics.uci.edu](https://archive.ics.uci.edu))
* **Google Dataset Search** ([https://datasetsearch.research.google.com](https://datasetsearch.research.google.com))
* Academic datasets such as:

  * **MedQA** (healthcare)
  * **Medical Meadow** datasets (e.g., `medalpaca/medical_meadow_medical_flashcards`)

Datasets may be adapted or converted into QA format using domain documents and multilingual interpretation scenarios.

#### Dataset Requirements

* Format: Instruction–response (QA) pairs
* Coverage: Diverse healthcare intents and scenarios
* Size: **1,000–5,000 high-quality examples** to balance efficiency and performance

#### Preprocessing Steps

* Tokenization using Hugging Face tokenizers
* Text normalization and cleaning
* Formatting into consistent instruction–response templates
* Ensuring sequence lengths fit within the model context window

---

### Fine-Tuning Procedure

The selected pre-trained LLM will be fine-tuned using the prepared dataset with the following considerations:

* **LoRA-based PEFT** for efficiency
* Training conducted on Google Colab GPUs
* Hyperparameter tuning, including:

  * Learning rate: typically **1e-4 to 5e-5**
  * Batch size: **2–4** with gradient accumulation
  * Optimizer selection
  * Training epochs: **1–3**

All experiments will be documented, including:

* Hyperparameter configurations
* GPU memory usage
* Training time
* Observed performance changes

An **experiment tracking table** will be maintained to record and compare results across runs.

---

### Evaluation & Comparison

Model evaluation will include both **quantitative** and **qualitative** methods:

* **BLEU Score** – generation quality
* **ROUGE Score** – overlap with reference answers
* **Perplexity** – language model fluency

Comparisons will be made between:

* The **base pre-trained model**
* The **fine-tuned domain-specific model**

Qualitative testing will involve interactive probing to assess:

* Clinical relevance
* Faithfulness to the healthcare domain
* Robustness to out-of-domain queries

---

### Deployment & User Interaction

The fine-tuned model will be deployed with an interactive interface to enable user access. Deployment options include:

* **Gradio** (recommended for simplicity and Colab compatibility)
* Flask or Streamlit (alternative options)

The interface will allow users to:

* Enter healthcare-related queries
* Receive interpreted and domain-specific responses from the customized LLM

---

## Training & Implementation

* Models are implemented and fine-tuned using **TensorFlow**
* Hugging Face tokenizers and model checkpoints are used
* Fine-tuning focuses on:

  * Multilingual understanding
  * Healthcare-related conversational patterns
  * Robustness in low-resource language scenarios

---

## Evaluation Methodology

The system is evaluated using a combination of **quantitative metrics** and **qualitative assessment**:

### Automatic Evaluation Metrics

* **BLEU Score**

  * Measures translation quality against reference translations
* **F1-Score**

  * Evaluates token-level accuracy or intent classification performance
* **Perplexity**

  * Measures language model fluency and coherence

### Qualitative Evaluation

* Human evaluation by bilingual or multilingual speakers
* Assessment of:

  * Clinical relevance
  * Clarity of interpretation
  * Context preservation
  * Safety and appropriateness of responses

---

## User Interface

* Chat-based UI for real-time interaction
* Supports:

  * Patient-facing input
  * Interpreted responses for healthcare professionals
* Designed with emphasis on:

  * Simplicity
  * Clinical usability
  * Deployment in low-resource environments

UI functionality and usability testing will be expanded incrementally.

---

## Project Structure (Planned)

```
.
├── data/               # Datasets and preprocessing scripts
├── models/             # Pretrained and fine-tuned models
├── training/           # Training and fine-tuning pipelines
├── evaluation/         # Evaluation scripts and metrics
├── ui/                 # Chatbot user interface
├── utils/              # Utility functions
├── notebooks/          # Experiments and prototyping
└── README.md
```

---

## Ethical Considerations & Limitations

* This system is intended for **academic activity**.
* Outputs must not be treated as medical advice.
* Patient privacy and data protection must be strictly respected.
* Performance limitations and bias in low-resource languages are acknowledged.

---

## Future Work

* Expand supported African languages
* Integrate speech-to-text and text-to-speech modules
* Lightweight and offline-capable model deployment
* Clinical pilot studies and real-world validation

---

## Project Status

**Active Development**

This repository and README will be updated continuously as the project evolves.

---

## Disclaimer

This project is for academic purposes only. It is not a certified medical device and should not be used as a standalone solution for medical decision-making.
