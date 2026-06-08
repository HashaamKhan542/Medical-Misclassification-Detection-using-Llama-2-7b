# 🔬 Medical Misinformation Detection — LLaMA 2 7B Fine-Tuning

> **Improving medical claim classification accuracy from 45% → 77% through QLoRA fine-tuning of LLaMA 2 7B on the HealthFact dataset**

---

## 📊 Results at a Glance

| Method | Accuracy | Notes |
|---|---|---|
| Base LLaMA 2 7B (no prompt) | ~45% | Untuned baseline on 100-sample holdout |
| One-shot prompting | ~50% | Minimal improvement from single example |
| Few-shot prompting | ~58% | Better, but inconsistent label parsing |
| **QLoRA fine-tuned (final)** | **77%** | **32pp improvement over baseline** |

---

## 🧠 Project Overview

This project was completed as part of the **DataBytes x Deakin University AI Capstone** (March – October 2025). I led a team of 5 using Agile methodology across an 8-month lifecycle.

The goal was to fine-tune an open-source large language model to classify medical health claims as **true**, **false**, or **misleading** — addressing the growing problem of medical misinformation online.

---

## 🗂️ Dataset

**HealthFact** — a benchmark dataset of medical claims with fact-check labels.

| Split | Size |
|---|---|
| Training | ~10,000 records |
| Validation (filtered) | ~1,500 records |
| Benchmark holdout | 100 samples (held out from training) |

Each record contains:
- `claim` — the medical health statement
- `explanation` — context or source information
- `label` — `true`, `false`, or `MISLEADING`

---

## 🏗️ Architecture & Pipeline

```
HealthFact Dataset
       │
       ▼
  Data Cleaning & Formatting
  (prompt template: Claim + Explanation → Answer)
       │
       ▼
  Benchmark Holdout (100 samples separated)
       │
       ▼
  ┌────────────────────────────────────┐
  │         LLaMA 2 7B Chat            │
  │   (meta-llama/Llama-2-7b-chat-hf) │
  └────────────────────────────────────┘
       │
  Baseline benchmark: 45% accuracy
  (confusion matrix evaluation)
       │
       ▼
  Prompting experiments
  (one-shot → few-shot)
       │
       ▼
  ┌────────────────────────────────────┐
  │     QLoRA Fine-Tuning              │
  │   4-bit quantization (NF4)         │
  │   via BitsAndBytes + PEFT          │
  └────────────────────────────────────┘
       │
  Post-tuning benchmark: 77% accuracy
       │
       ▼
  Confusion Matrix Evaluation
  (sklearn, 3-class: true / false / MISLEADING)
```

---

## ⚙️ Fine-Tuning Configuration

```python
# Quantization — 4-bit NF4 via BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)
```

---

## 📈 Evaluation

Evaluation was performed on a 100-sample holdout benchmark, separated before training to prevent data leakage.

**Metrics used:**
- Overall accuracy
- Confusion matrix (3-class: `true`, `false`, `MISLEADING`)
- Per-class precision and recall

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred, labels=["true", "false", "MISLEADING"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["true", "false", "MISLEADING"])
disp.plot(cmap="Blues", values_format="d")
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Base model | LLaMA 2 7B Chat (meta-llama/Llama-2-7b-chat-hf) |
| Fine-tuning | PEFT, QLoRA, BitsAndBytes (4-bit NF4) |
| Training framework | Hugging Face Transformers, Trainer API |
| Data processing | Pandas, Hugging Face Datasets |
| Evaluation | Scikit-learn (confusion matrix, accuracy) |
| Compute | Google Colab (GPU), Google Drive |
| Version control | Git, GitHub |
| Project management | Agile / Scrum (weekly standups, sprint planning) |

---

## 🗃️ Repository Structure

```
medical-misinformation-detection/
├── notebooks/
│   └── Llama2_Finetuning.ipynb       # Full pipeline: load → benchmark → fine-tune → evaluate
├── data/
│   └── README.md                      # HealthFact dataset instructions (not included — see below)
├── results/
│   └── confusion_matrix.png           # Post-tuning confusion matrix
└── README.md
```

> **Note on data:** The HealthFact dataset is publicly available. Download and place JSON files in `/data/` before running the notebook.

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/hashaamkhan/medical-misinformation-detection.git
cd medical-misinformation-detection
```

**2. Install dependencies**
```bash
pip install transformers datasets accelerate peft trl bitsandbytes scikit-learn pandas
```

**3. Add Hugging Face token**
```python
from huggingface_hub import login
login()  # You need access to meta-llama/Llama-2-7b-chat-hf
```

**4. Run the notebook**

Open `notebooks/Llama2_Finetuning.ipynb` in Google Colab (GPU runtime recommended — T4 or A100).

---

## 👥 Team

Led by **Hashaam Khan** (project lead) — team of 5, DataBytes x Deakin University AI Capstone.

Methodology: Agile Scrum — weekly standups, sprint retrospectives, fortnightly stakeholder presentations.

---

## 📬 Contact

**Hashaam Khan**
- 📧 hashaamkhan542@gmail.com


> *Master of Data Science (Distinction) · Deakin University · Melbourne, AU*
> *Open to data science and AI/ML roles · 485 Graduate Visa · Full work rights*
