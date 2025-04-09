
# Crosslingual NER Project

This project modularizes a Jupyter Notebook for Danish Named Entity Recognition (NER) using spaCy, with support for crosslingual transfer learning via pretrained embeddings.

---

## 📁 Project Structure

```
crosslingual_ner_project_with_transfer/
│
├── data_loader.py       # Functions to load and convert CoNLL format data
├── evaluation.py        # Evaluation metrics: precision, recall, F1
├── train.py             # Basic spaCy model training logic
├── transfer.py          # Transfer learning using pretrained vectors
├── main.py              # Pipeline script: loading → training → evaluating
```

---

## ⚙️ Setup

1. Place your data files inside a folder named `data/`:
    ```
    data/
    ├── danish-train.conll
    ├── danish-dev.conll
    ├── danish-test.conll
    └── vocab/               # Pretrained bilingual embeddings for spaCy
    ```

2. Install dependencies:
    ```bash
    pip install spacy torch numpy
    ```

---

## 🚀 Usage

Run the main script:

```bash
python main.py
```

---

## 🔍 Functionality Overview

### `data_loader.py`

- `read_data(file)` – Reads CoNLL formatted token/tag data.
- `get_spacy_ner_data(data)` – Converts to spaCy training format.

### `evaluation.py`

- `evaluate(system_data, gold_data)` – Calculates evaluation metrics.

### `train.py`

- `train_spacy_model(data, dev_data, iterations)` – Trains a basic Danish model from scratch.

### `transfer.py`

- `init_model(data, lang)` – Initializes a blank model with pretrained embeddings.
- `retrain(train_data, dev_data, epochs, model)` – Fine-tunes a pretrained model.
- `annotate(data, model)` – Uses the model to generate predictions.

---

## 🧠 Training Modes

### 1. **Baseline Training**

Trains a fresh model using Danish training data.

### 2. **Transfer Learning**

Uses pretrained bilingual embeddings and fine-tunes on the same Danish data for improved performance.

---

## 📈 Output

Console logs during execution display:

- Training losses per epoch
- Evaluation results on test set:
  - Precision
  - Recall
  - F1 Score

---

## ✍️ Author

Desmond Bai
