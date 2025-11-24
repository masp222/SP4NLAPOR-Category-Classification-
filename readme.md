# 📘 Automatic Classification of Public Complaints on SP4N-LAPOR!

### Using Vector Embeddings + Neural Network Classifiers

---

## 📌 Overview

This repository contains the implementation of an **automated text classification system** for public complaint reports from **SP4N-LAPOR!**, Indonesia’s national public service reporting platform.

The goal of this project is to **automatically predict the correct complaint category** using text embeddings and neural network classifiers. Automating this process helps reduce manual workload, minimize misrouting errors, and accelerate government response time.


---

## 🚀 Features

* End-to-end **text preprocessing**, including merging title + description.
* Multiple **embedding models** tested:

  * Multilingual-E5
  * Qwen3
  * FastText
  * IndoBERT
  * CendolT5
* Neural network classifier using **LSTM**.
* Class imbalance handling with **class weighting** instead of oversampling.
* Evaluation using **Accuracy**, **F1-Macro**, and **AUC**.
* Reproducible experiment pipeline.

---

## 📂 Dataset

The dataset is sourced from **SP4N-LAPOR! (lapor.go.id)** and contains:

| Field         | Description                  |
| ------------- | ---------------------------- |
| `title`       | Title of complaint           |
| `description` | Full complaint text          |
| `category`    | Government-assigned category |

**Total samples:** 5,000 reports
**Total categories:** 19 (e.g., Pekerjaan Umum, Perhubungan, Kesehatan, Sosial)


The dataset has **significant class imbalance**, so the model uses **class weighting** during training.


---

## 🧹 Preprocessing

1. Remove HTML, special characters, and noise
2. Lowercasing + normalization
3. Merge the `title` and `description` into a single input text
4. Tokenization handled by each embedding model


---

## 🔤 Embedding Models

We evaluated five embedding techniques:

| Embedding Model     | Type                  | Notes                  |
| ------------------- | --------------------- | ---------------------- |
| **Multilingual-E5** | Transformer           | Highest accuracy & AUC |
| **Qwen3**           | LLM-based             | Highest F1-Macro       |
| **FastText**        | Static embedding      | Mid performance        |
| **IndoBERT**        | Pretrained Indonesian | Lower than expected    |
| **CendolT5**        | Indonesian T5         | Lowest performance     |

Results from the paper show embedding quality significantly impacts classification performance.


---

## 🧠 Model Architecture

The classifier uses an **LSTM-based model**:

* LSTM layer (128 units)
* Dense layer (64 units, ReLU)
* Softmax output layer (19 categories)


**Training Setup:**

* 50 epochs
* Batch size: 32
* Optimizer: Adam
* Loss: Categorical Cross-Entropy (with class weights)


---

## 📊 Results

| Embedding           | Accuracy   | F1-Macro   | AUC        |
| ------------------- | ---------- | ---------- | ---------- |
| **Multilingual-E5** | **0.7177** | 0.5048     | **0.9367** |
| **Qwen3**           | 0.6899     | **0.5249** | 0.9317     |
| **FastText**        | 0.6530     | 0.4804     | 0.9320     |
| **IndoBERT**        | 0.6016     | 0.4440     | 0.9006     |
| **CendolT5**        | 0.5739     | 0.3860     | 0.8916     |
|                     |            |            |            |

**Key findings:**

* **Multilingual-E5** gives the best **Accuracy** and **AUC**
* **Qwen3** gives the best **F1-Macro**, indicating better balance across all categories
* Indonesian-specific models did *not* outperform multilingual models


---

## 🙌 Acknowledgements

This project is based on the paper **"Klasifikasi Otomatis Laporan Masyarakat pada Platform LAPOR!"** submitted to GEMASTIK 2025.

