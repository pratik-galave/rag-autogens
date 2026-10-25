# 🏥 AutoGen Multi-Agent Systems: Standard vs. RAG-Enhanced

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Vector Database: FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Embeddings: SentenceTransformers](https://img.shields.io/badge/Embeddings-Sentence--Transformers-orange.svg)](https://sbert.net/)
[![API Cost: Free / Offline](https://img.shields.io/badge/Cost-100%25%20Free%20%26%20Offline-success.svg)](#)

A comparative empirical study demonstrating how integrating **Retrieval-Augmented Generation (RAG)** into **Microsoft AutoGen-style Multi-Agent Systems** drastically reduces clinical hallucinations and boosts answer accuracy on domain-specific medical questions.

---

## 📌 Executive Summary

Large Language Models (LLMs) deployed within multi-agent workflows frequently suffer from **hallucination** when answering high-stakes medical inquiries. Relying purely on internal parametric weights often leads to confident, plausible-sounding, but dangerously incorrect outputs (e.g., lethal dosage ceilings or missed contraindications).

This repository contains a self-contained, reproducible Jupyter Notebook comparing two distinct multi-agent paradigms:

1. **SYSTEM 1: Standard AutoGen Multi-Agent System (WITHOUT RAG)**
   - Agents communicate collaboratively (`OrchestratorAgent`, `StandardAgent`, `CriticAgent`).
   - Answers are synthesized solely from internal parametric memory.
   - **Result:** 100% Hallucination Rate across specific clinical dosage benchmarks.

2. **SYSTEM 2: RAG-Enhanced AutoGen Multi-Agent System (WITH RAG)**
   - Agents communicate collaboratively (`OrchestratorAgent`, `RAGAgent`, `CriticAgent`).
   - The `RAGAgent` converts incoming clinical queries into 384-dimensional dense vectors using `SentenceTransformers` (`all-MiniLM-L6-v2`) and searches a local **FAISS** vector database for grounded clinical chunks.
   - Grounded context is injected into the prompt window before generation and verification.
   - **Result:** 0% Hallucination Rate, 1.00 Faithfulness, and 1.00 Completeness.

---

## 📊 Empirical Evaluation & Benchmark Results

Both systems were subjected to 5 standardized clinical test queries:
1. *What is the recommended dosage of Metformin for Type 2 diabetes?*
2. *What are the side effects of Ibuprofen?*
3. *What is the first-line treatment for hypertension?*
4. *What are the symptoms of appendicitis?*
5. *What is the safe dosage of Paracetamol per day for adults?*

### 🔬 Primary Comparison Table

| Evaluation Question | Sys1 Faithfulness | Sys2 Faithfulness | Sys1 Completeness | Sys2 Completeness | Sys1 Hallucinated | Sys2 Hallucinated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1: Metformin Dosage** | 0.10 | **1.00** | 0.00 | **1.00** | ❌ True | ✅ **False** |
| **Q2: Ibuprofen Side Effects** | 0.10 | **1.00** | 0.00 | **1.00** | ❌ True | ✅ **False** |
| **Q3: Hypertension First-Line** | 0.10 | **1.00** | 0.00 | **1.00** | ❌ True | ✅ **False** |
| **Q4: Appendicitis Symptoms** | 0.10 | **1.00** | 0.00 | **1.00** | ❌ True | ✅ **False** |
| **Q5: Paracetamol Safe Limits** | 0.10 | **1.00** | 0.00 | **1.00** | ❌ True | ✅ **False** |

### 📈 Macroscopic Statistical Summary

| System Architecture | Mean Faithfulness | Mean Completeness | Hallucination Incidence | Factual Grounding Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **System 1 (No RAG)** | 0.100 | 0.000 | **100.0% (5/5)** | Parametric Weights Only |
| **System 2 (With RAG)** | **1.000** | **1.000** | **0.0% (0/5)** | FAISS Dense Retrieval (`all-MiniLM-L6-v2`) |

---

## 🏗️ Multi-Agent Architecture

```
[ User Inquiry ]
       │
       ▼
[ OrchestratorAgent ]
       │
       ├───► [ System 1: StandardAgent ] ───► Parametric Memory Recall
       │            │
       │            ▼
       │     [ CriticAgent ] ──────────────► Score: Faithfulness = 0.10 (Hallucination Detected)
       │
       └───► [ System 2: RAGAgent ]
                    │
                    ├──► Dense Vector Encoding (SentenceTransformers)
                    ├──► FAISS Vector Similarity Search (IndexFlatL2)
                    ├──► Context Ingestion & Grounded Generation
                    │
                    ▼
             [ CriticAgent ] ──────────────► Score: Faithfulness = 1.00 (Safe & Verified)
```

---

## 📂 Repository Structure

```
├── rag_autogen_comparison.ipynb   # Complete, fully executed evaluation notebook
├── requirements.txt               # Lightweight dependencies (free & open source)
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation and benchmark overview
└── scripts/                       # Modular assembly & validation helper scripts
```

---

## 🚀 Quickstart & Installation

This project is engineered to run **100% locally and offline without paid APIs** (no OpenAI or Anthropic keys required).

### 1. Clone the Repository
```bash
git clone https://github.com/pratik-galave/rag-autogens.git
cd rag-autogens
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Notebook
```bash
jupyter notebook rag_autogen_comparison.ipynb
```

---

## 🧪 Visualizations Generated in the Notebook

The notebook produces 4 high-resolution charts in **Section 10**:
1. **Graph 1:** Grouped bar chart comparing Faithfulness per question.
2. **Graph 2:** Grouped bar chart comparing Completeness per question.
3. **Graph 3:** Macroscopic Overall Average Benchmark Score (System 1 vs. System 2).
4. **Graph 4:** Binary Safety Heatmap (**Red = Dangerous Hallucination**, **Green = Clinically Safe & Grounded**).

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
