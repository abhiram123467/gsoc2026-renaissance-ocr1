# 📜 RenAIssance OCR-1: Historical Document Recognition (CNN-RNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![GSoC](https://img.shields.io/badge/GSoC-2026-FFCA28?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Project Overview
This repository contains the evaluation pipeline and prototype for the **GSoC 2026 HumanAI — RenAIssance OCR-1** project. The core objective is to evaluate and scale a Convolutional Recurrent Neural Network (CRNN) to accurately transcribe 17th-century Spanish historical texts, overcoming the challenges of rare letterforms, complex diacritics, and variable typographies.

## 🧠 System Architecture
The current prototype implements a **6.7M parameter CRNN architecture** built in PyTorch, designed to train on unsegmented text sequences using CTC Loss.

* **Feature Extraction (CNN):** A robust 3-layer Convolutional Neural Network that extracts deep visual features from grayscale, normalized document slices.
* **Sequence Modeling (BiLSTM):** A 2-layer Bidirectional LSTM (Hidden Size: 256) that captures the crucial left-to-right contextual dependencies of the historical text.
* **Decoding:** Connectionist Temporal Classification (CTC) to handle variable-length sequences without requiring character-level bounding boxes.

### Architecture Flowchart
```mermaid
graph LR
    A[Raw Image] --> B[Preprocessing]
    B --> C[3-Layer CNN]
    C --> D[Feature Sequence]
    D --> E[2-Layer BiLSTM]
    E --> F[Linear Classifier]
    F --> G[CTC Decoding]
    G --> H[Text Output]


# RenAIssance Evaluation Test I: Printed Sources OCR + LLM Integration

This branch contains the evaluation test for the CERN HSF RenAIssance project. The goal is to perform Optical Character Recognition (OCR) on 17th-century Spanish printed sources and utilize a Large Language Model (LLM) to achieve >90% transcription accuracy.

## 🛠️ Pipeline Architecture
1. **Layout Analysis:** OpenCV (Otsu's thresholding and morphological dilation) is used to isolate the primary text block from historical marginalia.
2. **Baseline Inference:** A baseline CRNN model extracts raw text features.
3. **LLM Error Correction:** Google's Gemini 2.5 Flash is integrated via a highly constrained paleographic system prompt to correct historical spelling inconsistencies and OCR hallucinations.
4. **Metric Evaluation:** Character Error Rate (CER) is calculated using Levenshtein distance against the ground truth.

---

## 📊 Primary Evaluation (Printed Sources)
The pipeline was tested on a 17th-century printed source to meet the project's core requirements.

* **Baseline CER (Raw OCR):** 14.77% 
* **Final CER (After Gemini 2.5 Flash):** 8.86%
* **Relative Error Reduction:** ~40%
* **🎯 Final Accuracy:** **91.14%** *(Successfully exceeds the >90% project requirement).*

The LLM integration proved highly effective at context-aware error correction when provided with a reasonable baseline signal.

---

## 🔬 Boundary Testing & Improvement Analysis (Handwritten Sources)
To rigorously test the limits of this architecture and justify the need for custom CRNN training, a boundary test was conducted using a complex **handwritten** 17th-century manuscript.

**Boundary Test Metrics:**
* **Baseline CER (Raw OCR):** 71.54% (71.54% Error)
* **Final CER (After Gemini):** 70.66%
* **Net Improvement:** 1.22%

**Engineering Insights:**
The baseline printed-OCR model effectively output noise when faced with cursive historical script. While the late-stage Gemini integration successfully corrected minor contextual errors (yielding a 1.22% improvement), the experiment mathematically proves a critical thesis: **an LLM cannot reconstruct accurate historical meaning from a severely degraded baseline signal.** This justifies the core objective of my GSoC proposal: We cannot rely on LLM post-processing alone. The underlying custom PyTorch CRNN *must* be specifically trained on the unique 17th-century Spanish datasets (incorporating weighted loss for rare letterforms and constrained beam search) to bring the baseline CER down to a recoverable threshold. Once that custom signal is established, the LLM can bridge the final gap to high-fidelity preservation.

