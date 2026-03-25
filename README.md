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


## Architecture

![CRNN Architecture](assets/crnn_architecture.png)

## Problem vs Solution

![Problem Solution](problem_solution.png)
