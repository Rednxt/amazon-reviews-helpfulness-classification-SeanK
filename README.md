# 📚 E-Commerce Book Reviews Helpfulness Classification: Bi-LSTM vs RoBERTa

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)

## 📌 Project Overview
This repository contains the source code and experimental framework for a Master's Thesis project titled **"Studi Komparatif Arsitektur Bi-LSTM dan RoBERTa dalam Klasifikasi Rasio Kebergunaan (Helpfulness Ratio) pada Ulasan Buku E-Commerce"**. 

The primary objective of this project is to solve the **Cold-Start Ranking Problem** in e-commerce platforms. Currently, highly analytical and long-format literature reviews are often buried at the bottom of the page because they lack initial user engagement (zero likes/votes). This automated Natural Language Processing (NLP) pipeline classifies reviews as "Helpful" (Class 1) or "Not Helpful" (Class 0) based purely on their textual semantic structure in real-time, bypassing the traditional user-upvoting mechanism.

## 📊 Dataset
The experiment utilizes the [Amazon Books Reviews Dataset from Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews). 
*   **Target File:** `Books_rating.csv`
*   **Preprocessing Pipeline:**
    1. Filtered reviews to include only those with a minimum community interaction of 5 votes (`HelpfulnessDenominator >= 5`) to establish a valid communal Ground Truth.
    2. Calculated the Helpfulness Ratio (`HelpfulnessNumerator` / `HelpfulnessDenominator`).
    3. Applied **Binary Thresholding**: Reviews with a consensus ratio of $\ge$ 0.75 are labeled as **Class 1 (Helpful)**.
    4. Sub-sampled 15,000 data points to optimize GPU memory constraints in Google Colab.
    5. Stratified Train-Test Split (80:20).

## 🧠 Methodology & Architectures
This project acts as a comparative study between traditional sequential models and modern attention-based architectures for handling long-range textual dependencies.

1.  **Baseline Model (Bi-LSTM):** 
    *   Framework: TensorFlow / Keras
    *   Architecture: Two stacked Bidirectional LSTM layers reading tokenized text forward and backward, equipped with Dropout and Batch Normalization.
    *   Loss Function: Binary Cross-Entropy.
2.  **Proposed SOTA Model (RoBERTa):**
    *   Framework: PyTorch / Hugging Face Transformers
    *   Architecture: `roberta-base` utilizing parallel Self-Attention mechanisms via Transfer Learning.
    *   Loss Function: Standard Cross-Entropy directly mapped to logits.

## 📈 Key Findings & Results
The empirical evaluation on 3,000 unseen test samples demonstrates the absolute supremacy of the RoBERTa architecture in mapping logical arguments within long-format text. The Bi-LSTM model suffered from severe *vanishing gradient* and *overfitting*, struggling to retain early context in lengthy book reviews.

| Metric | Bi-LSTM (Baseline) | RoBERTa (Proposed SOTA) |
| :--- | :---: | :---: |
| **Accuracy** | 0.6713 | **0.7427** |
| **Precision** | 0.6659 | **0.7308** |
| **Recall** | 0.8368 | **0.8604** |
| **F1-Score** | 0.7416 | **0.7903** |

### Confusion Matrix Insights
*   **Bi-LSTM Weakness:** Exhibited high **False Positives (710)**. It failed to comprehend context and was easily tricked into classifying "spam/useless" reviews as helpful.
*   **RoBERTa Dominance:** Successfully suppressed False Positives down to 536 and skyrocketed **True Positives to 1455**. The self-attention matrix successfully identified objective literary criticism, even if the review contained negative sentiment words at the end of the text.

## 🚀 How to Run the Code
1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/amazon-reviews-helpfulness-classification.git](https://github.com/yourusername/amazon-reviews-helpfulness-classification.git)
