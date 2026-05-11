# E-Commerce Book Reviews Helpfulness Classification: Bi-LSTM vs RoBERTa

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)

## Project Overview
This repository contains the source code and experimental framework for a Deep Learning project titled **"Studi Komparatif Arsitektur Bi-LSTM dan RoBERTa dalam Klasifikasi Rasio Kebergunaan (Helpfulness Ratio) pada Ulasan Buku E-Commerce"**. 

The primary objective of this project is to solve the **Cold-Start Ranking Problem** in e-commerce platforms. According to prior research (McAuley & Leskovec, 2013), highly analytical and long-format literature reviews from experts often get buried at the bottom of the page because they lack initial user engagement (zero likes/votes). This automated Natural Language Processing (NLP) pipeline classifies reviews as "Helpful" (Class 1) or "Not Helpful" (Class 0) based purely on their textual semantic structure in real-time, bypassing the traditional user-upvoting mechanism.

## Dataset
The experiment utilizes the [Amazon Books Reviews Dataset from Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews). 
* **Target File:** `Books_rating.csv`
* **Preprocessing Pipeline:**
    1. Filtered reviews to include only those with a minimum community interaction of 5 votes (`HelpfulnessDenominator >= 5`) to establish a valid communal Ground Truth.
    2. Calculated the Helpfulness Ratio (`HelpfulnessNumerator` / `HelpfulnessDenominator`).
    3. Applied **Binary Thresholding**: Reviews with a consensus ratio of >= 0.75 are labeled as **Class 1 (Helpful)**.
    4. Sub-sampled 15,000 data points to optimize GPU memory constraints in Google Colab.
    5. Stratified Train-Test Split (80:20).

## Methodology & Architectures
This project acts as an **Apple-to-Apple comparative study** between traditional sequential models and modern attention-based architectures for handling long-range textual dependencies. Both models were trained under identical constraints (Max 10 Epochs, Early Stopping Patience = 5, monitoring Validation Loss).

1.  **Baseline Model (Bi-LSTM):** * Framework: TensorFlow / Keras
    * Architecture: Two stacked Bidirectional LSTM layers reading tokenized text forward and backward, equipped with Dropout and Batch Normalization.
    * Loss Function: Binary Cross-Entropy.
2.  **Proposed SOTA Model (RoBERTa):**
    * Framework: PyTorch / Hugging Face Transformers
    * Architecture: `roberta-base` utilizing parallel Self-Attention mechanisms via Transfer Learning.
    * Loss Function: Standard Cross-Entropy directly mapped to logits.

## Key Findings & Results
The empirical evaluation on 3,000 unseen test samples demonstrates the absolute supremacy of the RoBERTa architecture in maintaining stable performance across long-format text. The Bi-LSTM model suffered from severe *vanishing gradient* and extreme *overfitting* spikes, struggling to retain early context in lengthy book reviews.

| Metric | Bi-LSTM (Baseline) | RoBERTa (Proposed SOTA) |
| :--- | :---: | :---: |
| **Accuracy** | 0.6750 | **0.7507** |
| **Precision** | 0.6721 | **0.7586** |
| **Recall** | **0.8267** | 0.8179 |
| **F1-Score** | 0.7414 | **0.7871** |

### Confusion Matrix & Analytical Insights
* **The Recall "Illusion" of Bi-LSTM:** While Bi-LSTM showed a slightly higher Recall, this is an analytical illusion. Due to its inability to comprehend long text, Bi-LSTM acted like a giant net—blindly predicting "Helpful" on almost everything. This resulted in a massive amount of **False Positives (682)**, completely destroying its Precision.
* **RoBERTa as a "Sniper" (Precision-Recall Trade-off):** The self-attention matrix allowed RoBERTa to act as a strict filter. It willingly sacrificed a tiny fraction of Recall to aggressively suppress spam. RoBERTa successfully dropped **False Positives down to 440**, skyrocketing its **Precision to 0.7586**. 

RoBERTa successfully identified objective literary criticism even if the review contained negative sentiment words at the end of the text, proving its capability to resolve the Cold-Start Problem in production environments.

## How to Run the Code
1. Clone this repository:
   ```bash
   git clone [https://github.com/Rednxt/amazon-reviews-helpfulness-classification-SeanK.git](https://github.com/Rednxt/amazon-reviews-helpfulness-classification-SeanK.git)
