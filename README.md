
# User Audience Prediction Using NLP Approaches

This project implements a **Naive Bayes-based text classification model** for predicting the audience category of user queries. It uses **Natural Language Processing (NLP)** techniques for text preprocessing, feature extraction, and classification. Additionally, the project compares the performance of various algorithms like **Naive Bayes**, **Multinomial Naive Bayes**, and **Support Vector Machines (SVM)**.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This project classifies user queries into predefined categories using NLP techniques. It evaluates the efficiency and accuracy of different algorithms, providing insights into which algorithm performs better for specific datasets.

The code was initially developed in Google Colab and includes:
- Data preprocessing (normalization, stemming, tokenization, and stop-word removal).
- Word probability calculations for Naive Bayes classification.
- Comparisons between traditional and machine learning-based approaches.

---

## Features

1. **Text Preprocessing**:
   - Tokenization, stemming, and stop-word removal.
   - Handling duplicate entries and normalizing text.

2. **Classification Algorithms**:
   - Naive Bayes (from scratch).
   - Multinomial Naive Bayes (using scikit-learn).
   - Support Vector Machines (SVM).

3. **Performance Metrics**:
   - Accuracy, F1-Score, Recall, and Confusion Matrix.

4. **Visualization**:
   - Bar charts and pie charts to compare algorithm performance.

---

## Data Preprocessing

The text is preprocessed using tools like **Parsivar** and **Hazm**, focusing on Persian text normalization. Duplicate questions are removed, and word frequencies are calculated for each category using tokenized words.

---

## Algorithms Implemented

1. **Naive Bayes (Custom Implementation)**:
   - Calculates word probabilities for each category.
   - Assigns a category to each query based on probabilities.

2. **Multinomial Naive Bayes**:
   - Utilizes scikit-learn for implementation.
   - Features vectorization using TF-IDF.

3. **Support Vector Machines (SVM)**:
   - Linear kernel-based classification.
   - Focuses on optimizing the hyperplane for maximum separation.

---

## Results

The project compares the performance of the implemented algorithms using metrics like **accuracy** and **F1-score**. Below is a high-level comparison:

| Algorithm            | Average Accuracy |
|----------------------|-------------------|
| Naive Bayes          | XX.X%            |
| Multinomial Naive Bayes | XX.X%         |
| SVM                  | XX.X%            |

*Note: Results are visualized through pie charts and bar graphs.*

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your training (`train.csv`) and testing (`test.csv`) data to the repository.

---

## Usage

1. Run the main script to preprocess data, train models, and evaluate results:
   ```bash
   python user_audience_prediction_using_nlp_approaches.ipynb
   ```
2. Modify configurations or parameters in the notebook for customizations.

