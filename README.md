# Classify Scientific Papers Based on Abstracts

This repository contains the final project for an Artificial Intelligence course. The primary objective is to apply various Machine Learning algorithms to classify scientific papers into their respective fields based on the content of their abstracts.

## 📜 Project Overview

The project tackles a multi-class text classification problem. It involves a complete pipeline from data preprocessing and feature extraction to model training and evaluation. The goal is to determine which machine learning model performs best for this specific task.

---

## ⚙️ Methodology

The project follows these key steps:

1.  **Data Preprocessing**: The raw abstract text is cleaned by removing stop words, punctuation, and applying text normalization techniques like stemming or lemmatization.
2.  **Feature Extraction**: The cleaned text is converted into numerical vectors using the Embedding method. This allows machine learning models to process the textual data.
3.  **Model Training**: Several classic machine learning classifiers are trained on the vectorized data, including:
    * KNN
    * K-Means
    * Decision Tree
4.  **Evaluation**: Models are evaluated and compared based on standard classification metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

## 🚀 How to Run

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* Pip

### Installation

1.  Clone the repository:
    ```sh
    git clone [https://github.com/Datyth/Classify-Scientific-Paper-Based-on-Abstracts-Using-ML-Algorithms.git](https://github.com/Datyth/Classify-Scientific-Paper-Based-on-Abstracts-Using-ML-Algorithms.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd Classify-Scientific-Paper-Based-on-Abstracts-Using-ML-Algorithms
    ```
3.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## 🛠️ Technologies Used

* **Python**
* **Scikit-learn**: For ML models and metrics
* **Pandas**: For data manipulation
* **NLTK**: For text preprocessing
