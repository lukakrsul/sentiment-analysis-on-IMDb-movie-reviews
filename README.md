# IMDb Movie Reviews Sentiment Analysis

This project focuses on sentiment analysis of IMDb movie reviews using machine learning techniques. It includes data loading, preprocessing, feature extraction with TF-IDF, and evaluation of various classifiers such as Gradient Boosting, Multinomial Naive Bayes, Random Forest, Logistic Regression, and Support Vector Classifier (SVC).

## Overview

The goal of this project is to classify IMDb movie reviews into positive or negative sentiments based on the text content. It involves:

- Loading positive and negative movie reviews from local directories.
- Preprocessing text data and converting it into TF-IDF vectors.
- Training and evaluating multiple classifiers to predict sentiment.
- Assessing model performance using accuracy, classification reports, and confusion matrices.

## Usage

1. **Data Preparation:**
   - Update paths in the script to point to your local dataset if necessary.

2. **Dependencies:**
   - Python 3.6 or higher installed on your system.
   - Ensure `pip` is installed to manage Python packages.

3. **Running the Script:**
   - Clone the repository:
     ```bash
     git clone https://github.com/username/repository.git
     ```
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the script:
     ```bash
     python main.py
     ```

## Files

- `main.py`: Python script containing the main functionality for sentiment analysis.
- `README.md`: This file, providing an overview of the project, usage instructions, and file descriptions.
- `aclImdb/`: This folder contains datasets.

## Model Evaluation

The following table summarizes the performance of different supervised learning models on the sentiment analysis task:

| Model                  | Accuracy | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) | Confusion Matrix            |
|------------------------|----------|---------------|------------|--------------|---------------|------------|--------------|-----------------------------|
| **MultinomialNB**      | 0.8352   | 0.85          | 0.83       | 0.84         | 0.82          | 0.84       | 0.83         | `[[2132  441] [ 383 2044]]` |
| **LogisticRegression** | 0.8564   | 0.87          | 0.85       | 0.86         | 0.84          | 0.86       | 0.85         | `[[2183  386] [ 332 2099]]` |
| **SVC**                | 0.8614   | 0.87          | 0.85       | 0.86         | 0.85          | 0.87       | 0.86         | `[[2198  376] [ 317 2109]]` |
| **RandomForest**       | 0.8272   | 0.83          | 0.82       | 0.83         | 0.82          | 0.83       | 0.83         | `[[2096  445] [ 419 2040]]` |
| **GradientBoosting**   | 0.8038   | 0.86          | 0.77       | 0.82         | 0.75          | 0.84       | 0.79         | `[[2166  632] [ 349 1853]]` |

## Credits

- Dataset sourced from IMDb movie reviews.
