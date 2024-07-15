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
   - Ensure IMDb movie reviews dataset is organized in the expected directory structure.
   - Update paths in the script to point to your local dataset if necessary.

2. **Dependencies:**
   - Make sure you have installed the required Python libraries:
     - `numpy`, `pandas`, `scikit-learn`
     - Specifically, `os`, `sklearn.model_selection`, `sklearn.feature_extraction.text`, `sklearn.ensemble`, `sklearn.linear_model`, `sklearn.naive_bayes`, `sklearn.svm`, `sklearn.metrics`

3. **Running the Script:**
   - Run the script to train and evaluate the models.
   - Adjust parameters or add new classifiers as needed for experimentation.

## Files

- `main.py`: Python script containing the main functionality for sentiment analysis.
- `README.md`: This file, providing an overview of the project, usage instructions, and file descriptions.
- `aclImdb` : This folder contains datasets

## Credits

- Dataset sourced from IMDb movie reviews.
