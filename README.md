# Sentiment Analysis on IMDB Movie Reviews Dataset

## Problem Statement
The objective of this project is to analyze the sentiment of movie reviews from the IMDB dataset. The goal is to predict whether a movie review is positive or negative based on the text provided using machine learning models, with a focus on deep learning techniques like Long Short-Term Memory (LSTM) networks.

## About Dataset
The IMDB dataset contains 50,000 movie reviews, equally split into positive and negative reviews. The dataset consists of the following:
- 35,000 reviews for training
- 15,000 reviews for testing
- Each review is labeled as either positive or negative

The dataset is preprocessed to remove stopwords, tokenized, and vectorized for model input.

## Overall Analysis and Prediction
1. **Data Preprocessing**: The data is cleaned, and reviews are tokenized using Keras' Tokenizer. The reviews are padded to a fixed length to ensure uniformity.
2. **Model Architecture**: We use an LSTM-based model to classify the sentiment of the reviews. The model is trained on the training dataset and evaluated on the test dataset.
3. **Model Evaluation**: The model's performance is evaluated based on accuracy, and the results show how effectively the model can predict positive or negative sentiments.

### Key Results:
- The model achieves a test accuracy of 90%.
- The trained model can classify new reviews into positive or negative sentiment with high accuracy.

## Requirements
- Python 3.x
- TensorFlow / Keras
- Numpy
- Pandas
- Matplotlib
- Scikit-learn
