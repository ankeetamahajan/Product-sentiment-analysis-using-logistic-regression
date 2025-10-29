# Product-sentiment-analysis-using-logistic-regression
🌟 Product Review Sentiment Analysis

Unlock insights from Amazon reviews using AI and Machine Learning!

This project predicts whether a product review is Positive or Negative using Logistic Regression, providing probability scores and detailed analysis. It includes data preprocessing, visualization, ML modeling, and user input prediction.

🚀 Project Overview

The goal of this project is to perform sentiment analysis on Amazon product reviews to determine the emotional tone behind customer feedback. It can help:

Understand customer opinions

Identify product strengths and weaknesses

Assist businesses in improving products

Key Highlights:

Cleaned and processed raw review data

Visualized review trends using boxplots and word clouds

Trained Logistic Regression and Decision Tree models

Real-time prediction for new user reviews

Probability-based confidence for predictions

🗂 Dataset

File: Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv

Columns Used:

reviews.rating → Product rating (1–5)

reviews.text → Review content

Sentiment Labeling:

Positive (1) → Rating > 3

Negative (0) → Rating ≤ 3

Dataset Source: Datafiniti Amazon Reviews

🛠 Features

Data Cleaning: Lowercasing, removing URLs, punctuation, HTML tags, numbers, and empty reviews

Exploratory Data Analysis (EDA):

Boxplots of review lengths

Mean, Median, Mode of review lengths

Feature Extraction: TF-IDF Vectorization for converting text to numerical features

Machine Learning Models:

Logistic Regression for binary sentiment classification

Decision Tree for interpretability

Evaluation Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score

User Input Prediction: Enter any review and get sentiment + probability

📊 Visualizations

Review Length Distribution: See how long positive vs negative reviews are

Sentiment Distribution: Count of positive vs negative reviews

Word Clouds: Most common words in positive and negative reviews

Decision Tree: Understand model decisions visually

🧩 Installation

Clone the repository:

git clone https://github.com/yourusername/product-review-sentiment.git


Install required libraries:

pip install pandas matplotlib seaborn scikit-learn wordcloud


Upload the dataset CSV to your working directory or Google Colab.

🏃‍♂️ Usage

Load Dataset & Preprocess

import pandas as pd
df = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")


Clean Reviews

# Apply clean_text function


EDA & Visualization

# Boxplot, Mean/Median/Mode


Feature Extraction & Model Training

# TF-IDF, Logistic Regression


Evaluate Model

# Accuracy, Confusion Matrix, Classification Report


Predict User Input

user_input = input("Enter a product review: ")
predict_review(user_input)


Output Example:

Sentiment: Positive
Probability: 0.87

🌟 Future Enhancements

Use Word Embeddings (Word2Vec, GloVe, BERT) for better accuracy

Add multi-class sentiment: Positive, Neutral, Negative

Integrate a Streamlit/Flask app for live predictions

Aspect-based sentiment analysis to detect specific product features

Deploy on cloud platforms like Heroku, AWS, or GCP

📝 Authors

Ankeeta Mahajan
Email: ankeetamahajan151@gmail.com


📈 Project Impact

This project demonstrates how AI and NLP can turn unstructured text data into actionable insights. Perfect for portfolios, hackathons, or business intelligence tools.
