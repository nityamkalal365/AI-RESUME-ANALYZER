AI Resume Analyzer — NLP-Based ATS and Job Category Prediction

Overview

The AI Resume Analyzer is an end-to-end Natural Language Processing (NLP) and Machine Learning system designed to analyze resume text, predict job categories, extract relevant technical skills, and compute ATS-style matching scores against job descriptions.

The project demonstrates the complete machine learning lifecycle, including data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, confidence estimation, and real-time deployment through a web application.

Key Features

NLP-based resume text preprocessing and normalization

TF-IDF feature engineering for numerical text representation

Tuned Logistic Regression classifier achieving over 99% accuracy

Prediction confidence estimation for reliability awareness

Skill extraction from resumes and job descriptions

ATS-style matching score computation

Interactive Streamlit web application for real-time analysis

Experimental comparison with Sentence-BERT embeddings

Machine Learning Workflow

Dataset preprocessing and text cleaning

Text vectorization using TF-IDF

Model training and comparison:

Logistic Regression

Linear Support Vector Machine

Multinomial Naive Bayes

Hyperparameter tuning using GridSearchCV with cross-validation

Performance evaluation using accuracy metrics and confusion matrix

Integration of confidence-aware prediction

Deployment through a Streamlit-based user interface

Results

Test Accuracy: 99.48%

Best Hyperparameters: C = 10, solver = liblinear

Robust handling of noisy or low-quality resume inputs

Functional ATS-style resume and job description matching

Technology Stack
Programming and Libraries

Python

Pandas, NumPy

Scikit-learn

NLTK

Sentence-Transformers

Core Concepts

Natural Language Processing

TF-IDF vectorization

Supervised text classification

Hyperparameter tuning and cross-validation

Confidence-based prediction

ATS logic simulation

Deployment

Streamlit web application

Git-based version control

Project Structure
AI-Resume-Analyzer/
│
├── data/              Dataset files
├── notebooks/         Model development and experimentation
├── src/               Saved TF-IDF vectorizer and trained model
├── app/               Streamlit web application
├── requirements.txt   Project dependencies
└── README.md          Documentation

Future Work

Fine-tuned BERT-based semantic classifier

Multi-label role prediction for hybrid skill profiles

Resume PDF parsing and upload functionality

Containerized cloud deployment using Docker

Expansion with larger real-world datasets

Author

Nityam Kalal
Aspiring Machine Learning and AI Engineer

