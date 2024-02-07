# Sentiment Analysis on Moroccan Arabic Dialect

### Name:brahim louardi


## Overview

This readme file presents an overview of the proposed solution for the Sentiment Analysis on Moroccan Arabic Dialect competition, designed for Master WISD/MASD students at Sidi Mohamed Ben Abdellah University under the supervision of Professor El Habib Nfaoui. The competition revolves around predicting the sentiment (positive or negative) of comments written in the Moroccan Arabic dialect.

## Dataset Information

### Files
- **train.csv:** Training set
- **test_stage1.csv:** Test set for the first competition stage
- **sample_submission.csv:** Sample submission file for test set predictions

### Data Format
- **id:** Unique identifier for each comment
- **comment:** The text of a comment
- **label (in train.csv only):** Denotes sentiment (1 for positive, 0 for negative)

### Task
Train a machine learning model on the training dataset to predict sentiment in user comments and make predictions for the test dataset.

## Solution Summary

### Requirements
- General: `numpy`, `pandas`, `nltk`
- Specific: `gensim`, `scikit-learn`, `xgboost`, `tensorflow`, `fasttext`
- FastText is an open-source, free library from Facebook AI Research(FAIR) for learning word embeddings and word classifications. This model allows creating unsupervised learning or supervised learning algorithm for obtaining vector representations for words. It also evaluates these models. FastText supports both CBOW and Skip-gram models.

### Preprocessing:
In the preprocessing step,I used the NLTK library to clean and prepare the dataset for training. The following transformations are applied:
Character Normalization: Specific characters such as 'ة' and 'ى' are replaced with their standardized forms ('ه' and 'ي').
Stop Word Removal: Common Arabic stop words, including articles and prepositions, are removed to focus on meaningful content.
Newline Characters: Any newline characters are replaced with spaces.
These preprocessing steps collectively enhance the quality of the text data, making it more suitable for training models in the Sentiment Analysis on Moroccan Arabic Dialect competition.

### Solution 1 Steps

1. **FastText Pre-trained Model:**
   - utilize and fine-tune the pre-trained FastText model for Arabic sentiment analysis directly.
   - not so great results but we use the produced embeddings 



### Solution 2 Steps

1. **Feature Extraction(Word Embeddings) (FastText):**
   - Creating sentences embedding and using them as an input for ML model
   - In the Embeddings phase, FastText for arabic is employed to generate word embeddings for the given text data. The following steps outline this process:
   - FastText is good for this small and domain specific dataset because it handle Out of Vocabulary word better then word2vec ,bert and other models

2. **Model Training:**
   - Train various classifiers, including SVM, Logistic Regression, XGBoost,Random Forest, and Neural Networks.




### Evaluation:
- Accuracy list on the test set of different models:
  - Fine-Tuned PreTrained FastText Model: Accuracy : 0.7753424657534247
  - SVM with rbf Kernel:                  Accuracy : 0.9557291666666666
  - Logistic Regression:                  Accuracy : 0.9557291666666666
  - XGBoost:                              Accuracy : 0.953125
  - Random Forest:                        Accuracy : 0.953125
  - Deep learning model:                  Accuracy : 0.9479166865348816
  
  
  
  

### Notebook in Google Colab for more detail
  [google colab](https://colab.research.google.com/drive/1sAXJAetIauDBnE311RVmPq2l11p2GUpx?usp=sharing)

### Additional Notes

Two other approaches were tested for sentiment analysis on Moroccan Arabic Dialect:

1. **AraVec Approach:**
   - Utilized [AraVec](https://github.com/bakrianoo/aravec), an open-source project providing pre-trained distributed word representations (word embeddings) for the Arabic NLP research community.
   - The objective was to create word embeddings using AraVec and feed them into various ML and DL models such as LSTM, SVM, and stacking.
   - Unfortunately, the models did not generalize well.

2. **Hugging Face BERT-based Model Approach:**
   - Employed a [pre-trained BERT-based model](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment)   specifically designed for Arabic dialects from the Hugging Face model hub.
   - Despite using a sophisticated pre-trained model, the results did not meet expectations.
   - Colab Notebook: [BERT-based Model Approach](https://colab.research.google.com/drive/1S6YgZAz4UqUvAUDazyDuk3F3OkaUtzDM?usp=sharing)

These approaches, involving AraVec and a BERT-based model, were tested using Google Colab notebooks. Unfortunately, neither approach yielded satisfactory generalization results for sentiment analysis on Moroccan Arabic Dialect.




