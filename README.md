# Reddit-Flair-Predictor
A complete end to end project to collect, clean and perform Machine Learning tasks on r/india posts to predict the flair accordingly. (Apr-2020)

## Structure

The project is divided into 5 parts:
1) Data Collection: Data collection is done using Python Reddit API Wrapper(PRAW)
   200 instances of each flair are collected, cleaned and stored in a .csv file.
   
2) Exploratory Data Analysis: The folder contains all the code used for EDA, EDA includes chi-square analysis, 
   unigrams/bigrams, analysis of length and word clouds.
   
3) Model Training: This contains the code to train the following models along with hyperparameter optimization
    The trained models are:
    1) RandomForestClassifier
    2) SupportVectorClassifier
    3) XGBoost
    4) LightGB
    5) GBM
    6) Naive Bayes
    7) A stacked ensemble to harness the power of various tree based algorithms
    
4) Webapp creation with Django: The necessary models are loaded in apps.py to reduce overhead
  paths include:
    1) Home, The url can be entered here in the format: https://www.reddit.com/r/india/comments/g7qs3q/bjp_wants_us_to_see_indian_sonia_gandhi_as/
    2) /automated_testing: This is an endpoint developed for automated testing, 
      The code for testing is given in test.py

5) Heroku Deployment: Included a .txt file with instructions for heroku deployment

6) run project: This folder is to be used to run all the jupyter notebooks
  To be used in the following order:
    1) datagen.ipynb
    2) EDA_preproc.ipynb
    3) dataset_vis.ipynb
    4) random_forest.ipynb/SVC.ipynb/lightgbm.ipynb/naive_bayes.ipynb/stacking.ipynb/xgboost_classifier.ipynb
    5) analysis.ipynb


The requirements file is requirements.txt

## Codebase
The project is developed with Python, webapp created sing the Django framework and finally deployed on Heroku

## Tech Stack
  1) sklearn
  2) nltk
  3) Django
  4) HTML/CSS
  5) pandas
  6) gunicorn
  7) XGBoost
  8) LightGBM
  
  
