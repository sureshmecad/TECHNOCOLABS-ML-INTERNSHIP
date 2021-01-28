#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing essential libraries
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


# Load the XGBoost model and CountVectorizer
classifier = pickle.load(open('Naive_Bayes_model.pkl', 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))


# In[6]:


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)

