#!/usr/bin/env python
# coding: utf-8

# <h2 style="color:blue" align="left"> 1. Import necessary Libraries </h2>

# In[1]:


# Read Data
import numpy as np                     # Linear Algebra (calculate the mean and standard deviation)
import pandas as pd                    # manipulate data, data processing, load csv file I/O (e.g. pd.read_csv)

import pickle

import warnings                        # Ignore Warnings
warnings.filterwarnings("ignore")


# <h2 style="color:blue" align="left"> 2. Load data </h2>

# ### i) Dataset :1

# In[2]:


df1 = pd.read_csv("IRAhandle_tweets_1.csv")
df2 = pd.read_csv("IRAhandle_tweets_2.csv")
df3 = pd.read_csv("IRAhandle_tweets_3.csv")
df4 = pd.read_csv("IRAhandle_tweets_4.csv")
df5 = pd.read_csv("IRAhandle_tweets_5.csv")
df6 = pd.read_csv("IRAhandle_tweets_6.csv")
df7 = pd.read_csv("IRAhandle_tweets_7.csv")
df8 = pd.read_csv("IRAhandle_tweets_8.csv")
df9 = pd.read_csv("IRAhandle_tweets_9.csv")


# ### a) Combine input datasets together

# In[3]:


df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], ignore_index=True)


# ### b) Filter only English tweets

# In[4]:


df_english = df.loc[df['language']=='English']


# In[5]:


df_english['language'].value_counts()


# In[6]:


df_english['content'] = df_english['content'].fillna(df_english['content'].mode()[0])


# In[7]:


df_content = pd.DataFrame(df_english['content'])
df_content.head()


# In[8]:


temp1 = np.ones(len(df_content))


# In[9]:


df_content['label'] = temp1
df_content.head()


# In[10]:


df_content['label'] = df_content['label'].astype(int)


# In[11]:


df_content_shuffle = df_content.take(np.random.permutation(len(df_content))[:10000])
df_content_shuffle.head()


# ### i) Dataset :2

# ### a) Labels: Negative tweets

# In[12]:


label = pd.read_csv("tweets-2016-10000-textonly.txt", delimiter='\t', names=['content'])


# In[13]:


temp2 = np.zeros(len(label))
label['label'] = temp2


# In[14]:


label['label'] = label['label'].astype(int)
label.head()


# ### iv) Final dataframe only with 'content'

# In[15]:


tweets = pd.concat([df_content_shuffle, label], ignore_index=True)


# In[16]:


tweets.head()


# In[17]:


tweets = tweets.sample(frac = 1).reset_index(drop=True)
tweets.head(10)


# In[18]:


tweets['content'] = tweets['content'].astype(str)


# In[19]:


tweets['label'].value_counts()


# <h2 style="color:blue" align="left"> 4. Text Cleaning or Preprocessing </h2>

# In[20]:


# library to clean the text 
import re  
  
# Natural Language Tool Kit 
import nltk  
  
# stopwords is a list of unwanted words like the,and,of,etc...
nltk.download('stopwords') 
  
# to remove stopword; corpus is a collection of text.
from nltk.corpus import stopwords 
  
# for Stemming propose  
# Stemming means taking the root of the word eg. loved, loving, will love -> love
# This will reduce different versions of the same word and will hence reduce the sparsity of matrix
from nltk.stem.porter import PorterStemmer


# In[21]:


# Initialize empty array to append clean text  
corpus = []

# creating PorterStemmer object to take main stem of each word
ps = PorterStemmer()

# 17708 (reviews) rows to clean 
for i in range(0, len(tweets)):         
    
    # remove html tags
    review = re.sub(r"http\S+","", tweets['content'][i])
    
    # remove special characters
    review = re.sub('[^a-zA-Z]+', ' ', review)
          
    # convert all cases to lower cases 
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # loop for stemming each word in string array at ith row     
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    
    # rejoin all string array elements to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create array of clean text  
    corpus.append(review)  


# In[22]:


corpus


# <h2 style="color:blue" align="left"> 5. Making the bag of words via sparse matrix </h2>

# In[23]:


# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
  
# To extract max 1800 feature. "max_features" is attribute to experiment with to get better results 
cv = CountVectorizer(max_features = 1800)  
  
# X contains corpus (dependent variable) 
X = cv.fit_transform(corpus).toarray()  
  
# y contains answers if review is positive or negative 
y = tweets.iloc[:, 1].values


# In[24]:


# split  data into training and testing sets of 70:30 ratio
# 20% of test size selected
# random_state is random seed
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[25]:


# shape of X & Y test / train
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# <h2 style="color:blue" align="left"> 6. Model building and Evaluation </h2>

# ### d) Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)


# In[27]:


pickle.dump(LR, open('Logitic_Regression_model.pkl','wb'))
pickle.dump(cv, open('cv-transform.pkl', 'wb'))

