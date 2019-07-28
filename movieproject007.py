# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:05:20 2019

@author: lenovo
"""

import numpy as np
import pandas as pd
df=pd.read_csv('E:/machine learning course/movie review/train/train.csv')
df=pd.read_csv('E:/machine learning course/movie review/train/train.csv')
df.head()
data=df.values
x=data[:,:-1]
y=data[:,-1]
x=list(x)
y=list(y)
for i in range (40000):
    if(y[i]=='pos'):
        y[i]=1
    else:
        y[i]=0
    x[i]=list(x[i])
print(y[0])
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
en_stopwords=set(stopwords.words('english'))
tokenizer=RegexpTokenizer("[a-zA-Z]+")
ps=PorterStemmer()
def stemmedReview(sample):
    sample=sample.lower()
    sample=sample.replace("<br/><br/>"," ")
    tokens=tokenizer.tokenize(sample)
    new_token=[token for token in tokens if token not in en_stopwords]
    stem_token=[ps.stem(token) for token in new_token]
    cleaned_review=' '.join(stem_token)
    return cleaned_review
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
from sklearn.feature_extraction.text import CountVectorizer
x_clean=[]
cv=CountVectorizer(ngram_range=(1,2))
for i in range (40000):
    s=stemmedReview(x[i][0])
    x_clean.append(s)
print(x_clean[1])
xtrainx=x_clean[0:2000]
xt=cv.fit_transform(xtrainx).toarray()
y=y[0:2000]
mnb.fit(xt,y)
dft=pd.read_csv('E:/machine learning course/movie review/test/test.csv')
x_t=dft.values
print(x_t.shape)
xt_clean=[]
for i in range (10000):
    s=stemmedReview(x_t[i][0])
    xt_clean.append(s)
xt_vec=cv.transform(xt_clean[:2000]).toarray()
Prediction=mnb.predict(xt_vec)
print(type(Prediction))
Prediction=list(Prediction)
count=0
for i in Prediction:
    if i==1:
        count+=1
print(count)
import math as m
Ratings=m.ceil((count/2000)*5)
print('MOVIE RATINGS IS',Ratings,'STARS')