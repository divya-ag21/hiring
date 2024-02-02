#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


hiringdf=pd.read_csv('hiring.csv')
print(hiringdf)


# In[3]:


print(hiringdf.to_string())


# In[4]:


print(hiringdf.shape)


# In[5]:


print(hiringdf.isna().sum())


# In[6]:


hiringdf['experience'] = hiringdf['experience'].fillna('zero')
hiringdf


# In[7]:


from word2number import w2n
hiringdf['experience'] = hiringdf['experience'].apply(w2n.word_to_num)
hiringdf


# In[9]:


mean=hiringdf['test_score(out of 10)'].mean()
print(mean)


# In[11]:


hiringdf['test_score(out of 10)']=hiringdf['test_score(out of 10)'].fillna(mean)
print(hiringdf)


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[14]:


X=hiringdf.drop(columns=['salary($)'])
Y=hiringdf['salary($)']


# In[15]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[16]:


print(X_train.shape)
print(X_test.shape)


# In[17]:


print(Y_train.shape)
print(Y_test.shape)


# In[18]:


model=LinearRegression()
model.fit(X_train,Y_train)
ans=model.predict(X_test)
print(ans)


# In[19]:


n=int(input("Enter n"))
for i in range(n):
    expe=int(input("Enter Experience:"))
    test=int(input("Enter testscore:"))
    inte=int(input("Enter score:"))
    ans=model.predict([[expe,test,inte]])
    print(ans)


# In[ ]:




