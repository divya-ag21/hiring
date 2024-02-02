#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('HR_Comma_sep.csv')
df


# In[2]:


left = df[df['left']==1]
left.shape


# In[3]:


left = df[df['left']==1]
left.shape


# In[4]:


retain = df[df['left']==0]
retain.shape


# In[5]:


newdf = df.drop(columns = ['Department','salary'])


# In[6]:


newdf.groupby('left').mean()


# In[7]:


pd.crosstab(df['salary'],df['left']).plot(kind = 'bar')


# In[8]:


pd.crosstab(df['Department'],df['left']).plot(kind='bar')


# In[9]:


df=df.replace({'low':0,'medium':1,'high':2})


# In[10]:


df


# In[11]:


X=df.drop(columns=['last_evaluation','number_project','time_spend_company','Work_accident','left','Department'])
X


# In[12]:


Y=df['left']
Y


# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[14]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.73)


# In[15]:


model=LogisticRegression()
model.fit(X,Y)


# In[16]:


LogisticRegression()


# In[17]:


ans=model.predict([[18,22,43,0]])
ans


# In[18]:


model.score(X,Y)


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25)


# In[20]:


modelTrainTest = LogisticRegression()
modelTrainTest.fit(X_train,Y_train)


# In[21]:


traintestans = modelTrainTest.score(X_train,Y_train)
traintestans


# In[22]:





# In[ ]:




