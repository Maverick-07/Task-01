#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


db = pd.read_csv('SalaryData.csv')


# In[4]:


db.columns


# In[5]:


db.info()


# In[6]:


x = db['YearsExperience']


# In[9]:


x


# In[10]:


y = db['Salary']


# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


mind = LinearRegression()


# In[17]:


x=x.values


# In[20]:


type(x)


# In[25]:


x=x.reshape(30,1)


# In[26]:


mind.fit(x,y)


# In[28]:


mind.predict([[4]])


# In[29]:


import joblib


# In[30]:


joblib.dump(mind, 'salary.pk1')


# In[ ]:




