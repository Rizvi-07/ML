#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ad_data = pd.read_csv('advertising.csv')


# In[3]:


ad_data.head()


# In[4]:


ad_data.info()


# In[5]:


ad_data.describe()


# In[6]:


sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# In[7]:


sns.jointplot(x='Age',y='Area Income',data=ad_data)


# In[8]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');


# In[9]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')


# In[10]:


sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[16]:


predictions = logmodel.predict(X_test)


# In[17]:


from sklearn.metrics import classification_report


# In[18]:


print(classification_report(y_test,predictions))


# In[ ]:




