#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv("Ecommerce Customers")


# In[3]:


customers.head()


# In[4]:


customers.describe()


# In[5]:


customers.info()


# In[6]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[7]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# In[8]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# In[9]:


sns.pairplot(customers)


# In[10]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# In[11]:


y = customers['Yearly Amount Spent']


# In[12]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lm = LinearRegression()


# In[17]:


lm.fit(X_train,y_train)


# In[18]:


print('Coefficients: \n', lm.coef_)


# In[19]:


predictions = lm.predict( X_test)


# In[20]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[21]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[22]:


sns.distplot((y_test-predictions),bins=50);


# In[23]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# **_As the Coeffecient value of 'Time on app' ( 38.590159 ) is much higher than 'Time on Website' ( 0.190405 ) so the company focus their efforts on mobile app._**

# In[ ]:




