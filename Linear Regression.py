#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import os


# In[2]:


df=pd.read_csv("House_Price .csv")
df


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Area(sqrt)')
plt.ylabel('Price ')
plt.scatter(df.Area,df.Price, color='red',marker='*')


# In[4]:


reg=linear_model.LinearRegression()
reg.fit(df[['Area']],df.Price)


# In[15]:


reg.predict([[11]])


# In[ ]:





# In[ ]:




