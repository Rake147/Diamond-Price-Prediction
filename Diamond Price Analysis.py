#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[64]:


data=pd.read_csv("C:/Users/Rakesh/Datasets/diamonds.csv")


# In[65]:


data.head()


# In[66]:


data.shape


# In[67]:


data=data.drop("Unnamed: 0",axis=1)


# In[68]:


data.head()


# In[69]:


data.isnull().sum()


# # Analyzing the relationship between carrat and price of the diamond

# In[70]:


figure=px.scatter(data_frame=data,x='carat', y='price',size='depth', color='cut', trendline='ols')
figure.show()


# # We see a linear relationship between carat and price. Higher the carat, price automatically increases

# In[71]:


data['size']=data['x']*data['y']*data['z']


# In[72]:


data.head()


# # Relationship between size and price

# In[73]:


figure= px.scatter(data_frame=data, x='size', y='price', size='size', color='cut', trendline='ols')


# In[74]:


figure.show()


# There is a linear relationship between price and size

# In[75]:


fig=px.box(data, x='cut',y='price', color='color')


# In[76]:


fig.show()


# # Diamond based on their clarity

# In[77]:


fig=px.box(data_frame=data, x='cut',y='price', color='clarity')


# In[78]:


fig.show()


# In[79]:


correlation=data.corr()


# In[80]:


print(correlation['price'].sort_values(ascending=False))


# In[81]:


correlation


# In[82]:


print(correlation['carat'].sort_values(ascending=False))


# # Diamond Price Prediction

# In[83]:


data['cut']= data['cut'].map({"Ideal":1, "Premium":2,"Good":3,"Very Good":4,"Fair":5})


# Splitting the data in train and test

# In[84]:


import sklearn
from sklearn.model_selection import train_test_split


# In[85]:


data.head()


# In[86]:


x=np.array(data[["carat","cut","size"]])
y=np.array(data[["price"]])


# In[87]:


xtrain,xtest,ytrain,ytest= train_test_split(x,y, test_size=0.10, random_state=42)


# In[88]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(xtrain,ytrain)


# In[91]:


print('Enter diamond details to predict price')
a=float(input("Carat Size: "))
b=int(input("Cut Type (Ideal:1,Premium:2,Good:3,Very Good:4, Fair:5)"))
c=float(input("Size: "))
features=np.array([[a,b,c]])
print("Predicted Diamond's price= ", model.predict(features))


# So with the help of diamond details like carat, cut and size we can able to predict the price of the diamond

# In[ ]:




