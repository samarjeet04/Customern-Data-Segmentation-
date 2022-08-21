#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")


# In[3]:


#importing the dataset
dataset = pd.read_csv('Wholesale customers data.csv')
dataset.shape
dataset.head(5)


# In[4]:


dataset['Channel'] = dataset['Channel'].astype('category')
dataset['Region'] = dataset['Region'].astype('category')
dataset.info()


# In[10]:


x = dataset
num_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


# In[11]:


#creating dummy variables for categorical data
cat_cols = ["Channel", "Region"]
dummies = pd.get_dummies(x[cat_cols])


# In[12]:


dummies.head(5)


# In[13]:


#combining dummy variables and numeric variables 
x1 = x[num_cols]
x2 = pd.concat([dummies, x1], axis=1)
x = x2


# In[14]:


y = dataset["Frozen"]


# In[15]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y , test_size = 0.25, random_state = 0)


# In[18]:


from sklearn.decomposition import PCA
pca = PCA (n_components = None)
xtrain_1 = pca.fit_transform(xtrain)
xtest_1 = pca.transform(xtest)
explained_variance = pca.explained_variance_ratio_
print("variaition explained by each principal component")
list(explained_variance)


# In[22]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state=45)
    kmeans.fit(xtest)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show


# In[24]:


#fitting kMeans to the dataset
x3 = xtest
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 45)
y_kmeans = kmeans.fit_predict(x3)
plt.plot(y_kmeans, 'g^')
plt.show()


# In[ ]:




