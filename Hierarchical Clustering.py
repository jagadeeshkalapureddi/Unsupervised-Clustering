#!/usr/bin/env python
# coding: utf-8
' Changing the Work Directory '
import os
os.chdir('C:\\Users\\jagad\\OneDrive\\Documents\\Python\\My Notes\\Vinod sir,\\New folder (2)\\K Means')
# In[3]:


# RBI Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from random import sample
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from pyclustertend import vat


# In[4]:


data = pd.read_csv('RBIdata.csv')


# In[5]:


data.head()


# In[6]:


data = data.set_index('state')


# In[7]:


data.head()


# In[16]:


sample = data.sample(frac = 0.29, replace = False, random_state = 123)
print(len(sample))
sample


# In[17]:


# Distance Matrix.

DM = pd.DataFrame(distance_matrix(sample.values, sample.values), index = sample.index, columns = sample.index)
round(DM,2)


# In[18]:


# Plot Distance Matrix

plt.plot(DM)
plt.ylabel('K-Distances')
plt.grid(True)
plt.show()


# In[19]:


# Visualize Distance Matrix

from pyclustertend import vat
vat(sample)


# In[20]:


data_scaled = StandardScaler().fit_transform(data)


# In[22]:


data_scaled


# In[24]:


plt.figure(figsize = (10,8))
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1,11), wcss, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('wcss')
plt.show()


# In[25]:


# Cluster membership

kmeans = KMeans(n_clusters = 4) # just making the cluster in the backned (not fitted to dataset here)
clusters = kmeans.fit_predict(data_scaled)
clusters


# In[28]:


# Let's add column 'cluster' to the data

Final_cluster = clusters + 1
Cluster = list(Final_cluster)
data['cluster'] = Cluster
data.head()


# In[29]:


data[data['cluster'] == 1]


# In[30]:


data[data['cluster'] == 2]


# In[31]:


data[data['cluster'] == 3]


# In[32]:


data[data['cluster'] == 4]


# In[33]:


# Cluster profiling

data.groupby('cluster').mean()


# In[35]:


# Plot Clusters

plt.figure(figsize = (12,6))
sns.scatterplot(data['BirthRate'], data['MortalitityRate'], hue = Final_cluster, palette = ['green', 'orange', 'blue', 'red'])


# In[ ]:




