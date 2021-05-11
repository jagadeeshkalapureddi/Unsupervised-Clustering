#!/usr/bin/env python
# coding: utf-8

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


df = pd.read_csv('RBIdata.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


# Set the state as an index.
df = df.set_index('state')


# In[10]:


df.head()


# In[11]:


# Take 10 random sample from the data.
sample = df.sample(frac = 0.29, replace = False, random_state = 123)
print(len(sample))
sample


# In[12]:


# Distance matrix.

DM = pd.DataFrame(distance_matrix(sample.values, sample.values), index = sample.index, columns = sample.index)


# In[13]:


round(DM,2)


# In[14]:


plt.plot(DM)
plt.ylabel('k-distances')
plt.grid(True)
plt.show()


# In[15]:


vat(sample)


# In[16]:


data_scaled = StandardScaler().fit_transform(df[['BirthRate', 'MortalitityRate', 'PowerAvailability', 'RoadLength']])


# In[17]:


data_scaled


# In[18]:


plt.figure(figsize = (10,3))
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


# In[19]:


# Cluster membership

kmeans = KMeans(n_clusters = 4) # just making the cluster in the backned (not fitted to dataset here)
clusters = kmeans.fit_predict(data_scaled)
clusters


# In[20]:


# Let's add column 'cluster' to the data

Final_cluster = clusters + 1
Cluster = list(Final_cluster)
df['cluster'] = Cluster
df.head()


# In[21]:


df[df['cluster'] == 1]


# In[22]:


df[df['cluster'] == 2]


# In[23]:


df[df['cluster'] == 3]


# In[24]:


df[df['cluster'] == 4]


# In[25]:


# Cluster profiling

df.groupby('cluster').mean()


# In[26]:


# Plot Clusters

plt.figure(figsize = (12,6))
sns.scatterplot(df['BirthRate'], df['MortalitityRate'], hue = Final_cluster, palette = ['green', 'orange', 'blue', 'red'])


# In[27]:


# Silhouette Score
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(clusters)
n_cluster = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(data_scaled, clusters, metric = 'euclidean')
y_ax_lower, y_ax_upper = 0,0
yticks = []

for i,c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[clusters == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_cluster)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height = 1, edgecolor = 'none', color = color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color = 'red', linestyle = '--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()


# In[28]:


# Avg silhouette score

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data_scaled, clusters)
silhouette_avg


# In[29]:


sample_silhouette_values = silhouette_samples(data_scaled, clusters)
sample_silhouette_values


# In[30]:


from sklearn.metrics import silhouette_score

range_n_clusters = list(range(2,10))
print('Number of clusters from 2 to 9: \n', range_n_clusters)

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters = n_clusters)
    preds = clusterer.fit_predict(data_scaled)
    centers = clusterer.cluster_centers_
    
    score = silhouette_score(data_scaled, preds)
    print('For n_clusters = {}, silhouette score is {}'. format(n_clusters, score))


# In[31]:


# Cluster membership

kmeans = KMeans(n_clusters = 3) # just making the cluster in the backned (not fitted to dataset here)
clusters = kmeans.fit_predict(data_scaled)
clusters


# In[32]:


# Let's add column 'cluster' to the data

Final_cluster = clusters + 1
Cluster = list(Final_cluster)
df['cluster'] = Cluster
df.head()


# In[33]:


df.to_excel('rbiCluster.xlsx')


# In[34]:


# Cluster profiling

df.groupby('cluster').mean()


# In[35]:


# Plot Clusters

plt.figure(figsize = (12,6))
sns.scatterplot(df['BirthRate'], df['MortalitityRate'], hue = Final_cluster, palette = ['green', 'red', 'blue'])


# In[36]:


# Silhouette Score
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(clusters)
n_cluster = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(data_scaled, clusters, metric = 'euclidean')
y_ax_lower, y_ax_upper = 0,0
yticks = []

for i,c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[clusters == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_cluster)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height = 1, edgecolor = 'none', color = color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color = 'red', linestyle = '--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()


# In[37]:


# Avg silhouette score

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data_scaled, clusters)
silhouette_avg


# In[38]:


sample_silhouette_values = silhouette_samples(data_scaled, clusters)
sample_silhouette_values


# In[39]:


from sklearn.metrics import silhouette_score

range_n_clusters = list(range(2,10))
print('Number of clusters from 2 to 9: \n', range_n_clusters)

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters = n_clusters)
    preds = clusterer.fit_predict(data_scaled)
    centers = clusterer.cluster_centers_
    
    score = silhouette_score(data_scaled, preds)
    print('For n_clusters = {}, silhouette score is {}'. format(n_clusters, score))


# In[ ]:





# In[40]:


data = pd.DataFrame(data_scaled)


# In[41]:


Final_cluster = clusters + 1
Cluster = list(Final_cluster)
data['cluster'] = Cluster
data.head()


# In[42]:


data.to_excel('data.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:




