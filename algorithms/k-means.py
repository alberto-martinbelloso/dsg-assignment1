#!/usr/bin/env python
# coding: utf-8

# # Mandatory assignment - Data Science in Games
# Implementation of K-Means algorithm by Rasmus Emil Odgaard
# 

# ## Import libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

#For plotting
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


# ## Load data

# In[49]:


basedir = "./"
file = "fifa.csv"
assert os.path.isdir(f"{basedir}data") and os.path.exists(f"{basedir}data/{file}"), 'Data not found. Make sure to have the most recent version!'

data = pd.read_csv(f'{basedir}/data/fifa.csv', sep=",")

all_features = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing',
            'BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina',
            'Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure',
            'Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes']

features = ['Finishing','SlidingTackle']
data = data.dropna(subset=features)



# ## K-Means algorithm

# In[50]:


def k_means_clustering(data, k, max_iter=300):
    iter = -1
    features = list(data)
    centroids = np.random.rand(k,len(features))*100
    clust_data = data.copy()    

    #for i in range (0,len(features)):
    #    clust_data=(clust_data-clust_data.min())/(clust_data.max()-clust_data.min())
            
    clust_data = clust_data.to_numpy()
                
    for i in range (0,max_iter):
        iter = i
        classifications = {}
        classification_list = []
        
        for j in range (k):
            classifications[j] = []
        
        for index, featureset in enumerate(clust_data):
            distances = [np.linalg.norm(featureset-centroids[c]) for c in range(k)]
            classification = distances.index(min(distances))
            classification_list.append(classification)
            classifications[classification].append(featureset)
            
        prev_centroids = np.copy(centroids)    
        
        for classification in classifications:
            if (len(classifications[classification]) > 0):
                centroids[classification] = np.average(classifications[classification], axis=0)
        
        if np.array_equal(prev_centroids, centroids):
            data['Centroid'] = classification_list
            break
        
    print('Final iteration: ', iter) 
    data['Centroid'] = classification_list
    return data, centroids


# In[69]:


K = 5

new_data, new_centroids = k_means_clustering(data[features].head(1000),K)


# ## Plotting

# In[70]:


def silhouette(X,cluster_labels,n_clusters, centroids,features):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.5, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = centroids
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()


# In[71]:


data_points = new_data.loc[:, features].to_numpy()
data_labels = new_data.loc[:,'Centroid'].to_numpy()
sns.pairplot(new_data, vars=features, hue='Centroid')
silhouette(data_points, data_labels, K, new_centroids, list(new_data))


# In[ ]:




