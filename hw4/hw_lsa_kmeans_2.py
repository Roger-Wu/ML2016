
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np


# In[2]:

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering


# In[3]:

import re


# In[4]:

# from nltk.corpus import stopwords
# stops = set(stopwords.words("english"))


# In[5]:

CLUSTER_COUNT = 30


# In[6]:

n_components = 20
n_features = 10000
verbose = False




import sys

# DATA_FOLDER = "data/"
DATA_FOLDER = sys.argv[1]
OUTPUT_FILE_PATH = sys.argv[2]  # output model

if DATA_FOLDER[-1] != '/':
	DATA_FOLDER = DATA_FOLDER + '/'


# In[7]:

METHOD = "lsa_kmeans"
# DATA_FOLDER = "data/"
# OUTPUT_FOLDER = "output/"
TITLE_FILE = DATA_FOLDER + "title_StackOverflow.txt"
CHECK_INDEX_FILE = DATA_FOLDER + "check_index.csv"


# In[8]:

# if not os.path.exists(OUTPUT_FOLDER):
#     os.makedirs(OUTPUT_FOLDER)


# In[9]:


# from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

# import logging
# from optparse import OptionParser
# import sys
# from time import time

# import numpy as np


# # Read and cleaning data

# In[10]:

# # substitute symbols with whitespaces, to lowercase, remove stopwords
# def cleanTitle(title):
#     return ' '.join([w for w in re.sub("[^a-zA-Z]", " ", title).lower().split() if not w in stops])


# In[11]:

# substitute symbols with whitespaces, to lowercase
def cleanTitle(title):
    return ' '.join(re.sub("[^a-zA-Z]", " ", title).lower().split())


# In[12]:

with open(TITLE_FILE) as f:
    titles = f.read().splitlines()


# In[13]:

clean_titles = [cleanTitle(title) for title in titles]


# In[14]:

clean_titles[:5]


# # Extract Features

# In[15]:


# Perform an IDF normalization on the output of HashingVectorizer
hasher = HashingVectorizer(n_features=n_features,
                           stop_words='english', non_negative=True,
                           norm=None, binary=False)
vectorizer = make_pipeline(hasher, TfidfTransformer())

X = vectorizer.fit_transform(clean_titles)

svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)


# # Clustering

# In[16]:


km = KMeans(n_clusters=CLUSTER_COUNT, init='k-means++', max_iter=100, n_init=1,
            verbose=verbose)

km.fit(X)


# In[17]:

title_clusters = np.array(km.labels_.tolist())


# In[18]:

from scipy.stats import entropy

unique, counts = np.unique(title_clusters, return_counts=True)
print(entropy(counts))
print(np.asarray((unique, counts)).T)


# # Visualize

# In[19]:

# from sklearn.manifold import TSNE
# from sklearn.decomposition import TruncatedSVD, PCA
# import matplotlib.pyplot as plt


# In[20]:

# pca = PCA(n_components=2)
# pca.fit(X)
# Y = pca.transform(X)


# In[21]:

# plt.scatter(Y[:,0], Y[:,1])
# plt.show()


# # output

# In[22]:

checkIndexDF = pd.read_csv(CHECK_INDEX_FILE)


# In[23]:

checkIndexDF["Ans"] = (title_clusters[checkIndexDF["x_ID"]] == title_clusters[checkIndexDF["y_ID"]]).astype(int)


# In[24]:

# checkIndexDF.head(20)


# In[25]:

checkIndexDF.to_csv(OUTPUT_FILE_PATH, columns=["ID", "Ans"], index=False)


# In[ ]:



