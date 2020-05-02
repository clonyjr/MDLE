#!/usr/bin/env python
# coding: utf-8

# # MOVIES RECCOMENDATION - Collaborative Filtering

# In[2]:


from IPython.display import display

import pandas as pd
import numpy as np

APP_NAME = 'CollaborativeFiltering'
# path = "/Assignment_2/data/ml-latest-small/"
path = "./data/ml-latest-small/"
pathlinks = path + 'links.csv'
pathmovies = path + 'movies.csv'
pathratings = path + 'ratings.csv'
pathtags = path + 'tags.csv'


# In[3]:


ratings_df = pd.read_csv(pathratings)
movies_df = pd.read_csv(pathmovies)
movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)


# In[4]:


# display(movies_df.shape)
display(movies_df)
# movies_df.head()


# In[5]:


# display(ratings_df.shape)
display(ratings_df)
# ratings_df.head()


# ## Formatting the ratings matrix to be one row per user and one column per movie

# In[6]:


R_df = ratings_df.pivot(index = 'movieId', columns ='userId', values = 'rating').fillna(0)
# R_df.head()
display(R_df)


# ### Normalize by each movies mean
user_ratings_pd_mean = pd.DataFrame.mean(R_df, axis=1)

display(user_ratings_pd_mean)

# In[10]:


R_pd_demeaned = R_df.subtract(user_ratings_pd_mean, axis='index', fill_value=None)
display(R_pd_demeaned)


# ### Cosine Simmilarity

# In[11]:


from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(R_pd_demeaned)


# In[12]:


display(cos_sim.shape)
display(cos_sim)


# In[13]:


movies = list(R_df.columns.values)
users = list(R_df)

cos_sim_df = pd.DataFrame(data=cos_sim[1:,1:],)    # values
#                           index=R_pd_demeaned[1:,0],    # 1st column as index
#                           columns=R_pd_demeaned[0,1:])  # 1st row as the column names

display(cos_sim_df)

print("FALTA CALCULAR WEIGHTED AVERAGE")