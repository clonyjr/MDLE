'''
Created on Apr 10, 2020

@author: clonyjr - 94085, Andre - 62058
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster import hierarchy as shc
from sklearn.cluster import AgglomerativeClustering as agglo
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances as euc_dist
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix as dm
from scipy.spatial import distance

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
from sklearn.preprocessing import scale, normalize
import pyspark.sql.functions as F
from pyspark.sql import *
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import FloatType, IntegerType, StringType, DoubleType, ShortType, DecimalType
import sys
import math



APP_NAME = 'agglomerative'
path = "/Users/clonyjr/Library/Mobile Documents/com~apple~CloudDocs/Aveiro/UA/CLONY/MEI/2019-2020-SEM-2/MDLE/" \
       "Assignments/Assignment_2/data/fma_metadata/"
pathfeature = path + 'features.csv'
pathtracks = path + 'tracks.csv'
pathgenres = path + 'genres.csv'
pathechonest = path + 'echonest.csv'
delimeter = ','
str_features_names = ''
X = list()
y = list()

'''RangeIndex(start=0, stop=106577, step=1)'''

class Hierarchical_Clustering:
    def __init__(self, filename, num_k, feature_name_start, feature_name_end):
        self.df_feature = None
        self.df_track = None
        self.df_normalized = None
        self.df_feature_small = None
        self.df_to_clusterize = None
        self.features = None
        self.filename = filename
        self.spark = None
        self.spark_context = None
        self.spark_df = None
        self.spark_df_track = None
        self.spark_df_feature_small = None
        self.spark_df_clusterized = None
        self.pandas = pd
        self.hierarchy_cluster = shc
        self.plotter = plt
        self.cluster = None
        self.number_of_k = num_k
        self.affinity = "euclidean"
        self.affinity_linkage = "ward"
        self.feature_name_start = feature_name_start
        self.feature_name_end = feature_name_end
        self.metrics = sm
        self.density = None
        self.diameter = None
        self.radius = None
        self.dimension = 0
        
        
    # Function load data to spark
    def load_data_to_spark(self):
        """Read csv file and Return pyspark dataframe"""
        df = self.spark.read.option("header", "true") \
            .option("delimeter",delimeter)\
            .option("inferSchema", "true") \
            .csv(pathfeature) 

        return df
    
    def load_data_to_pandas(self):
        df_pandas = self.pandas.read_csv(self.filename)
        return df_pandas
        

    def initialize(self):
        print("""==== Initializing spark session, spark context, reading cvs file, creating dataframe ... ====""")
        self.spark = SparkSession.builder.appName(APP_NAME).getOrCreate()
        self.spark_context = SparkContext.getOrCreate()

        sns.set_context("notebook", font_scale=1.5)
        self.plotter.rcParams['figure.figsize'] = (17, 5)
        self.df_feature = self.load_data_to_pandas()
        self.df_feature_small = self.Create_Small_Dataset(pathtracks,self.df_feature)
        list_features_name = list(self.df_feature.loc[3:,self.feature_name_start:self.feature_name_end])
        self.df_to_clusterize = self.df_feature_small.loc[3:,self.feature_name_start:self.feature_name_end]
        self.dimension = self.df_to_clusterize.ndim
        #self.df_to_clusterize = self.Data_Process(self.df_feature_small.loc[3:,self.feature_name_start:self.feature_name_end])

        
        # ====== Spark to Pandas ======
        # df_pd = self.df_chroma_mean.toPandas()
        # df_pd_track_small = self.df_feature_small.toPandas()
        # ====== spark dataframe schema ======
        # # Pandas to Spark
        self.spark_df_feature_small = self.spark.createDataFrame(self.df_to_clusterize)
        

    def Data_Process(self, df2):
        print('===== Processing the data ... =====')
        self.df_normalized = normalize(df2)
        self.df_normalized = self.pandas.DataFrame(self.df_normalized, columns=df2.columns)
        return self.df_normalized
        
    def Create_Small_Dataset(self, file_to_read, df_to_compare):
        df_of_small = self.pandas.read_csv(file_to_read)
        
        temp_dataset = df_of_small.loc[df_of_small['set.1'] == 'small']
        temp_dataset['track_id'] = temp_dataset['track_id'].apply(self.pandas.to_numeric)
        track_id_list = temp_dataset["track_id"].to_list()
        df_to_compare[3:] = df_to_compare[3:].apply(self.pandas.to_numeric)
        df_to_return = df_to_compare[df_to_compare['feature'].isin(track_id_list)]
        return df_to_return
        
    def Create_Dendrogram(self):
        self.plotter.figure(figsize=(10, 7))
        self.plotter.title("Dendrograms")
        self.plotter.xlabel("Features")
        self.plotter.ylabel("Euclidean Distance")
        self.hierarchy_cluster.dendrogram(self.hierarchy_cluster
                                                 .linkage(self.df_to_clusterize, method=self.affinity_linkage))
        self.plotter.axhline(y=100000, color='r', linestyle='--')
        self.plotter.show()

    def Clustering(self):
        print('===== Clustering ... =====')
        self.cluster = agglo(n_clusters=self.number_of_k, affinity=self.affinity,
                                               linkage=self.affinity_linkage)
        self.cluster.fit_predict(self.df_to_clusterize)
        print('===== Points of clusters (''0'' are points of cluster 1, ''1'' are points of cluster 2 =====')
        print(self.cluster.fit(self.df_to_clusterize.to_numpy()))
        

    def Plotting_Cluster(self):
        print('===== Ploting the Clusters ... =====')
        self.cluster.fit_predict(self.df_to_clusterize.iloc[:, :-1].values)
        self.plotter.scatter(self.df_to_clusterize.iloc[:,0].values, self.df_to_clusterize.iloc[:,-1].values, c=self.cluster.labels_, cmap='rainbow')
        self.plotter.xlabel("Cluster Size")
        self.plotter.ylabel("Distance")
        #divide the cluster
        #self.plotter.axhline(y=15)
        #self.plotter.axhline(5)
        #self.plotter.axhline(10)
        self.plotter.show()

    #Create 3d visualization
    def Plot_3d_Visualization(self):
        print('===== Ploting the Clusters in 3D ... =====')
        threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
        threedee.scatter(self.df_to_clusterize.iloc[:,0].values, self.df_to_clusterize.iloc[:,-1].values,
                             c=self.cluster.labels_, cmap="seismic")
        threedee.set_title('Clustering')
        threedee.set_xlabel(self.feature_name_start)
        threedee.set_ylabel(self.feature_name_end)
        threedee.set_zlabel('some feature')
        plt.show()

    # Not working
    def plot_distance(self):
        print('===== Ploting the distance ... =====')
        labels = ('Cluster 1', 'Cluster 2', 'Cluster 3')
        for index, metric in enumerate(["euclidean"]):
            self.plotter.figure(figsize=(5, 4.5))
            avg_dist = np.zeros((self.number_of_k, self.number_of_k))
            for i in range(self.number_of_k):
                for j in range(self.number_of_k):
                    avg_dist[i, j] = pairwise_distances(X[y == i], X[y == j],
                                                        metric=metric).mean()
            avg_dist /= avg_dist.max()
            for i in range(self.number_of_k):
                for j in range(self.number_of_k):
                    self.plotter.text(i, j, '%5.3f' % avg_dist[i, j],
                             verticalalignment='center',
                             horizontalalignment='center')
            self.plotter.imshow(avg_dist, interpolation='nearest', cmap=self.plotter.cm.gnuplot2,vmin=0)
            self.plotter.xticks(range(self.number_of_k), labels, rotation=45)
            self.plotter.yticks(range(self.number_of_k), labels)
            self.plotter.colorbar()
            self.plotter.suptitle("Interclass %s distances" % metric, size=18)
            self.plotter.tight_layout()        
            
    def calculate_euclidean_distance(self, df_to_calculate_distance):
        """
        Euclidean Distance: https://en.wikipedia.org/wiki/Euclidean_distance
        assuming that two data points have same dimension
        """

        max_distance = 0.00
        min_distance = 999999.99
        Train = df_to_calculate_distance.to_numpy()
        Train[0].reshape(-1,1)
        print("Euclidean Distance: ", euc_dist(Train))
        result = euc_dist(Train)
        for i in result:
            if (i > max_distance).any():
                max_distance = i
         
        for j in result:       
            if (j < min_distance).all():
                min_distance = j
        
        self.diameter = max_distance
        return result

    def compute_centroid_of_two_clusters(self, current_clusters, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0]*dim
        for index in data_points_index:
            dim_data = current_clusters[str(index)]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    def compute_centroid(self, dataset, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0]*dim
        for idx in data_points_index:
            dim_data = dataset.iloc[idx]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid
            


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    filename = pathfeature
    num_k = 6
    feature_in = 'chroma_cens'
    feature_out = 'zcr.6'

    hc = Hierarchical_Clustering(filename, num_k, feature_in, feature_out)
    hc.initialize()
    hc.Create_Dendrogram()
    hc.Clustering()
    hc.Plotting_Cluster()
    hc.Plot_3d_Visualization()
    hc.calculate_euclidean_distance(hc.df_to_clusterize)
    ## compute_centroid() Not working
    #print(hc.compute_centroid(hc.df_to_clusterize, hc.df_to_clusterize.index))