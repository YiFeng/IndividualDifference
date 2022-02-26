## this is a new line for testing.
# cluster model
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np

class ClusterModel:
    def __init__(self, n_clusters: int, clustering_col_names: list[str]):
        self.n_clusters = n_clusters
        self.cluster_method = None
        self.random_state = 0
        self.clustering_col_names = clustering_col_names

    def clustering_process(self, data: DataFrame) -> np.array:
        # Standardize
        input_x = StandardScaler().fit_transform(data[self.clustering_col_names].to_numpy())
        cluster_labels = self.cluster_method.fit_predict(input_x)
        silhouette_avg = silhouette_score(input_x, cluster_labels)
        data['label'] = cluster_labels
        num_group_count = data.groupby('label').size()
        print('The silhouette score of {} is: {:.3f}'.format(self.cluster_method, silhouette_avg))
        print('Sample of each cluster: {}'.format(num_group_count))
        return input_x
    
class Kmeans(ClusterModel):
    def __init__(self, n_clusters: int, clustering_col_names: list[str]):
        ClusterModel.__init__(self, n_clusters, clustering_col_names)
        self.cluster_method = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def clustering(self, data: DataFrame):
        input_x = self.clustering_process(data)
        print('The cluster centers are: {}'.format(self.cluster_method.fit(input_x).cluster_centers_))

class EM(ClusterModel):
    def __init__(self, n_clusters: int, clustering_col_names: list[str]):
        ClusterModel.__init__(self, n_clusters, clustering_col_names)
        self.cluster_method = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
    
    def clustering(self, data: DataFrame):
        input_x = self.clustering_process(data)
        prob = self.cluster_method.predict_proba(input_x)
        for i in range(self.n_clusters):
            data['cluster_'+str(i)] = prob[:,i]
        print('The mean of each mixture component: {}'.format(self.cluster_method.fit(input_x).means_))
        return input_x

class hierarchical(ClusterModel):
    def __init__(self, n_clusters: int, clustering_col_names: list[str]):
        ClusterModel.__init__(self, n_clusters, clustering_col_names)
        self.cluster_method = AgglomerativeClustering(n_clusters=self.n_clusters, affinity = 'euclidean', linkage = 'ward')

    def clustering(self, data: DataFrame):
        input_x = self.clustering_process(data)
        dendro = sch.dendrogram(sch.linkage(input_x, method = 'ward', metric = 'euclidean'))
        
