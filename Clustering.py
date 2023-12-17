
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

data = pd.read_csv("results.csv")

data = data.values

#KMean Clustering
def run_kmeans(n, data):
    machinekmean = KMeans(n_clusters=n)
    machinekmean.fit(data)
    resultskmean = machinekmean.predict(data)
    centroids = machinekmean.cluster_centers_
    plt.scatter(data[:, 0], data[:, 1], c=resultskmean)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="*", s=300)
    plt.savefig("scatterplot_kmeans_" + str(n) + ".png")
    plt.close()
    return silhouette_score(data, resultskmean, metric="euclidean")

#GMM Clustering
def run_gmm(n, data):
    machinegmm = GaussianMixture(n_components=n)
    machinegmm.fit(data)
    resultsgmm = machinegmm.predict(data)
    centroids = machinegmm.means_
    plt.scatter(data[:, 0], data[:, 1], c=resultsgmm)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="*", s=300)
    plt.savefig("scatterplot_gmm_" + str(n) + ".png")
    plt.close()
    return silhouette_score(data, resultsgmm, metric="euclidean")

# Running Clustering Algorithms
num_clusters = 5  
kmeans_score = run_kmeans(num_clusters, data)
gmm_score = run_gmm(num_clusters, data)

print("Silhouette Score - K-Means:", kmeans_score)
print("Silhouette Score - GMM:", gmm_score)
