import numpy as np
from sklearn.cluster import KMeans
from .statistics import compute_laplacian

def spectral_clustering(graph, clusters_num=2, normalized=True):
    L = compute_laplacian(graph, normalized=normalized)
    eigenvalues, eigenvectors = np.linalg.eig(L)
    idx = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, idx]
    embedding = eigenvectors[:, :clusters_num]
    kmeans = KMeans(n_clusters=clusters_num)
    labels = kmeans.fit_predict(embedding.real)
    return labels
