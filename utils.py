import numpy as np
import igraph
import scipy
import scipy.sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import itertools
import nltk
from nltk.cluster.kmeans import KMeansClusterer

"""
Useful function to draw graphs from SBM
"""

def simple_SBM(n_nodes, block_sizes, p_in, p_out):
    """
    SBM graph from intercluster and intrasluster probability
    """
    nb_communities = len(block_sizes)
    W = np.full((nb_communities, nb_communities), p_out)
    np.fill_diagonal(W, p_in)
    W = W.tolist()
    gr = igraph.Graph.SBM(n_nodes,W , block_sizes)
    return gr

def random_SBM(n_nodes, block_sizes, p_in_min, p_out_max):
    """
    Full random SBM
    """
    nb_communities = len(block_sizes)
    outmat = np.random.uniform(low=0, high=p_out_max, size=(nb_communities,nb_communities))
    W = 0.5 * (outmat + outmat.T)
    inmat = np.random.uniform(low=p_in_min, high=1, size=(nb_communities,))
    np.fill_diagonal(W, inmat)
    W = W.tolist()
    gr = igraph.Graph.SBM(n_nodes,W , block_sizes)
    return gr

"""
Interface between ipgraph and numpy array
"""

def get_np_adjency_matrix(graph):
    mat = np.array(graph.get_adjacency().data)
    return mat

def graph_from_array(mat):
    mat = mat.tolist()
    g = igraph.Graph.Adjacency(mat)
    return g

"""
Laplacian from the adjency matrix
"""
def get_laplacian(A, normalization_mode = None):
    """
    Compute the different laplacian of a graphs given a
    Code inspired by networkx python library
    """

    A = scipy.sparse.csr_matrix(A)
    diags = A.sum(axis=1).flatten()#Degree
    n,m = A.shape
    D = scipy.sparse.spdiags(diags, [0], m, n, format='csr')
    L = D - A

    if normalization_mode not in ['sym', 'rw', None]:
        raise Exception('Normalisation mode {} unknown'.format(normalization_mode))

    elif normalization_mode == None:
        return L

    elif normalization_mode == 'sym':
        with scipy.errstate(divide='ignore'):
            diags_sqrt = 1.0/scipy.sqrt(diags)
        diags_sqrt[scipy.isinf(diags_sqrt)] = 0
        DH = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
        return DH.dot(L.dot(DH))

    elif normalization_mode == 'rw':
        with scipy.errstate(divide='ignore'):
            diags_inverse = 1.0/diags
        diags_inverse[scipy.isinf(diags_inverse)] = 0
        DH = scipy.sparse.spdiags(diags_inverse, [0], m, n, format='csr')
        return DH.dot(L)

"""
Spectral clustering algorithms
Use the different functions computed before
"""

def spectral_clustering(A, nb_clusters, laplacian_normalization = None, algo = None):
    """
    Compute the clusters assignement from spectral clustering algorithm
    steps :
    * Compute laplacian
    * Compute k smaller eigenvalues and associated eigenvectors
    * Train a kmean on this vectors
    * Apply this kmean to the Laplacian
    """
    if algo not in ['sph', None]:
        raise Exception('Algorithm {} unknown'.format(algo))

    L = get_laplacian(A, laplacian_normalization)
    L = scipy.sparse.csr_matrix(L, dtype=np.float64)
    v, w = eigsh(L, nb_clusters, which='SM')

    if algo == None :
        km = KMeans(n_clusters= nb_clusters)
        km.fit(np.transpose(w))
        clusters = km.predict(L)

    elif algo == 'sph':
        clusterer = KMeansClusterer(nb_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
        cluster = clusterer.cluster(np.transpose(w), True)
        vectors = [np.transpose(L[i, :].toarray()[0]) for i in range(0, L.shape[1])]
        clusters = [clusterer.classify(vector) for vector in vectors]
    return clusters

def clustering_from_adjency(A, nb_clusters):
    """
    Spectral clustering with approximate kmeans
    """
    A = scipy.sparse.csr_matrix(A, dtype=np.float64)
    v, w = eigsh(A, nb_clusters, which='LM')
    km = KMeans(n_clusters= nb_clusters)
    km.fit(np.transpose(w))
    clusters = km.predict(A)
    return clusters

def spherical_clustering_from_adjency(A, nb_clusters):
    """
    Spectral clustering with spherical kmeans
    """
    A = scipy.sparse.csr_matrix(A, dtype=np.float64)
    v, w = eigsh(A, nb_clusters, which='LM')
    clusterer = KMeansClusterer(nb_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
    cluster = clusterer.cluster(np.transpose(w), True)
    vectors = [np.transpose(A[i, :].toarray()[0]) for i in range(0, A.shape[1])]
    clusters = [clusterer.classify(vector) for vector in vectors]
    return clusters

"""
Useful plot function
"""
color_list = ['red','blue','green','cyan','pink','orange','grey','yellow','white','black','purple']

def plot_communities_array(adjency_matrix, communities):
    graph = graph_from_array(adjency_matrix)
    graph = graph.as_undirected()
    vertex_col = [color_list[com] for com in communities]
    return igraph.drawing.plot(graph, vertex_color = vertex_col)

"""
Evaluate the accuracy of the clustering
"""
def accuracy_clustering(clusters, block_sizes):
    final = []
    for perm in itertools.permutations(range(0,4)):
        res = [block_sizes[i]* [e] for (i,e) in enumerate(perm)]
        res = list(itertools.chain.from_iterable(res))
        res = np.array(res)
        acc = (res == clusters).astype(int).sum()
        final.append(acc)
    val = max(final)
    return float(val)/ sum(block_sizes)
