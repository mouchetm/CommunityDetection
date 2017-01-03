from utils import *

import os
import pandas as pd
import time
from tqdm import tqdm

paths_to_graphs = ['../data/simple_sbm1/', '../data/simple_sbm2/','../data/complex_sbm1/','../data/complex_sbm2/']

res = []

nb_clusters = 4
number_of_rounds = 5

for round_number in range(0, number_of_rounds):
    for path in paths_to_graphs:
        graphs = os.listdir(path)
        for graph in tqdm(graphs) :
            adjency = np.load(path + graph)
            """
            First classical spherical clustering based on Laplacian
            """
            for norm in ['sym', 'rw', None]:
                t1 = time.clock()
                clusters = spectral_clustering(adjency,nb_clusters, laplacian_normalization=norm)
                tot_time = time.clock() - t1
                acc = accuracy_clustering(clusters, nb_clusters*[500])
                res.append({'name' : graph, 'norm': str(norm), 'accuracy': acc, 'time':tot_time,
                            'input_graph' : 'Laplacian', 'algo' : 'Kmeans', 'source' : path,
                            'round' : str(round_number)})
            """
            Clustering based on the adjency matrix :
            - Kmeans
            - Spherical Kmeans
            """
            t1 = time.clock()
            clusters = spherical_clustering_from_adjency(adjency, nb_clusters)
            tot_time = time.clock() - t1
            acc = accuracy_clustering(clusters, nb_clusters*[500])
            res.append({'name' : graph, 'norm': str(None), 'accuracy': acc, 'time':tot_time,
                        'input_graph' : 'Adjency', 'algo' : 'Spherical-Kmeans','source' : path,
                        'round' : str(round_number)})
            t1 = time.clock()
            clusters = clustering_from_adjency(adjency, nb_clusters)
            tot_time = time.clock() - t1
            acc = accuracy_clustering(clusters, nb_clusters*[500])
            res.append({'name' : graph, 'norm': str(None), 'accuracy': acc, 'time':tot_time,
                        'input_graph' : 'Adjency', 'algo' : 'Kmeans', 'source' : path,
                        'round' : str(round_number)})

final_results = pd.DataFrame.from_records(res)
print final_results.head()
final_results.to_json('res_4_clusters.json')
