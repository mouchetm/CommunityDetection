from utils import *
from tqdm import tqdm

outdir = '../data/simple_sbm1/'
#outdir = '../data/simple_sbm2/'
#outdir = '../data/complex_sbm1/'
#outdir = '../data/complex_sbm2/'

number_of_graphs = 50

p_in, p_out = 0.9, 0.1 #change for complexity of recovery of the clusters
#p_in, p_out = 0.7, 0.3
#p_in_min, p_out_max = 0.9, 0.1
#p_in_min, p_out_max = 0.7, 0.3
nodes_by_clusters = 500
nb_clusters = 4
block_sizes = nb_clusters * [nodes_by_clusters]
n_nodes = sum(block_sizes)

for i in tqdm(range(0, number_of_graphs)):
    gr = simple_SBM(n_nodes, block_sizes, p_in, p_out)
    #gr = random_SBM(n_nodes, block_sizes, p_in_min, p_out_max)
    A = get_np_adjency_matrix(gr)
    np.save(outdir + 'matrix_'+str(i)+'.npy', A)

outdir = '../data/simple_sbm2/'
p_in, p_out = 0.7, 0.3
for i in tqdm(range(0, number_of_graphs)):
    gr = simple_SBM(n_nodes, block_sizes, p_in, p_out)
    #gr = random_SBM(n_nodes, block_sizes, p_in_min, p_out_max)
    A = get_np_adjency_matrix(gr)
    np.save(outdir + 'matrix_'+str(i)+'.npy', A)

outdir = '../data/complex_sbm1/'
p_in_min, p_out_max = 0.9, 0.1

for i in tqdm(range(0, number_of_graphs)):
    gr = simple_SBM(n_nodes, block_sizes, p_in, p_out)
    #gr = random_SBM(n_nodes, block_sizes, p_in_min, p_out_max)
    A = get_np_adjency_matrix(gr)
    np.save(outdir + 'matrix_'+str(i)+'.npy', A)

outdir = '../data/complex_sbm2/'
p_in_min, p_out_max = 0.7, 0.3

for i in tqdm(range(0, number_of_graphs)):
    gr = simple_SBM(n_nodes, block_sizes, p_in, p_out)
    #gr = random_SBM(n_nodes, block_sizes, p_in_min, p_out_max)
    A = get_np_adjency_matrix(gr)
    np.save(outdir + 'matrix_'+str(i)+'.npy', A)
