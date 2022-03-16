import graphlearning as gl
from glob import glob
from joblib import Parallel, delayed

def recalculate_eig(fname):
    print(fname)
    datasetname = fname.split("/")[-1].split("_")[0]
    G = gl.graph.load(fname[:-4])
    normalization = "combinatorial"
    if datasetname not in ["mnist", "mstar"]:
        normalization = "normalized"

    method = "exact" if G.num_nodes < 10000 else "lowrank"
    evals, evecs = G.eigen_decomp(normalization=normalization, k=200, method=method, q=150)
    print(datasetname, evals.size)
    return



if __name__ == "__main__":
    graph_fnames = glob("data/*_20.pkl")
    numcores = min(len(graph_fnames), 8)
    print(len(graph_fnames), numcores)
    # Parallel(n_jobs=numcores)(delayed(recalculate_eig)(fname) for fname in graph_fnames)
