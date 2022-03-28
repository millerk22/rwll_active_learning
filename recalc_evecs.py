import graphlearning as gl
from glob import glob
from joblib import Parallel, delayed
import os

def recalculate_eig(fname):
    print(fname)
    datasetname = fname.split("/")[-1].split("_")[0]
    G = gl.graph.load(fname[:-4])
    normalization = "combinatorial"
    if datasetname not in ["mnist", "mstar", "mnistimb", "mstarimb"]:
        normalization = "normalized"

    method = "exact" if G.num_nodes < 10000 else "lowrank"
    evals, evecs = G.eigen_decomp(normalization=normalization, k=200, method=method, q=150)
    print(datasetname, evals.size)
    G.save(fname[:-4])
    return

def recalculate_eig_graph(fname):
    print(fname)
    datasetname, metric = fname.split("/")[-1].split("_")
    metric = metric.split(".")[0]
    print(datasetname, metric)
    if not os.path.exists(os.path.join("data", f"{datasetname}_20.pkl")):
        X, labels = gl.datasets.load(datasetname, metric=metric)
        W = gl.weightmatrix.knn(X, 20)
        G = gl.graph(W)
        G.save(f"data/{datasetname}_20.pkl")
    recalculate_eig(f"data/{datasetname}_20.pkl")
    return
if __name__ == "__main__":
    dataset_names = [fname for fname in glob("data/*imb_*.npz") if "labels" not in fname]
    print(dataset_names)
    numcores = min(len(dataset_names), 8)
    Parallel(n_jobs=numcores)(delayed(recalculate_eig_graph)(fname) for fname in dataset_names)
    '''
    graph_fnames = glob("data/*_20.pkl")
    numcores = min(len(graph_fnames), 8)
    print(len(graph_fnames), numcores)
    print(graph_fnames)
    #Parallel(n_jobs=numcores)(delayed(recalculate_eig)(fname) for fname in graph_fnames)
    '''
