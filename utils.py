import graphlearning as gl
import os
import numpy as np

def load_graph(dataset, metric, numeigs=200, data_dir="data"):
    X, labels = gl.datasets.load(dataset.split("-")[0], metric=metric)
    if dataset.split("-")[-1] == 'evenodd':
        labels = labels % 2
    elif dataset.split("-")[-1][:3] == "mod":
        modnum = int(dataset[-1])
        labels = labels % modnum

    if dataset.split("-")[0] == 'mstar': # allows for specific train/test set split TODO
        trainset = None
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {dataset}")
    knn = 20
    graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{knn}")

    normalization = "combinatorial"
    method = "lowrank"
    if dataset.split("-")[0] in ["fashionmnist", "cifar", "emnist"]:
        normalization = "normalized"
    if labels.size < 10000:
        method = "exact"

    try:
        G = gl.graph.load(graph_filename)
        found = True
    except:
        W = gl.weightmatrix.knn(X, knn)
        G = gl.graph(W)
        found = False

    if numeigs is not None:
        print("Computing Eigendata...")
        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs, method=method, q=150)

    if not found:
        G.save(graph_filename)

    return G, labels, trainset
