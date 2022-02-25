'''
Need to define a new gl.ssl class that does the tau reweighting

'''import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
from al_util import *
from graph_util import *
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
from glob import glob


from joblib import Parallel, delayed
import time

def get_poisson_weighting(G, train_ind, tau=0.0):
    n = G.num_nodes
    F = np.zeros(n)
    F[train_ind] = 1
    if tau == 0.0:
        F -= np.mean(F)


    L = G.laplacian()
    if tau > 0.0:
        L += tau*sparse.eye(L.shape[0])

    w = gl.utils.conjgrad(L, F, tol=1e-5)
    if tau == 0.0:
        w -= np.min(w, axis=0)

    return w

def unc(u):
    u_sort = np.sort(u)
    return 1. - (u_sort[:,-1] - u_sort[:,-2]) # smallest margin acquisition function
def random(u):
    return np.random.rand(u.shape[0])


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Larger Tests in Parallel of Active Learning Test for RW Laplace Learning")
    parser.add_argument("--dataset", type=str, default='mnist-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--numcores", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--labelseed", type=int, default=2)
    args = parser.parse_args()

    if args.tau == 0.0:
        tau = None
    else:
        tau = args.tau

    X, labels = gl.datasets.load(args.dataset.split("-")[0], metric=args.metric)

    if args.dataset.split("-")[-1] == 'evenodd':
        labels = labels % 2

    labeled_ind = gl.trainsets.generate(labels, rate=1, seed=args.labelseed)


    # Construct the similarity graph
    print(f"Constructing similarity graph for {args.dataset}")
    W = gl.weightmatrix.knn(X, 20)
    G = gl.graph(W)
    # deg = G.degree_vector()
    # deg /= deg.max()
    # def beta_degvar(u):
    #     return beta_var(u)*deg
    w = get_poisson_weighting(G, labeled_ind, tau=args.tau)
    D = sparse.spdiags(w, 0, G.num_nodes, G.num_nodes)
    Gw = gl.graph(D * W * D)

    acc_models = {'poisson':gl.ssl.poisson(G), \ # poisson learning
                'rwll0':gl.ssl.laplace(G, reweighting='poisson'), \ # reweighted laplace learning, tau = 0
                 'rwll001':gl.ssl.laplace(Gw)}
    ############################################
    ####### Can Change these variables #########
    ############################################

    acq_funcs_names = ['poisson_unc', 'rwll0_unc', 'rwll001_unc', 'random']
    acq_funcs = [unc, unc, unc, random]
    models = [acc_models['poisson'], acc_models['rwll0'], acc_models['rwll001'], acc_models['poisson']]

    RESULTS_DIR = os.path.join("results", f"{args.dataset}_results_{args.labelseed}_{args.iters}}")
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    all_accs = {}
    all_choices = {}
    iters = args.iters
    def active_learning_test(acq_func_name, acq_func, model):
        acc = {name : [] for name in acc_models}
        # compute initial accuracies in each of the accuracy models
        for name, ssl_model in acc_models.items():
            pred_labels = ssl_model.fit_predict(labeled_ind, labels[labeled_ind])
            acc[name].append(gl.ssl.ssl_accuracy(pred_labels, labels, labeled_ind.size))

        train_ind = labeled_ind.copy()

        tic = time.time()
        for it in range(iters):
            if acq_func_name == "random":
                k = np.random.choice(np.delete(np.arange(G.num_nodes), train_ind))
            else:
                u = model.fit(train_ind, labels[train_ind])
                acq_func_vals = acq_func(u)

                # active learning query choice
                maximizer_inds = np.where(np.isclose(acq_func_vals, acq_func_vals.max()))[0]
                k = np.random.choice(maximizer_inds)

            # oracle and model update
            train_ind = np.append(train_ind, k)


            for name, ssl_model in acc_models.items():
                pred_labels = ssl_model.fit_predict(train_ind, labels[train_ind])
                acc[name].append(gl.ssl.ssl_accuracy(pred_labels, labels, train_ind.size))

            if it == args.iters // 10:
                print(f"Estimated time left for {acq_func_name} = {9.*(time.time() - tic)/60. : .3f} minutes")

        print(f"\tDone with {acq_func_name}")
        acc_df = pd.DataFrame(acc)
        acc_df.to_csv(os.path.join(RESULTS_DIR, f"accs_{acq_func_name}.csv"), index=None)
        np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}.npy"), train_ind)
        return
    print("------Starting Active Learning Tests-------")
    Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, acq, mdl) for acq_name, acq,mdl in zip(acq_funcs_names, acq_funcs, models))

    # Consolidate results
    accs_fnames = glob(os.path.join(RESULTS_DIR, "*.csv"))
    dfs = []
    for fname in accs_fnames:
        df = pd.read_csv(fname)
        acq_func_name = "".join(fname.split("/")[-1].split(".")[0].split("_")[1:])
        df.rename(columns=lambda name: acq_func_name + " : " + name, inplace=True)
        dfs.append(df)
    acc_df = pd.concat(dfs, axis=1)
    acc_df.to_csv(os.path.join(RESULTS_DIR, "accs.csv"), index=None)
