

import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
from copy import deepcopy
from glob import glob
from scipy.special import softmax
from functools import reduce
from gl_models import * 
from acquisitions import *


from joblib import Parallel, delayed

#ACQS_MODELS = ["uncsftmax:rwll0", "uncsftmax:rwll01", "uncsftmax:rwll1", "uncnorm:rwll0", "uncnorm:rwll01", "uncnorm:rwll1",\
#        "uncsftmax:poisson", "uncsftmaxnorm:poisson", "vopt:laplace", "mcvopt:laplace", "mc:laplace"]

#ACQS_MODELS = ["random:rwll1", "uncdist:rwll0", "uncdist:rwll01", "uncdist:rwll1", "unc:rwll0", "unc:rwll01", "unc:rwll1"]

ACQS_MODELS = ["uncnorm:rwll001", "uncnorm:rwll01", "uncnorm:rwll1", "uncdist:rwll001", "uncdist:rwll01", "uncdist:rwll1", \
        "unc:rwll001", "unc:rwll01", "unc:rwll1"]

ACQS = {'unc': unc,
        'uncsftmax':uncsftmax,
        'uncdist':uncdist,
        'uncsftmaxnorm':uncsftmaxnorm,
        'uncnorm':uncnorm,
        'vopt':vopt,
        'mc':mc,
        'mcvopt':mcvopt,
        'random':random,
        'betavar':beta_var}






if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for RW Laplace Learning")
    parser.add_argument("--dataset", type=str, default='mnist-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=9)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--labelseed", type=int, default=2)
    parser.add_argument("--numtests", type=int, default=5)
    parser.add_argument("--numeigs", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.1)
    args = parser.parse_args()


    acq_funcs_names = [name.split(":")[0] for name in ACQS_MODELS]
    acq_funcs = [ACQS[name] for name in acq_funcs_names]


    X, labels = gl.datasets.load(args.dataset.split("-")[0], metric=args.metric)
    if args.dataset.split("-")[-1] == 'evenodd':
        labels = labels % 2

    if args.dataset.split("-")[0] == 'mstar': # allows for specific train/test set split TODO
        trainset = None
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {args.dataset}")
    knn = 20
    graph_filename = os.path.join("data", f"{args.dataset.split('-')[0]}_{knn}")

    normalization = "combinatorial"
    method = "lowrank"
    if args.dataset.split("-")[0] in ["fashionmnist", "cifar"]:
        normalization = "normalized"
    if labels.size < 10000:
        method = "exact"

    try:
        G = gl.graph.load(graph_filename)
    except:
        W = gl.weightmatrix.knn(X, knn)
        G = gl.graph(W)
        if np.isin(acq_funcs_names, ["mc", "vopt", "mcvopt"]).any():
            print("Computing Eigendata...")

            evals, evecs = G.eigen_decomp(normalization=normalization, k=args.numeigs, method=method, q=150, c=50)
        G.save(graph_filename)


    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
               'laplace':gl.ssl.laplace(G), # laplace learning
               'rwll0':gl.ssl.laplace(G, reweighting='poisson'), # reweighted laplace
               'rwll001':poisson_rw_laplace(G, tau=0.001), 
               'rwll01':poisson_rw_laplace(G, tau=0.01),
              'rwll1':poisson_rw_laplace(G, tau=0.1)}


    model_names = [name.split(":")[1] for name in ACQS_MODELS]
    models = [deepcopy(MODELS[name]) for name in model_names]

    if args.numcores > len(models):
        args.numcores = len(models)


    # eigendecomposition for VOpt, MC, MCVOPT criterions
    if np.isin(acq_funcs_names, ["mc", "vopt", "mcvopt"]).any():
        print("Retrieving Eigendata...")
        evals, evecs = G.eigen_decomp(normalization=normalization, k=args.numeigs, method=method, q=150, c=50)
        evals, evecs = evals[1:], evecs[:,1:]  # we will ignore the first eigenvalue/vector



    # Iterations for the different tests
    for it in range(args.numtests):
        # get initially labeled indices
        seed = int(args.labelseed + 3*(it**2))
        labeled_ind = gl.trainsets.generate(labels, rate=1, seed=seed)

        RESULTS_DIR = os.path.join("results", f"{args.dataset}_results_{seed}_{args.iters}")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        np.save(os.path.join(RESULTS_DIR, "init_labeled.npy"), labeled_ind)

        iters = args.iters

        def active_learning_test(acq_func_name, acq_func, model_name, model, show_tqdm):
            # check if test already completed previously
            choices_run_savename = os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy")
            if os.path.exists(choices_run_savename):
                print(f"Found choices for {acq_func_name} in {model_name}")
                return


            train_ind = labeled_ind.copy()
            u = model.fit(train_ind, labels[train_ind])
            acc = np.array([gl.ssl.ssl_accuracy(model.predict(), labels, train_ind.size)])

            if show_tqdm:
                iterator_object = tqdm(range(iters), desc=f"{args.dataset} test {it+1}/{args.numtests}, seed = {seed}")
            else:
                iterator_object = range(iters)

            for j in iterator_object:
                if trainset is None:
                    candidate_set = np.delete(np.arange(G.num_nodes), train_ind)
                else:
                    candidate_set = np.delete(trainset, train_ind)

                if acq_func_name == "random":
                    k = np.random.choice(candidate_set)
                else:
                    if acq_func_name in ["betavar"]:
                        acq_func_vals = acq_func(model.A)
                    elif acq_func_name in ["mc", "mcvopt", "vopt"]:
                        C_a = np.linalg.inv(np.diag(evals) + evecs[train_ind,:].T @ evecs[train_ind,:] / args.gamma**2.)
                        acq_func_vals = acq_func(u, C_a, evecs, gamma=args.gamma)
                    else:
                        acq_func_vals = acq_func(u)

                    # active learning query choice
                    acq_func_vals = acq_func_vals[candidate_set]
                    maximizer_inds = np.where(np.isclose(acq_func_vals, acq_func_vals.max()))[0]
                    k = candidate_set[np.random.choice(maximizer_inds)]


                # oracle and model update
                train_ind = np.append(train_ind, k)
                u = model.fit(train_ind, labels[train_ind])
                acc = np.append(acc, gl.ssl.ssl_accuracy(model.predict(), labels, train_ind.size))

            acc_dir = os.path.join(RESULTS_DIR, model_name)
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)
            np.save(os.path.join(acc_dir, f"acc_{acq_func_name}_{model_name}.npy"), acc)
            np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy"), train_ind)
            return

        print("------Starting Active Learning Tests-------")
        # show_bools is for tqdm iterator to track progress of one of the acquisition functions
        show_bools = np.zeros(len(models), dtype=bool)
        show_bools[::args.numcores] = True

        Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, acq, mdlname, mdl, show) for acq_name, acq, mdlname, mdl, show \
                in zip(acq_funcs_names, acq_funcs, model_names, models, show_bools))
