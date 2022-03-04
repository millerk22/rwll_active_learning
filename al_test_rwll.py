'''
Notes:
    * why did tau = 0.001 do so weird?
    * just send in objects to parallel function
'''

import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
from glob import glob
from scipy.special import softmax
from functools import reduce
from gl_models import poisson_rw_laplace
from acquisitions import *


from joblib import Parallel, delayed

ACQS_MODELS = ["vopt:laplace", "uncnorm:rwll01", "uncnorm:rwll1", "uncnorm:rwll0", "mc:laplace", "mcvopt:laplace"]

ACQS = {'unc': unc,
        'uncsftmax':uncsftmax,
        'uncdist':uncdist,
        'uncsftmaxnorm':uncsftmaxnorm,
        'uncnorm':uncnorm,
        'vopt':vopt,
        'mc':mc,
        'mcvopt':mcvopt,
        'random':random}




if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for RW Laplace Learning")
    parser.add_argument("--dataset", type=str, default='mnist-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=6)
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

    # Construct the similarity graph
    print(f"Constructing similarity graph for {args.dataset}")
    knn = 20
    graph_filename = os.path.join("data", f"{args.dataset}_{knn}")
    try:
        G = gl.graph.load(graph_filename)
    except:
        W = gl.weightmatrix.knn(X, knn)
        G = gl.graph(W)
        if np.isin(acq_funcs_names, ["mc", "vopt", "mcvopt"]).any():
            print("Computing Eigendata...")
            evals, evecs = G.eigen_decomp(normalization="combinatorial", k=args.numeigs, method="lowrank", q=150, c=50)
        G.save(graph_filename)

    # while not G.isconnected():
    #     print(f"Graph not connected with knn = {knn}, using knn = {knn+10}")
    #     knn += 10
    #     graph_filename = os.path.join("data", f"{args.dataset}_{knn}")
    #     try:
    #         G = gl.graph.load(graph_filename)
    #     except:
    #         W = gl.weightmatrix.knn(X, knn)
    #         G = gl.graph(W)
    #         if np.isin(acq_funcs_names, ["mc", "vopt", "mcvopt"]).any():
    #             print("Computing Eigendata...")
    #             evals, evecs = G.eigen_decomp(normalization="combinatorial", k=args.numeigs, method="lowrank", q=150, c=50)
    #         G.save(graph_filename)

    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
                'laplace':gl.ssl.laplace(G), # laplace learning (no reweighting)
                'rwll0':gl.ssl.laplace(G, reweighting='poisson'),  # reweighted laplace learning, tau = 0
                'rwll001':poisson_rw_laplace(G, tau=0.001),  # reweighted laplace learning, tau = 0.001
                 'rwll01':poisson_rw_laplace(G, tau=0.01),  # reweighted laplace learning, tau = 0.01
                 'rwll1':poisson_rw_laplace(G, tau=0.1)}   # reweighted laplace learning, tau = 0.1

    model_names = [name.split(":")[1] for name in ACQS_MODELS]
    models = [MODELS[name] for name in model_names]

    if args.numcores > len(models):
        args.numcores = len(models)


    # eigendecomposition for VOpt, MC, MCVOPT criterions
    if np.isin(acq_funcs_names, ["mc", "vopt", "mcvopt"]).any():
        print("Retrieving Eigendata...")
        evals, evecs = G.eigen_decomp(normalization="combinatorial", k=args.numeigs, method="lowrank", q=150, c=50)
        evals, evecs = evals[1:], evecs[:,1:]  # we will ignore the first eigenvalue/vector



    # Iterations for the different tests
    for it in range(args.numtests):
        # get initially labeled indices
        seed = int(args.labelseed + 3*(it**2))
        labeled_ind = gl.trainsets.generate(labels, rate=1, seed=seed)

        RESULTS_DIR = os.path.join("results", f"{args.dataset}_results_{seed}_{args.iters}")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        iters = args.iters

        def active_learning_test(acq_func_name, acq_func, model_name, model, show_tqdm):
            # check if test already completed previously
            acc_run_savename = os.path.join(RESULTS_DIR, f"accs_{acq_func_name}_{model_name}.csv")
            if os.path.exists(acc_run_savename):
                # check if there is an evaluation model that has not yet been evaluated for this set of choices
                completed_df = pd.read_csv(acc_run_savename)
                not_evaluated = [name for name in MODELS if name not in completed_df.columns]
                if len(not_evaluated) == 0:
                    return
                print(f"Found evaluation models that do not have recorded accuracies for {acq_func_name} in {model_name}")
                acc = {name: [] for name in not_evaluated}
                choices = np.load(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy"))

                if show_tqdm:
                    iterator_object = tqdm(range(labeled_ind.size,choices.size), desc=f"{args.dataset} test {it+1}/{args.numtests}, seed = {seed}")
                else:
                    iterator_object = range(labeled_ind.size,choices.size)

                for i in iterator_object:
                    train_ind = choices[:i]
                    for name in not_evaluated:
                        pred_labels = MODELS[name].fit_predict(train_ind, labels[train_ind])
                        acc[name].append(gl.ssl.ssl_accuracy(pred_labels, labels, train_ind.size))

                new_df = pd.DataFrame(acc)
                completed_df = pd.concat([completed_df, new_df], axis=1)
                completed_df.to_csv(acc_run_savename, index=None)
                return

            # Run test since don't have any previously completed run
            acc = {name : [] for name in MODELS}
            # compute initial accuracies in each of the accuracy models
            for name, ssl_model in MODELS.items():
                pred_labels = ssl_model.fit_predict(labeled_ind, labels[labeled_ind])
                acc[name].append(gl.ssl.ssl_accuracy(pred_labels, labels, labeled_ind.size))

            train_ind = labeled_ind.copy()

            if show_tqdm:
                iterator_object = tqdm(range(iters), desc=f"{args.dataset} test {it+1}/{args.numtests}, seed = {seed}")
            else:
                iterator_object = range(iters)

            for j in iterator_object:
                if acq_func_name == "random":
                    k = np.random.choice(np.delete(np.arange(G.num_nodes), train_ind))
                elif acq_func_name in ["vopt", "mc", "mcvopt"]:
                    u = model.fit(train_ind, labels[train_ind])
                    C_a = np.linalg.inv(np.diag(evals) + evecs[train_ind,:].T @ evecs[train_ind,:] / args.gamma**2.)
                    acq_func_vals = acq_func(u, C_a, evecs, gamma=args.gamma)
                    maximizer_inds = np.where(np.isclose(acq_func_vals, acq_func_vals.max()))[0]
                    k = np.random.choice(maximizer_inds)
                else:
                    u = model.fit(train_ind, labels[train_ind])
                    acq_func_vals = acq_func(u)

                    # active learning query choice
                    maximizer_inds = np.where(np.isclose(acq_func_vals, acq_func_vals.max()))[0]
                    k = np.random.choice(maximizer_inds)

                # oracle and model update
                train_ind = np.append(train_ind, k)


                for name, ssl_model in MODELS.items():
                    pred_labels = ssl_model.fit_predict(train_ind, labels[train_ind])
                    acc[name].append(gl.ssl.ssl_accuracy(pred_labels, labels, train_ind.size))

            acc_df = pd.DataFrame(acc)
            acc_df.to_csv(acc_run_savename, index=None)
            np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy"), train_ind)
            return

        print("------Starting Active Learning Tests-------")
        # show_bools is for tqdm iterator to track progress of one of the acquisition functions
        show_bools = len(models)*[False]
        show_bools[0] = True

        Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, acq, mdlname, mdl, show) for acq_name, acq, mdlname, mdl, show \
                in zip(acq_funcs_names, acq_funcs, model_names, models, show_bools))

        # Consolidate results
        print(f"Consolidating results of run to: {os.path.join(RESULTS_DIR, 'accs.csv')}...")
        accs_fnames = glob(os.path.join(RESULTS_DIR, "accs_*.csv")) # will NOT include accs.csv (previously consolidated file)
        dfs = []
        for fname in accs_fnames:
            df = pd.read_csv(fname)
            acq_func_name = "".join(fname.split("/")[-1].split(".")[0].split("_")[1:])
            df.rename(columns=lambda name: acq_func_name + " : " + name, inplace=True)
            dfs.append(df)
        acc_df = pd.concat(dfs, axis=1)
        acc_df.to_csv(os.path.join(RESULTS_DIR, "accs.csv"), index=None)




    # Get average and std curves over all tests
    overall_results_dir = os.path.join("results", f"{args.dataset}_results_{args.iters}")
    if not os.path.exists(overall_results_dir):
        os.makedirs(overall_results_dir)
    overall_results_file = os.path.join(overall_results_dir, "stats.csv")
    print(f"Saving results over all runs to: {overall_results_file}")
    acc_files = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", "accs.csv"))
    dfs = [pd.read_csv(f) for f in sorted(acc_files)]
    possible_columns = reduce(np.union1d, [df.columns for df in dfs])
    all_columns = {}
    for col in possible_columns:
        vals = np.array([df[col].values for df in dfs if col in df.columns])
        all_columns[col + " : avg"] = np.average(vals, axis=0)
        all_columns[col + " : std"] = np.std(vals, axis=0)

    all_df = pd.DataFrame(all_columns)
    all_df.to_csv(overall_results_file, index=None)
