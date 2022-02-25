import numpy as np
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



class beta_learning(gl.ssl.ssl):
    def __init__(self, W=None, class_priors=None, propagation='poisson', tau=0., eval_cutoff=50):
        """Beta Learning
        ===================

        Semi-supervised learning
        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        propagation : str, default='poisson'
            Which propagation from labeled points to use for alpha and beta. Possibilities include: 'poisson' and 'spectral'
        eval_cutoff : int, default=50
            If propagation = 'spectral', this controls the number of eigenvalues/vectors to use in spectral propagation.
        """
        super().__init__(W, class_priors)

        self.propagation = propagation
        self.tau = tau

        if self.propagation == 'spectral':
            self.evals, self.evecs = G.eigen_decomp(normalization='combinatorial', k=eval_cutoff, method='lowrank', q=150, c=50)

        #Setup accuracy filename
        fname = '_beta'
        self.name = 'Beta Learning'

        self.accuracy_filename = fname


    def _fit(self, train_ind, train_labels, all_labels=None):
        # Poisson propagation
        if self.propagation == "poisson":
            n = self.graph.num_nodes
            onehot = gl.utils.labels_to_onehot(train_labels)
            F = np.zeros((n, onehot.shape[1]))
            F[train_ind] = onehot
            if self.tau == 0.0:
                F -= np.mean(F, axis=0)


            L = self.graph.laplacian()
            if self.tau > 0.0:
                L += self.tau*sparse.eye(L.shape[0])

            A = gl.utils.conjgrad(L, F, tol=1e-5)
            if self.tau == 0.0:
                A -= np.min(A, axis=0)

        # Spectral propagation
        elif self.propagation == "spectral":
            assert np.unique(train_labels).size == 2
            alpha, beta = prop_alpha_beta_thresh(self.evecs, self.evals, c0_ind, c1_ind, .1, thresh=1e-9)
            A = np.hstack((beta[:,np.newaxis], alpha[:,np.newaxis]))

        self.A = A + 1.
        u = self.A/self.A.sum(axis=1)[:,np.newaxis]

        return u


# Acquisition functions
def beta_var(A):
    '''
    Beta Learning's Variance -- computes Tr[C], where C is covariance matrix of entries of the Dirichlet R.V.

    Note -- this gives 2 * (Beta Variance) when # classes = 2
    '''
    a0 = A.sum(axis=1)
    a = (A * A).sum(axis=1)
    return (1. - a/(a0**2.))/(1. + a0)

def unc(u):
    u_sort = np.sort(u)
    return 1. - (u_sort[:,-1] - u_sort[:,-2]) # smallest margin acquisition function
def random(u):
    return np.random.rand(u.shape[0])


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Larger Tests in Parallel of Active Learning Test for Beta Learning")
    parser.add_argument("--dataset", type=str, default='mnist-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--tau", type=float, default=0.0)
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

    acc_models = {'poisson':gl.ssl.poisson(G), 'rwll':gl.ssl.laplace(G, reweighting='poisson'), \
                 'beta':beta_learning(G, tau=tau)}
    ############################################
    ####### Can Change these variables #########
    ############################################

    acq_funcs_names = ['beta_var', 'poisson_unc', 'rwll_unc', 'random']
    acq_funcs = [beta_var,  unc, unc, random]
    models = [acc_models['beta'], acc_models['poisson'], acc_models['rwll'], acc_models['beta']]

    RESULTS_DIR = os.path.join("results", f"{args.dataset}_results_{args.labelseed}_{args.iters}_{int(args.tau*1000)}")
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
                if model.name == 'Beta Learning':
                    u = model.fit(train_ind, labels[train_ind])
                    acq_func_vals = acq_func(model.A)
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
