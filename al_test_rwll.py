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


from joblib import Parallel, delayed
import time


# Wrap joblib with tqdm as a context manager to show progress bar
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class poisson_rw_laplace(gl.ssl.ssl):
    def __init__(self, W=None, class_priors=None, tau=0.0, normalization='combinatorial', tol=1e-5, alpha=2, zeta=1e7, r=0.1):
        """Laplace Learning
        ===================

        Semi-supervised learning via the solution of the Laplace equation
        \\[L u_j = 0, \\ \\ j \\geq m+1,\\]
        subject to \\(u_j = y_j\\) for \\(j=1,\\dots,m\\), where \\(L=D-W\\) is the
        combinatorial graph Laplacian and \\(y_j\\) for \\(j=1,\\dots,m\\) are the
        label vectors.

        The original method was introduced in [1]. This class also implements reweighting
        schemes `poisson` proposed in [2], `wnll` proposed in [3], and `properly`, proposed in [4].
        If `properly` is selected, the user must additionally provide the data features `X`.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        tau : float (optional), default=0.0
            Diagonal perturbation for controlling "model confidence"
        tol : float (optional), default=1e-5
            Tolerance for conjugate gradient solver.
        alpha : float (optional), default=2
            Parameter for `properly` reweighting.
        zeta : float (optional), default=1e7
            Parameter for `properly` reweighting.
        r : float (optional), default=0.1
            Radius for `properly` reweighting.

        References
        ---------
        [1] X. Zhu, Z. Ghahramani, and J. D. Lafferty. [Semi-supervised learning using gaussian fields
        and harmonic functions.](https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf) Proceedings
        of the 20th International Conference on Machine Learning (ICML-03), 2003.

        [2] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html),
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.

        [3] J. Calder, D. Slepčev. [Properly-weighted graph Laplacian for semi-supervised learning.](https://link.springer.com/article/10.1007/s00245-019-09637-3) Applied mathematics & optimization (2019): 1-49.
        """
        super().__init__(W, class_priors)

        self.normalization = normalization
        self.tau = tau
        self.tol = tol

        #Setup accuracy filename
        fname = '_poisson_rw_laplace'
        self.name = 'Poisson Reweighted Laplace Learning'
        if self.tau != 0.0:
            fname += '_' + str(int(self.tau*1000))
        self.accuracy_filename = fname


    def _fit(self, train_ind, train_labels, all_labels=None):

        #Reweighting -- including tau
        w = get_poisson_weighting(self.graph, train_ind, tau=self.tau)
        D = sparse.spdiags(w, 0, w.size, w.size)
        G = gl.graph(D * self.graph.weight_matrix * D)


        #Get some attributes
        n = G.num_nodes
        unique_labels = np.unique(train_labels)
        k = len(unique_labels)

        #Graph Laplacian and one-hot labels
        L = G.laplacian(normalization=self.normalization)
        F = gl.utils.labels_to_onehot(train_labels)

        #Locations of unlabeled points
        idx = np.full((n,), True, dtype=bool)
        idx[train_ind] = False

        #Right hand side
        b = -L[:,train_ind]*F
        b = b[idx,:]

        #Left hand side matrix
        A = L[idx,:]
        A = A[:,idx]

        #Preconditioner
        m = A.shape[0]
        M = A.diagonal()
        M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()

        #Conjugate gradient solver
        v = gl.utils.conjgrad(M*A*M, M*b, tol=self.tol)
        v = M*v

        #Add labels back into array
        u = np.zeros((n,k))
        u[idx,:] = v
        u[train_ind,:] = F

        return u

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
    parser.add_argument("--numtests", type=int, default=5)
    args = parser.parse_args()

    if args.tau == 0.0:
        tau = None
    else:
        tau = args.tau

    X, labels = gl.datasets.load(args.dataset.split("-")[0], metric=args.metric)

    if args.dataset.split("-")[-1] == 'evenodd':
        labels = labels % 2

    # Construct the similarity graph
    print(f"Constructing similarity graph for {args.dataset}")
    W = gl.weightmatrix.knn(X, 20)
    G = gl.graph(W)
    if not G.isconnected():
        print("Graph not connected with knn = 20, using knn = 30")
        W = gl.weightmatrix.knn(X, 30)
        print(f"\tGraph is connected = {G.isconnected()}")

    acc_models = {'poisson':gl.ssl.poisson(G),  # poisson learning
                'rwll0':gl.ssl.laplace(G, reweighting='poisson'),  # reweighted laplace learning, tau = 0
                'rwll001':poisson_rw_laplace(G, tau=0.001),  # reweighted laplace learning, tau = 0.001
                 'rwll01':poisson_rw_laplace(G, tau=0.01),  # reweighted laplace learning, tau = 0.01
                 'rwll1':poisson_rw_laplace(G, tau=0.1)}   # reweighted laplace learning, tau = 0.1

    ############################################
    ####### Can Change these variables #########
    ############################################

    acq_funcs_names = ['poisson_unc', 'rwll0_unc', 'rwll001_unc', 'rwll01_unc', 'rwll1_unc', 'random']
    acq_funcs = [unc, unc, unc, unc, random]
    models = [acc_models['poisson'], acc_models['rwll0'], acc_models['rwll001'], acc_models['rwll01'], acc_models['rwll001'], acc_models['poisson']]


    # Iterations for the different tests
    for it in range(args.numtests):
        seed = int(args.labelseed + 3*(it**2))
        labeled_ind = gl.trainsets.generate(labels, rate=1, seed=seed)

        RESULTS_DIR = os.path.join("results", f"{args.dataset}_results_{seed}_{args.iters}")
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

            acc_df = pd.DataFrame(acc)
            acc_df.to_csv(os.path.join(RESULTS_DIR, f"accs_{acq_func_name}.csv"), index=None)
            np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}.npy"), train_ind)
            return

        print("------Starting Active Learning Tests-------")
        with tqdm_joblib(tqdm(desc=f"{args.dataset} test {it+1}/{args.numtests}, seed = {seed}", total=len(models))) as progress_bar:
            Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, acq, mdl) for acq_name, acq,mdl in zip(acq_funcs_names, acq_funcs, models))

        # Consolidate results
        print(f"Consolidating results to {os.path.join(RESULTS_DIR, 'accs.csv')}...")
        accs_fnames = glob(os.path.join(RESULTS_DIR, "*.csv"))
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
    print(f"Saving overall results to {overall_results_file}")
    acc_files = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", "accs.csv"))
    dfs = [pd.read_csv(f) for f in acc_files]
    all_columns = {}
    for col in dfs[0].columns:
        vals = np.array([df[col].values for df in dfs])
        all_columns[col + " : avg"] = np.average(vals, axis=0)
        all_columns[col + " : std"] = np.std(vals, axis=0)

    all_df = pd.DataFrame(all_columns)
    all_df.to_csv(overall_results_file, index=None)
