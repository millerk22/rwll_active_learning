import numpy as np
import graphlearning as gl
import scipy.sparse as sparse
from copy import deepcopy



def get_models(G, model_names):
    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
               'laplace':gl.ssl.laplace(G), # laplace learning
               'mbo':gl.ssl.multiclass_mbo(G),
               'rwll0':gl.ssl.laplace(G, reweighting='poisson'), # reweighted laplace
               'rwll001':poisson_rw_laplace(G, tau=0.001),
               'rwll01':poisson_rw_laplace(G, tau=0.01),
              'rwll1':poisson_rw_laplace(G, tau=0.1),
             'beta0':beta_learning(G),
               'beta001':beta_learning(G, tau=0.001),
               'beta01':beta_learning(G, tau=0.01),
              'beta1':beta_learning(G, tau=0.1),
              'rwlldecay01':poisson_rw_laplace_decay(G, tau=0.01, tau_ll=0.01),
              'rwlldecayzero01': poisson_rw_laplace_decay(G, tau=0.0, tau_ll=0.01),
              'rwlldecay1':poisson_rw_laplace_decay(G, tau=0.1, tau_ll=0.1),
              'rwlldecayzero1': poisson_rw_laplace_decay(G, tau=0.0, tau_ll=0.1)}

    return [deepcopy(MODELS[name]) for name in model_names]

def get_poisson_weighting(G, train_ind, tau=0.0):
    n = G.num_nodes
    F = np.zeros(n)
    F[train_ind] = 1
    F -= np.mean(F)


    L = G.laplacian()
    if tau > 0.0:
        L += tau*sparse.eye(L.shape[0])

    w = gl.utils.conjgrad(L, F, tol=1e-5)
    w -= np.min(w, axis=0)

    return w


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



class beta_learning(gl.ssl.ssl):
    def __init__(self, W=None, class_priors=None, propagation='poisson', tau=0.0, eval_cutoff=50):
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
        self.train_ind = np.array([])

        if self.propagation == 'spectral':
            self.evals, self.evecs = G.eigen_decomp(normalization='combinatorial', k=eval_cutoff, method='lowrank', q=150, c=50)

        #Setup accuracy filename
        fname = '_beta'
        self.name = f'Beta Learning tau = {self.tau}'

        self.accuracy_filename = fname


    def _fit(self, train_ind, train_labels, all_labels=None):
        # Not currently designed for repeated indices in train_ind
        if train_ind.size >= self.train_ind.size:
            mask = ~np.isin(train_ind, self.train_ind)
            prop_ind = train_ind[np.where(mask)[0]]
            prop_labels = train_labels[np.where(mask)[0]]
        else: # if give fewer training labels than before, we assume that this is a "new" instantiation
            prop_ind, prop_labels = train_ind, train_labels
            mask = np.ones(3, dtype=bool)
        self.train_ind = train_ind

        # Poisson propagation
        if self.propagation == "poisson":
            n, num_prop, nc = self.graph.num_nodes, prop_ind.size, np.unique(train_labels).size
            F = np.zeros((n, num_prop))
            F[prop_ind,:] = np.eye(num_prop)
            F -= np.mean(F, axis=0)

            L = self.graph.laplacian()
            if self.tau  > 0.0:
                L += self.tau*sparse.eye(L.shape[0])

            P = gl.utils.conjgrad(L, F, tol=1e-5)
            P -= np.min(P, axis=0)

            P /= P[prop_ind,np.arange(F.shape[1])][np.newaxis,:] # scale by the value at the point sources

            if mask.all(): # prop_ind == train_ind, so all inds are "new"
                self.A = np.ones((n, nc))  # Dir(1,1,1,...,1) prior on each node

            # Add propagations according to class for the propagation inds (prop_inds)
            for c in np.unique(prop_labels):
                self.A[:, c] += np.sum(P[:,np.where(prop_labels == c)[0]], axis=1) # sum propagations together according to class


        if self.propagation == "spectral":
            assert np.isin(np.unique(train_labels), np.array([0,1])).all()
            c0_ind, c1_ind = train_ind[train_labels == 0], train_ind[train_labels == 1]
            alpha, beta = prop_alpha_beta_thresh(self.evecs, self.evals, c0_ind, c1_ind, .1, thresh=1e-9)
            self.A = np.hstack((beta[:,np.newaxis], alpha[:,np.newaxis])) + 1. # include the Beta(1,1) prior

        u = self.A / (self.A.sum(axis=1)[:,np.newaxis]) # mean estimator
        return u








class poisson_rw_laplace_decay(gl.ssl.ssl):
    def __init__(self, W=None, class_priors=None, tau=0.0, tau_ll=0.0, normalization='combinatorial', tol=1e-5, alpha=2, zeta=1e7, r=0.1):
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
        self.tau = tau       # tau for poisson reweighting
        self.tau_ll = tau_ll # tau for LL solve
        self.tol = tol

        #Setup accuracy filename
        fname = '_poisson_rw_laplace_decay'
        self.name = 'Poisson Reweighted Laplace Learning with Decay'
        if self.tau != 0.0:
            fname += '_' + str(int(self.tau*1000)) + '_' + str(int(self.tau_ll*1000))
        self.accuracy_filename = fname


    def _fit(self, train_ind, train_labels, all_labels=None):

        #Reweighting -- including tau
        w = get_poisson_weighting(self.graph, train_ind, tau=self.tau) # tau not in the weighting, but in the solve after
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
        A += self.tau_ll*sparse.eye(L.shape[0]) # add the tau diagonal for Laplace Learning

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
