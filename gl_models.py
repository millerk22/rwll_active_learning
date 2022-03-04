import numpy as np
import graphlearning as gl
import scipy.sparse as sparse


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
