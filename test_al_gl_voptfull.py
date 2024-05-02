import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
import yaml
from copy import deepcopy
from glob import glob
from scipy.special import softmax
from functools import reduce
from utils import *


from joblib import Parallel, delayed


def solve_vopt_subset(L, train_ind, candidate_set, sopt=False):
    n = L.shape[0]

    #Locations of unlabeled points
    idx = np.full((n,), True, dtype=bool)
    idx[train_ind] = False
    
    # right hand side
    b = np.zeros((n, candidate_set.size))
    b[candidate_set,:] = np.eye(candidate_set.size)
    b_ = b[idx,:] # remove labeled rows
    
    #Left hand side matrix
    A = L[idx,:]
    A = A[:,idx]

    #Preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()

    #Conjugate gradient solver
    v = gl.utils.conjgrad(M*A*M, M*b_, tol=1e-5)
    b[idx,:] = M*v
    
    # calculate the maximum column norm of these
    if not sopt:
        vopt_vals = np.linalg.norm(b, axis=0) / np.sqrt(b[candidate_set,:].diagonal())
    else:
        vopt_vals = np.sum(b, axis=0) / np.sqrt(b[candidate_set,:].diagonal())
    
    
    return np.array([candidate_set[np.argmax(vopt_vals)]])
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for Graph Learning performing VOpt/SOpt full on subset.")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=5)
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--sopt", type=int, default=0)
    parser.add_argument("--knn", type=int, default=0)
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    
    # load in graph and models that will be used in this run of tests
    model_names = ["rwll0010"]
    acq_funcs_names = ["voptfull"] 
    if args.sopt:
        acq_funcs_names = ["soptfull"]
        
    models, labels, trainset, normalization, K = get_graph_and_models(acq_funcs_names, model_names, args)

    if trainset is not None:
        print("test_al_gl_voptfull.py not implemented to handle a trainset isn't the full dataset")

    # define the seed set for the iterations. Allows for defining in the configuration file
    try:
        seeds = config["seeds"]
    except:
        seeds = [0]
        print(f"Did not find 'seeds' in config file, defaulting to : {seeds}")
        
    # use only enough cores as length of seeds
    if args.numcores > len(seeds):
        args.numcores = len(seeds)
        
        
    model_name = model_names[0]
    model = models[0]
    
    def active_learning_test(s, sopt_flag=False):
        '''
        Active learning test definition for parallelization.
        '''
        v_or_s = "v" 
        if sopt_flag: 
            v_or_s = "s"
        
        
        RESULTS_DIR = os.path.join(args.resultsdir, f"{args.dataset}_results_{s}_{args.iters}")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            # get initially labeled indices, based on the given seed
            labeled_ind = gl.trainsets.generate(labels, rate=1, seed=s)
            np.save(os.path.join(RESULTS_DIR, "init_labeled.npy"), labeled_ind) # save initially labeled points that are common to each test
            
        else: # if other tests already exist, use their labeled_ind
            labeled_ind = np.load(os.path.join(RESULTS_DIR, "init_labeled.npy"))
        
        # check if test already completed previously
        choices_run_savename = os.path.join(RESULTS_DIR, f"choices_{v_or_s}optfull_{model_name}.npy")
        if os.path.exists(choices_run_savename):
            print(f"Found choices for {v_or_s}optfull in {model_name}")
            return
            
        # Calculate initial accuracy
        current_inds = labeled_ind.copy()
        ninit = labeled_ind.size
        current_labels = labels[current_inds]
        u = model.fit(current_inds, current_labels)
        acc = np.array([gl.ssl.ssl_accuracy(model.predict(), labels, current_inds)])


        # Perform active learning iterations
        for j in tqdm(range(ninit-1, args.iters), desc=f"{args.dataset}, {v_or_s}optfull test, seed = {s}"):
            # take random sample 
            unlabeled_inds = np.delete(np.arange(model.graph.num_nodes), current_inds)
            candidate_set = np.random.choice(unlabeled_inds, 500, replace=False)
            query_inds = solve_vopt_subset(model.graph.laplacian(), current_inds, candidate_set, sopt=sopt_flag)
            current_inds = np.append(current_inds, query_inds)
            current_labels = np.append(current_labels, labels[query_inds])


            # model update
            u = model.fit(current_inds, current_labels)
            acc = np.append(acc, gl.ssl.ssl_accuracy(model.predict(), labels, current_inds))

        acc_dir = os.path.join(RESULTS_DIR, model_name)
        if not os.path.exists(acc_dir):
            os.makedirs(acc_dir)
        np.save(os.path.join(acc_dir, f"acc_{v_or_s}optfull_{model_name}.npy"), acc)
        np.save(os.path.join(RESULTS_DIR, f"choices_{v_or_s}optfull_{model_name}.npy"), current_inds)
        return

    print("------Starting Active Learning Tests-------")
    Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(seed) for seed in seeds)
    
    if args.sopt:
        Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(seed, sopt_flag=True) for seed in seeds)
