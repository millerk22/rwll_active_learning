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
from acquisitions import ACQS


from joblib import Parallel, delayed



def get_active_learner(acq_func_name, G, labels, labeled_ind, normalization, args):
    """
        Based on the acquisition function name, determine if need to compute the covariance matrix for instantiating 
        active_learner object
    """
    if acq_func_name.split("-")[0] in ["mc", "mcvopt", "vopt", "vopt1", "sopt"]:
        print(f"gamma = {args.gamma}")
        active_learner = get_active_learner_eig(deepcopy(G), labeled_ind.copy(), labels, acq_func_name, 
                                                gamma=args.gamma, normalization=normalization)

    else:
        active_learner = gl.active_learning.active_learning(deepcopy(G), labeled_ind.copy(), 
                                                            labels[labeled_ind], eval_cutoff=None)
        
        # special case of needing to calculate FULL covariance matrix for voptfull
        if acq_func_name == "voptfull":
            print("Calculating fullC")
            active_learner.fullC = sparse.linalg.inv(sparse.csc_matrix(G.laplacian() + 
                                                                       0.001*sparse.eye(G.num_nodes))).toarray()
    
    return active_learner


def get_graph_and_models(acq_funcs_names, model_names, args):
    # Determine if we need to calculate more eigenvectors/values for mc, vopt, mcvopt acquisitions
    maxnumeigs = 0
    for acq_func_name in acq_funcs_names:
#         if acq_func_name in ["mc", "mcvopt", "vopt", "vopt1"]:
#             maxnumeigs = max(maxnumeigs, 50)
            
        if len(acq_func_name.split("-")) == 1:
            continue
        d = acq_func_name.split("-")[-1]
        if len(d) > 0:
            if maxnumeigs < int(d):
                maxnumeigs = int(d)
    if maxnumeigs == 0:
        maxnumeigs = None

    # Load in the graph and labels
    print("Loading in Graph...")
    G, labels, trainset, normalization, K = load_graph(args.dataset, args.metric, maxnumeigs, returnK=True)
    models = get_models(G, model_names)
    
    return G, labels, trainset, normalization, models, K


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for Graph Learning")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=9)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--cheatK", type=int, default=50)
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # Define ssl models and acquisition functions from configuration file
    ACQS_MODELS = [name for name in config["acqs_models"] if (name.split(" ")[-1][:3] != "gcn" and name.split(" ")[-1][:4] != "LAND")]
    acq_funcs_names = [name.split(" ")[0] for name in ACQS_MODELS]
    acq_funcs = [ACQS[name.split("-")[0]] for name in acq_funcs_names]
    
    
    

    # load in graph and models that will be used in this run of tests
    model_names = [name.split(" ")[1] for name in ACQS_MODELS]
    G, labels, trainset, normalization, models, K = get_graph_and_models(acq_funcs_names, model_names, args)
    
    # if manually pass in K value in command line then overwrite value of K
    if args.K != 0:
        K = args.K
    
    # If have a proportional sampling acquisition function then set K accordingly
    for i, name in enumerate(acq_funcs_names):
        if "prop" in name:
            print(name)
            acq_funcs[i].set_K(K)
    
     
    
    # use only enough cores as length of models
    if args.numcores > len(models):
        args.numcores = len(models)


    # define the seed set for the iterations. Allows for defining in the configuration file
    try:
        seeds = config["seeds"]
    except:
        seeds = [0]
        print(f"Did not find 'seeds' in config file, defaulting to : {seeds}")

    # Iterations for the different tests
    for it, seed in enumerate(seeds):
        # get initially labeled indices, based on the given seed
        labeled_ind = gl.trainsets.generate(labels, rate=1, seed=seed)

        # define the results directory for this seed's test
        RESULTS_DIR = os.path.join(args.resultsdir, f"{args.dataset}_results_{seed}_{args.iters}")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        np.save(os.path.join(RESULTS_DIR, "init_labeled.npy"), labeled_ind) # save initially labeled points that are common to each test


        def active_learning_test(acq_func_name, acq_func, model_name, model):
            '''
            Active learning test definition for parallelization.
            '''

            # check if test already completed previously
            choices_run_savename = os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy")
            if os.path.exists(choices_run_savename):
                print(f"Found choices for {acq_func_name} in {model_name}")
                return
            
            # if need to decay tau, calculate mu from epsilon and 2K. K = # of clusters.
            if "decaytau" in acq_func_name or "switchK" in acq_func_name:
                eps = 1e-9
                mu = (eps / model.tau)**(.5/K)
            
            # fetch active_learning object
            active_learner = get_active_learner(acq_func_name, G, labels, labeled_ind, normalization, args)
            
            # restrict training set (and subsequently the candidate set) to non-outliers, as determined by a KDE estimator
            if acq_func_name[-3:] == 'kde':
                knn_ind, knn_dist = gl.weightmatrix.load_knn_data(args.dataset.split("-")[0], metric=args.metric)
                d = np.max(knn_dist,axis=1)
                kde = (d/d.max())**(-1)
                outlier_inds = np.where(kde < np.percentile(kde, 10))[0] # throw out 10% of "outliers"
                active_learner.training_set = np.setdiff1d(active_learner.training_set, outlier_inds)
                print(f"Set training_set for active learner of {acq_func_name} to throw out outliers")
            
            print(f"{acq_func_name}, training_set size = {active_learner.training_set.size}, dataset size = {G.num_nodes}")
                

            # Calculate initial accuracy
            u = model.fit(active_learner.current_labeled_set, active_learner.current_labels)
            acc = np.array([gl.ssl.ssl_accuracy(model.predict(), labels, active_learner.current_labeled_set.size)])
            
            
            # Perform active learning iterations
            for j in tqdm(range(args.iters), desc=f"{args.dataset}, {acq_func_name} test {it+1}/{len(seeds)}, seed = {seed}"):
                # should handle oracle update inside object
                query_inds = active_learner.select_query_points(acq_func, u)
                active_learner.update_labeled_data(query_inds, labels[query_inds])
                
                if acq_func_name == 'voptfull': 
                    # update of "full" covariance matrix not currently in gl.active_learning
                    for idx in query_inds:
                        active_learner.fullC -= np.outer(active_learner.fullC[:,idx], 
                                                             active_learner.fullC[:,idx])/active_learner.fullC[idx, idx] 

                
                # if need to decay tau in the model, then do so before updating the model
                if "decaytau" in acq_func_name:
                    if model.tau[0] != 0:
                        model.tau = mu*np.copy(model.tau)
                        if model.tau[0] < eps:
                            model.tau = np.zeros_like(model.tau)
                elif "switchK" in acq_func_name:
                    if "switchKcheat" in acq_func_name:
                        if j == args.cheatK:
                            model.tau = np.zeros_like(model.tau)
                    else:
                        if j == K:
                            model.tau = np.zeros_like(model.tau)
                        
                            
                    
                # model update
                u = model.fit(active_learner.current_labeled_set, active_learner.current_labels)
                acc = np.append(acc, gl.ssl.ssl_accuracy(model.predict(), labels, active_learner.current_labeled_set.size))

            acc_dir = os.path.join(RESULTS_DIR, model_name)
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)
            np.save(os.path.join(acc_dir, f"acc_{acq_func_name}_{model_name}.npy"), acc)
            np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy"), active_learner.current_labeled_set)
            return

        print("------Starting Active Learning Tests-------")

        Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, acq, mdlname, mdl) for acq_name, acq, mdlname, mdl \
                in zip(acq_funcs_names, acq_funcs, model_names, models))
