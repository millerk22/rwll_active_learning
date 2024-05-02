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



if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for Graph Learning on Isolet")
    parser.add_argument("--dataset", type=str, default='isolet')
    parser.add_argument("--metric", type=str, default='raw')
    parser.add_argument("--numcores", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--resultsdir", type=str, default="results_isolet")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--cheatK", type=int, default=50)
    parser.add_argument("--knn", type=int, default=0)
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # Define ssl models and acquisition functions from configuration file
    ACQS_MODELS = [name for name in config["acqs_models"] if name.split(" ")[-1][:4] != "LAND"]
    acq_funcs_names = [name.split(" ")[0] for name in ACQS_MODELS]
    

    # load in graph and models that will be used in this run of tests
    model_names = [name.split(" ")[1] for name in ACQS_MODELS]
    models, labels, trainset, normalization, K = get_graph_and_models(acq_funcs_names, model_names, args)
    
    # if manually pass in K value in command line then overwrite value of K
    if args.K != 0:
        K = args.K
    else:
        K = np.unique(labels).size * 5
     
    
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
        labeled_ind = gl.trainsets.generate(labels, rate=1, seed=seed)[:2]
        # because of gl.ssl implementation, we need to sequentially change the corresponding values of labels to keep the given labels 0, 1, ..., k-1 to gl.ssl
        labels_map = {labels[idx] : i for i, idx in enumerate(labeled_ind)}
        
        # define the results directory for this seed's test
        RESULTS_DIR = os.path.join(args.resultsdir, f"{args.dataset}_results_{seed}_{args.iters}")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        np.save(os.path.join(RESULTS_DIR, "init_labeled.npy"), labeled_ind) # save initially labeled points that are common to each test


        def active_learning_test(acq_func_name, model_name, model):
            '''
            Active learning test definition for parallelization.
            '''
            
            # check if test already completed previously
            choices_run_savename = os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy")
            if os.path.exists(choices_run_savename):
                print(f"Found choices for {acq_func_name} in {model_name}")
                return
            
            # if need to decay tau, calculate mu from epsilon and 2K. K = # of clusters.
            if "decaytau" in acq_func_name in acq_func_name:
                eps = 1e-9
                mu = (eps / model.tau)**(.5/K)
            
            # fetch active_learning object
            labels_test = np.array([labels_map[labels[idx]] for idx in labeled_ind])
            AL = get_active_learner(acq_func_name, model, labeled_ind, labels_test, normalization, args)
            # If have a proportional sampling acquisition function then set K accordingly
            if "prop" in acq_func_name:
                AL.acq_function.set_K(K)


            # restrict candidate set to non-outliers, as determined by a KDE estimator
            if trainset is None:
                candidate_ind_all = np.arange(model.graph.num_nodes)
            else:
                candidate_ind_all = trainset.copy()
                
            if acq_func_name[-3:] == 'kde':
                knn_ind, knn_dist = gl.weightmatrix.load_knn_data(args.dataset.split("-")[0], metric=args.metric)
                d = np.max(knn_dist,axis=1)
                kde = (d/d.max())**(-1)
                outlier_inds = np.where(kde < np.percentile(kde, 10))[0] # throw out 10% of "outliers"
                candidate_ind_all = np.setdiff1d(candidate_ind_all, outlier_inds)
                print(f"Set candidate_ind for active learner of {acq_func_name} to throw out outliers")
            
            print(f"{acq_func_name}, training_set size = {candidate_ind_all.size}, dataset size = {model.graph.num_nodes}")

            # Calculate initial accuracy
            u_sub = AL.u
            u = np.zeros((model.graph.num_nodes, np.unique(labels).size))
            sorted_keys = np.array([t[0] for t in sorted(labels_map.items(), key=lambda a: a[1])]) 
            u[:,sorted_keys] = u_sub
            acc = np.array([gl.ssl.ssl_accuracy(np.argmax(u, axis=1), labels, AL.labeled_ind)])
            
            
            # Perform active learning iterations
            for j in tqdm(range(args.iters), desc=f"{args.dataset}, {acq_func_name} test {it+1}/{len(seeds)}, seed = {seed}"):
                query_point = AL.select_queries(candidate_ind=np.setdiff1d(candidate_ind_all, AL.labeled_ind))[0] # return this iteration's newly chosen points
                # have to handle to computation of gl.ssl differently because in Isolet we don't have a
                max_label_seen_so_far = max(AL.labels)
                if not labels[query_point] in labels_map:
                    labels_map[labels[query_point]] = max_label_seen_so_far + 1
                query_label = labels_map[labels[query_point]]
                AL.update([query_point], [query_label]) # update the active_learning object's labeled set
                
                # if need to decay tau in the model, then do so before updating the model
                if "decaytau" in acq_func_name:
                    if model.tau[0] != 0:
                        model.tau = mu*np.copy(model.tau)
                        if model.tau[0] < eps:
                            model.tau = np.zeros_like(model.tau)
                    
                # model update -- special for Isolet
                u_sub = AL.u
                u = np.zeros((model.graph.num_nodes, np.unique(labels).size))
                sorted_keys = np.array([t[0] for t in sorted(labels_map.items(), key=lambda a: a[1])]) 
                u[:,sorted_keys] = u_sub
                acc = np.append(acc, gl.ssl.ssl_accuracy(np.argmax(u, axis=1), labels, AL.labeled_ind))
                
            acc_dir = os.path.join(RESULTS_DIR, model_name)
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)
            np.save(os.path.join(acc_dir, f"acc_{acq_func_name}_{model_name}.npy"), acc)
            np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy"), AL.labeled_ind)
            return

        print("------Starting Active Learning Tests-------")

        Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, mdlname, mdl) for acq_name, mdlname, mdl \
                in zip(acq_funcs_names, model_names, models))