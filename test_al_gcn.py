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
from gl_models import get_models
from utils import *
from gcn_util import *
from acquisitions import *
import torch


from joblib import Parallel, delayed




if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for GCN")
    parser.add_argument("--dataset", type=str, default='mnist-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--labelseed", type=int, default=3)
    parser.add_argument("--config", type=str, default="./config_gcn.yaml")
    parser.add_argument("--resultsdir", type=str, default="results_gcn")
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load in the graph and labels
    print("Loading in Graph...")
    G, labels, trainset, normalization, X = load_graph(args.dataset, args.metric, numeigs=None, returnX=True)
    print("Done loading in Graph...")
    # prep structures for GCN
    i,j,v = sparse.find(G.weight_matrix)
    edge_weight= v
    edge_index = np.vstack((i,j))
    # Convert to torch
    x = torch.from_numpy(X).float()
    y = torch.from_numpy(labels).long()
    edge_index = torch.from_numpy(edge_index).long()
    
    
    
    dataset = Dataset(args.dataset, X.shape[1], np.unique(labels).size)
    
    ACQS_MODELS = config["acqs_models"]
    ACQS_MODELS = [thing for thing in ACQS_MODELS if thing.split(" ")[1][:3] == "gcn"]
    acq_funcs_names = [name.split(" ")[0] for name in ACQS_MODELS]
    acq_funcs = [ACQS[name.split("-")[0]] for name in acq_funcs_names]
    model_names = [name.split(" ")[1] for name in ACQS_MODELS]
    models = get_gcn_models(model_names, dataset)
    

    # define the seed set for the iterations. Allows for defining in the configuration file
    try:
        seeds = config["seeds"]
    except:
        seeds = int(args.labelseed + 3*(it**2))
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
            Active learning test definition.
            '''

            # check if test already completed previously
            choices_run_savename = os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy")
            if os.path.exists(choices_run_savename):
                print(f"Found choices for {acq_func_name} in {model_name}")
                return

            # based on the acquisition function name, determine if need to compute the covariance matrix for instantiating
            # active_learner object
            active_learner = gl.active_learning.active_learning(deepcopy(G), labeled_ind.copy(), labels[labeled_ind], eval_cutoff=None)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # if acq_func is static ('pagerank', 'degree') calculate up front
            # assign this as a member variable of active learner object for use by acquisition_function class
            if acq_func_name == 'pagerank':
                active_learner.static_values = G.page_rank() # default param settings alpha = 0.85
            elif acq_func_name == 'degree':
                active_learner.static_values = G.degree_vector()
            elif acq_func_name == 'age':
                static_values = G.page_rank()
                static_values = (static_values - static_values.min())/(static_values.max() - static_values.min())
                active_learner.static_values = np.asarray([perc(static_values,i) for i in range(len(static_values))]) # perc function located in acquisitions.py
            else:
                active_learner.static_values = None
            
            # masks for pytorch geometric data definition
            train_mask = np.zeros((G.weight_matrix.shape[0],),dtype=bool)
            train_mask[active_learner.current_labeled_set] = True
            val_mask = train_mask
            test_mask = np.ones((G.weight_matrix.shape[0],),dtype=bool)
            test_mask[active_learner.current_labeled_set] = False
            train_mask = torch.from_numpy(train_mask).bool()
            test_mask = torch.from_numpy(test_mask).bool()
            val_mask = torch.from_numpy(val_mask).bool()
            data = Data(x=x, y=y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, edge_index=edge_index,
                        edge_weight=edge_weight)
            
            # prep GCN training optimizer
            model, data = model.to(device), data.to(device)
            model.data = data
            
            param_list = []
            param_list.append(dict(params=model.convinitial.parameters(), weight_decay=5e-4))
            for i,l in enumerate(model.conv):
                param_list.append(dict(params=l.parameters(), weight_decay=0))
            param_list.append(dict(params=model.convfinal.parameters(), weight_decay=0))
            optimizer = torch.optim.Adam(param_list,lr=0.01)
            
            # Train initial GCN 100 epochs
            model = train(model, optimizer, data, num_epochs=100)
            
            # Calculate initial testing accuracy
            testing_acc, u = test(model, data)
            acc = np.array([testing_acc]) # testing accuracy 
            

            # define iterator object that can have tqdm output
            iterator_object = tqdm(range(args.iters), desc=f"{args.dataset} test of {acq_func_name} {it+1}/{len(seeds)}, seed = {seed}")
           
            for j in iterator_object:
                # should handle oracle update inside object
                query_inds = active_learner.select_query_points(acq_func, u)
                active_learner.update_labeled_data(query_inds, labels[query_inds])
                
                # masks for pytorch geometric data definition
                train_mask = np.zeros((G.weight_matrix.shape[0],),dtype=bool)
                train_mask[active_learner.current_labeled_set] = True
                val_mask = train_mask
                test_mask = np.ones((G.weight_matrix.shape[0],),dtype=bool)
                test_mask[active_learner.current_labeled_set] = False
                train_mask = torch.from_numpy(train_mask).bool()
                test_mask = torch.from_numpy(test_mask).bool()
                val_mask = torch.from_numpy(val_mask).bool()
                data = Data(x=x, y=y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, edge_index=edge_index,
                            edge_weight=edge_weight)
                data = data.to(device)
                model.data = data
                
                # model update
                model = train(model, optimizer, data, num_epochs=20) # hard coded 20 epochs of training with each additional labeled point
                testing_acc, u = test(model, data)
                acc = np.append(acc, testing_acc)

            acc_dir = os.path.join(RESULTS_DIR, model_name)
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)
            np.save(os.path.join(acc_dir, f"acc_{acq_func_name}_{model_name}.npy"), acc)
            np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy"), active_learner.current_labeled_set)
            return

        print("------Starting Active Learning Tests-------")
        # show_bools is for tqdm iterator to track progress of one of the acquisition functions
        for acq_name, acq, mdlname, mdl in zip(acq_funcs_names, acq_funcs, model_names, models):
            active_learning_test(acq_name, acq, mdlname, mdl)
