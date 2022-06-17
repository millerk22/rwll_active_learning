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
from acquisitions import ACQS
import torch



if __name__ == "__main__":
    parser = ArgumentParser(description="Compute Accuracies of Active Learning Tests for GCN")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=9)
    parser.add_argument("--config", type=str, default="./config_gcn.yaml")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--resultsdir", type=str, default="results_gcn")
    args = parser.parse_args()
    
    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load in the graph and labels
    print("Loading in Graph...")
    G, labels, trainset, normalization, X = load_graph(args.dataset, args.metric, numeigs=None, returnX=True)
    # prep structures for GCN
    i,j,v = sparse.find(G.weight_matrix)
    edge_weight= v
    edge_index = np.vstack((i,j))
    # Convert to torch
    x = torch.from_numpy(X).float()
    y = torch.from_numpy(labels).long()
    edge_index = torch.from_numpy(edge_index).long()
    
    
    
    dataset = Dataset(args.dataset, X.shape[1], np.unique(labels).size)
    
    model_names = [name for name in config["acc_models"] if name[:3] == "gcn"]
    models = get_gcn_models(model_names, dataset)
    models_dict = {name:model for name, model in zip(model_names, models)}
    results_directories = glob(os.path.join(args.resultsdir, f"{args.dataset}_results_*_{args.iters}/"))
    acqs_models = config["acqs_models"]
    
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    for out_num, RESULTS_DIR in enumerate(results_directories):
        choices_fnames = glob(os.path.join(RESULTS_DIR, "choices_*.npy"))
        choices_fnames = [fname for fname in choices_fnames if " ".join(fname.split("_")[-2:]).split(".")[0] in acqs_models ]

        labeled_ind = np.load(os.path.join(RESULTS_DIR, "init_labeled.npy")) # initially labeled points that are common to all acq_func:gbssl modelname pairs
        for num, acc_model_name in enumerate(models_dict.keys()):
            acc_dir = os.path.join(RESULTS_DIR, acc_model_name)
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)
            
            
            

            def compute_accuracies(choices_fname):
                # get acquisition function - gbssl modelname that made this sequence of choices
                acq_func_name, modelname = choices_fname.split("_")[-2:]
                modelname = modelname.split(".")[0]

                # load in the indices of the choices
                choices = np.load(choices_fname)

                # define the filepath of where the results of evaluating this acq_func:modelname combo had in acc_model_name
                acc_fname = os.path.join(acc_dir, f"acc_{acq_func_name}_{modelname}.npy")
                if os.path.exists(acc_fname):
                    print(f"Already computed accuracies in {acc_model_name} for {acc_fname}")
                    return
                else:
                    print(f"Computing accuracies in {acc_model_name} for {acq_func_name} in {modelname}")

                # masks for pytorch geometric data definition
                train_mask = np.zeros((G.weight_matrix.shape[0],),dtype=bool)
                train_mask[labeled_ind] = True
                val_mask = train_mask
                test_mask = np.ones((G.weight_matrix.shape[0],),dtype=bool)
                test_mask[labeled_ind] = False
                train_mask = torch.from_numpy(train_mask).bool()
                test_mask = torch.from_numpy(test_mask).bool()
                val_mask = torch.from_numpy(val_mask).bool()
                data = Data(x=x, y=y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, edge_index=edge_index,
                            edge_weight=edge_weight)
                
                
                model = deepcopy(models_dict[acc_model_name]) 
                

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

                # define iterator to have same number of accuracy evaluations as those already done
                iterator_object = tqdm(range(labeled_ind.size,choices.size+1), desc=f"Computing Acc of {acq_func_name}-{modelname}")

                # Compute accuracies at each sequential subset of choices
                acc = np.array([])
                for j in iterator_object:
                    train_ind = choices[:j]
                    
                    # masks for pytorch geometric data definition
                    train_mask = np.zeros((G.weight_matrix.shape[0],),dtype=bool)
                    train_mask[train_ind] = True
                    val_mask = train_mask
                    test_mask = np.ones((G.weight_matrix.shape[0],),dtype=bool)
                    test_mask[train_ind] = False
                    train_mask = torch.from_numpy(train_mask).bool()
                    test_mask = torch.from_numpy(test_mask).bool()
                    val_mask = torch.from_numpy(val_mask).bool()
                    data = Data(x=x, y=y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, edge_index=edge_index,
                                edge_weight=edge_weight)
                    data = data.to(device)
                    model.data = data

                    # model update
                    model = train(model, optimizer, data, num_epochs=20) # hard-coded 20 epochs of training with each additional labeled point
                    testing_acc, _ = test(model, data)
                    acc = np.append(acc, testing_acc)

                # save accuracy results to corresponding filename
                np.save(acc_fname, acc)
                return

            print(f"-------- Computing Accuracies in {acc_model_name}, {num+1}/{len(models_dict)} in {RESULTS_DIR} ({out_num+1}/{len(results_directories)}) -------")
            
            
            for choice_fname in choices_fnames:
                compute_accuracies(choice_fname)
            print()

        # Consolidate results
        print(f"Consolidating accuracy results of run in: {os.path.join(RESULTS_DIR)}...")
        for acc_model_name in models_dict.keys():
            acc_dir = os.path.join(RESULTS_DIR, acc_model_name)
            accs_fnames = glob(os.path.join(acc_dir, "acc_*.npy"))
            columns = {}
            for fname in accs_fnames:
                acc = np.load(fname)
                acq_func_name, modelname = fname.split("_")[-2:]
                modelname = modelname.split(".")[0]
                columns[acq_func_name + " : " + modelname] = acc

            acc_df = pd.DataFrame(columns)
            acc_df.to_csv(os.path.join(acc_dir, "accs.csv"), index=None)

        print("-"*40)
        print("-"*40)
    print()


    
