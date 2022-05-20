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
from sklearn.metrics import cohen_kappa_score
from utils import *
from acquisitions import ACQS

from joblib import Parallel, delayed



if __name__ == "__main__":
    parser = ArgumentParser(description="Compute Accuracies in Parallel of Active Learning Tests for RWLL Learning")
    parser.add_argument("--dataset", type=str, default='mstar-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=9)
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    G, labels, trainset, normalization = load_graph(args.dataset, args.metric, numeigs=None) # don't compute any eigenvalues
    model_names = config["acc_models"]
    models = get_models(G, model_names)
    models_dict = {name:model for name, model in zip(model_names, models)}
    results_directories = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}/"))
    acqs_models = config["acqs_models"]


    for out_num, RESULTS_DIR in enumerate(results_directories):
        choices_fnames = glob(os.path.join(RESULTS_DIR, "choices_*.npy"))
        choices_fnames = [fname for fname in choices_fnames if " ".join(fname.split("_")[-2:]).split(".")[0] in acqs_models ]

        labeled_ind = np.load(os.path.join(RESULTS_DIR, "init_labeled.npy")) # initially labeled points that are common to all acq_func:gbssl modelname pairs
        for num, acc_model_name in enumerate(models_dict.keys()):
            acc_dir = os.path.join(RESULTS_DIR, acc_model_name)
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)

            def compute_metrics(choices_fname, show_tqdm):
                # get acquisition function - gbssl modelname that made this sequence of choices
                acq_func_name, modelname = choices_fname.split("_")[-2:]
                modelname = modelname.split(".")[0]

                # load in the indices of the choices
                choices = np.load(choices_fname)

                # define the filepath of where the results of evaluating this acq_func:modelname combo had in acc_model_name
                metric_fname = os.path.join(acc_dir, f"metrics_{acq_func_name}_{modelname}.npy")
                if os.path.exists(acc_fname):
                    print(f"Already computed metrics in {acc_model_name} for {acc_fname}")
                    return
                else:
                    print(f"Computing metrics in {acc_model_name} for {acq_func_name} in {modelname}")

                # get copy of model on this cpu
                model = deepcopy(models_dict[acc_model_name])

                # define iterator to have same number of accuracy evaluations as those already done
                if show_tqdm:
                    iterator_object = tqdm(range(labeled_ind.size,choices.size+1), desc=f"Computing Metrics of {acq_func_name}-{modelname}")
                else:
                    iterator_object = range(labeled_ind.size,choices.size+1)

                # Compute accuracies at each sequential subset of choices
                acc, avg_acc, kappa = np.array([]), np.array([]), np.array([])
                for j in iterator_object:
                    train_ind = choices[:j]
                    u = model.fit(train_ind, labels[train_ind])
                    u_thresh = model.predict()
                    acc = np.append(acc, gl.ssl.ssl_accuracy(u_thresh, labels, train_ind.size))
                    avg_acc_j = np.array([gl.ssl.ssl_accuracy(u_thresh[labels==c], labels[labels==c], (train_ind == c).sum()) for c in range(nc)])
                    avg_acc = np.append(avg_acc, np.average(avg_acc_j))
                    kappa = np.append(kappa, cohen_kappa_score(u_thresh, labels))

                # save metric results to corresponding filename
                np.save(metric_fname, np.vstack((acc, avg_acc, kappa)))
                return

            print(f"-------- Computing Metrics in {acc_model_name}, {num+1}/{len(models_dict)} in {RESULTS_DIR} ({out_num+1}/{len(results_directories)}) -------")
            # show_bools is for tqdm iterator to track progress of some
            show_bools = np.zeros(len(choices_fnames), dtype=bool)
            show_bools[::args.numcores] = True

            Parallel(n_jobs=args.numcores)(delayed(compute_metrics)(choices_fname, show) for choices_fname, show \
                    in zip(choices_fnames, show_bools))
            print()

        # Consolidate results
        print(f"Consolidating accuracy results of run in: {os.path.join(RESULTS_DIR)}...")
        for acc_model_name in models_dict.keys():
            acc_dir = os.path.join(RESULTS_DIR, acc_model_name)
            metrics_fnames = glob(os.path.join(acc_dir, "metrics_*.npy"))
            columns = {}
            for fname in metrics_fnames:
                metrics = np.load(fname)
                acq_func_name, modelname = fname.split("_")[-2:]
                modelname = modelname.split(".")[0]
                for i, met_name in enumerate(['OA', 'AA', 'kappa']):
                    columns[acq_func_name + " : " + modelname + " : " + met_name] = metrics[i,:]

            metrics_df = pd.DataFrame(columns)
            metrics_df.to_csv(os.path.join(acc_dir, "metrics.csv"), index=None)

        print("-"*40)
        print("-"*40)
    print()


    # Get average and std curves over all tests
    overall_results_dir = os.path.join("results", f"{args.dataset}_overall_{args.iters}")
    if not os.path.exists(overall_results_dir):
        os.makedirs(overall_results_dir)

    results_models_directories = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", "*/"))
    acc_model_names_list = np.unique([fpath.split("/")[-2] for fpath in results_models_directories])
    for acc_model_name in tqdm(acc_model_names_list, desc=f"Saving results over all runs to: {overall_results_dir}", total=len(acc_model_names_list)):
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats_metrics.csv")
        metric_files = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", f"{acc_model_name}", "metrics.csv"))
        dfs = [pd.read_csv(f) for f in sorted(metric_files)]
        if len(dfs) == 0:
            continue
        possible_columns = reduce(np.union1d, [df.columns for df in dfs])
        all_columns = {}
        for col in possible_columns:
            vals = np.array([df[col].values for df in dfs if col in df.columns])
            all_columns[col + " : avg"] = np.average(vals, axis=0)
            all_columns[col + " : std"] = np.std(vals, axis=0)

        all_df = pd.DataFrame(all_columns)
        all_df.to_csv(overall_results_file, index=None)
    print("-"*40)
