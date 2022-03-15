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
from acquisitions import ACQS


from joblib import Parallel, delayed






if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for RW Laplace Learning")
    parser.add_argument("--dataset", type=str, default='mstar-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=9)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--labelseed", type=int, default=3)
    parser.add_argument("--numeigs", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load in the graph and labels
    G, labels, trainset = load_graph(args.dataset, args.metric, args.numeigs)

    # Define ssl models and acquisition functions from configuration file
    ACQS_MODELS = config["acqs_models"]
    acq_funcs_names = [name.split(" ")[0] for name in ACQS_MODELS]
    acq_funcs = [ACQS[name] for name in acq_funcs_names]
    model_names = [name.split(" ")[1] for name in ACQS_MODELS]
    models = get_models(G, model_names)

    # use only enough cores as length of models
    if args.numcores > len(models):
        args.numcores = len(models)

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
        RESULTS_DIR = os.path.join("results", f"{args.dataset}_results_{seed}_{args.iters}")
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        np.save(os.path.join(RESULTS_DIR, "init_labeled.npy"), labeled_ind) # save initially labeled points that are common to each test


        def active_learning_test(acq_func_name, acq_func, model_name, model, show_tqdm):
            '''
            Active learning test definition for parallelization.
            '''

            # check if test already completed previously
            choices_run_savename = os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy")
            if os.path.exists(choices_run_savename):
                print(f"Found choices for {acq_func_name} in {model_name}")
                return

            # based on the acquisition function name, determine if need to compute the covariance matrix for instantiating
            # active_learner object
            if acq_func_name in ["mc", "mcvopt", "vopt"]:
                active_learner = gl.active_learning.active_learning(deepcopy(G), labeled_ind.copy(), labels[labeled_ind], \
                                eval_cutoff=args.numeigs, gamma=args.gamma)
            else:
                active_learner = gl.active_learning.active_learning(deepcopy(G), labeled_ind.copy(), labels[labeled_ind], eval_cutoff=None)

            # Calculate initial accuracy
            u = model.fit(active_learner.current_labeled_set, active_learner.current_labels)
            acc = np.array([gl.ssl.ssl_accuracy(model.predict(), labels, active_learner.current_labeled_set.size)])

            # define iterator object that can have tqdm output
            if show_tqdm:
                iterator_object = tqdm(range(args.iters), desc=f"{args.dataset} test {it+1}/{len(seeds)}, seed = {seed}")
            else:
                iterator_object = range(args.iters)

            for j in iterator_object:
                # should handle oracle update inside object
                query_inds = active_learner.select_query_points(acq_func, u)
                active_learner.update_labeled_data(query_inds, labels[query_inds])

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
        # show_bools is for tqdm iterator to track progress of one of the acquisition functions
        show_bools = np.zeros(len(models), dtype=bool)
        show_bools[::args.numcores] = True

        Parallel(n_jobs=args.numcores)(delayed(active_learning_test)(acq_name, acq, mdlname, mdl, show) for acq_name, acq, mdlname, mdl, show \
                in zip(acq_funcs_names, acq_funcs, model_names, models, show_bools))
