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
import networkx as nx
from heapq import *
from itertools import product


from joblib import Parallel, delayed


def MSSP(Graph, class_dict, shortest_paths=None, cut_edges=None):
    if shortest_paths is None:
        shortest_paths = []
        heapify(shortest_paths)
    
    if cut_edges is None: # initial computation of shortest_paths
        for pair in product(class_dict[0], class_dict[1]):
            try: # compute path, and if exists path then add to heapq
                path = nx.dijkstra_path(Graph, source=pair[0], target=pair[1])
                heappush(shortest_paths, (len(path), pair, path))
            except: 
                pass
    else: # update case of shortest_paths
        num_sp = len(shortest_paths)
        
        new_shortest_paths = []
        heapify(new_shortest_paths)
        while shortest_paths:
            l, pair, path = heappop(shortest_paths)
            recompute = False
            # only need to recompute paths that pass through cut_edge
            for u, v in zip(path[:-1], path[1:]):
                if (u,v) in cut_edges: # only need to look at this way cuz of ordering 0-1
                    recompute = True
                    break
            
            if recompute:
                try:# compute path, and if exists path then add to heapq
                    path = nx.dijkstra_path(Graph, source=pair[0], target=pair[1])
                    heappush(new_shortest_paths, (len(path), pair, path))
                except:
                    pass
            else:
                heappush(new_shortest_paths, (l, pair, path))
        
        shortest_paths = new_shortest_paths
        del new_shortest_paths
        
    return shortest_paths

def run_S2(Graph, labeled_ind, labeled_labels, oracle, budget=100, max_iters=1e6, it_print=1e2, verbose=True):
    '''
    Inputs:
        * G = NetworkX graph
        * ...
        * oracle = numpy array of length n (# nodes in G) with ground truth labels of nodes
        * max_iters = max # of edges to cut
    
    Outputs:
        * G = modified graph with edges cut according to S^2 algorithm
        * labeled_ind
        * labeled_labels
        * shortest_paths heapq
    '''
    
    class_dict = {c:list(labeled_ind[labeled_labels == c]) for c in np.unique(labeled_labels)}
    assert len(list(class_dict.keys())) == 2
    
    # get shortest paths between each pair of labeled points with different labels
    shortest_paths = MSSP(Graph, class_dict)
    
    it = 0
    num_edges_cut = 0
    print("Commencing with S^2 (true) main loop...")
    while it < max_iters and shortest_paths:
        
        
        if verbose and (it + 1) % it_print == 0:
            print(f"Iter = {it+1}/{max_iters}, budget left = {budget}, size of shortest_paths heapq = {len(shortest_paths)}")
        
        cut_edges = []
        
        # get shortest path pair
        path_length, pair, path = heappop(shortest_paths)
        if path_length == np.inf:
            print(f"No more paths between oppositely labeled points!")
            break
        
        # base case --> cut edge in G, recompute shortest path between these two and add to heapq
        if path_length == 2:
            #try to remove edges that connect to these connected ones. By shortest-shortest path assumption, should suffice to just look here at base case.
            for s in class_dict[0]:
                try:
                    Graph.remove_edge(s, pair[1])
                    cut_edges.append((s, pair[1]))
                except:
                    pass
            for t in class_dict[1]:
                try:
                    Graph.remove_edge(pair[0], t)
                    cut_edges.append((pair[0], t))
                except:
                    pass
            
            # update corresponding paths
            try:
                new_path = nx.dijkstra_path(Graph, source=s, target=t)
                heappush(shortest_paths, (len(new_path), (s,t), new_path))
            except:
                pass

        # find midpoint of path, label point, and then compute shortest paths between the oppositely labeled points
        else:
            midpoint = path[path_length // 2]
            midpoint_label = oracle[midpoint]
            if midpoint not in labeled_ind:
                class_dict[midpoint_label].append(midpoint)
                budget -= 1
                labeled_ind = np.append(labeled_ind, midpoint)
                labeled_labels = np.append(labeled_labels, midpoint_label)
            
            if midpoint_label == 0:
                md_path = path[path_length//2:]
                s, t = midpoint, pair[1]
            else:
                md_path = path[:path_length//2 + 1]
                s, t = pair[0], midpoint
                
            # need to add back original path between pair[0] and pair[1] to shortest_paths for other computations
            heappush(shortest_paths, (path_length, pair, path))
            heappush(shortest_paths, (len(md_path), (s,t), md_path))
        
        # update increment and stop if budget reached
        it += 1
        if budget == 0:
            print(f"Budget reached, ending sampling!")
            break
        
        # update shortest_paths only if we cut an edge
        if cut_edges:
            shortest_paths = MSSP(Graph, class_dict, shortest_paths=shortest_paths, cut_edges=cut_edges)
            num_edges_cut += len(cut_edges)
        
        
    
    print(f"Done, # cut edges = {num_edges_cut}")
    
    return labeled_ind, labeled_labels, shortest_paths




if __name__ == "__main__":
    parser = ArgumentParser(description="Run Large Tests in Parallel of Active Learning Test for S^2 Algorithm")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=6)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--knn", type=int, default=0)
    args = parser.parse_args()

    # load in configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    af_names = [name.split(" ")[0] for name in config["acqs_models"]]
    if "s2" not in af_names:
        print("S2 acquisition function not requested in config file. Ending...")
    
    else:
        print("Loading in Graph...")
        G, labels, trainset, _, K = load_graph(args.dataset, args.metric, None, returnK=True, knn=args.knn)
    
        # convert graph to networkx, use just 0,1 edge weights
        A = G.adjacency().astype(int)
        A.setdiag(0)
        A.eliminate_zeros()
        G = nx.Graph(A)
        
        # if manually pass in K value in command line then overwrite value of K
        if args.K != 0:
            K = args.K
    
        # define the seed set for the iterations. Allows for defining in the configuration file
        try:
            seeds = config["seeds"]
        except:
            seeds = [0]
            print(f"Did not find 'seeds' in config file, defaulting to : {seeds}")
    
        # use only enough cores as length of models
        if args.numcores > len(seeds):
            args.numcores = len(seeds)
        
        # copy the graph since this will be the "model"
        Graphs = len(seeds)*[G.copy()]

        
        
        def s2_test(seed, Graph):
            # get initially labeled indices, based on the given seed
            labeled_ind = gl.trainsets.generate(labels, rate=1, seed=seed)
            num_init = labeled_ind.size

            # define the results directory for this seed's test
            RESULTS_DIR = os.path.join(args.resultsdir, f"{args.dataset}_results_{seed}_{args.iters}")
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
            np.save(os.path.join(RESULTS_DIR, "init_labeled.npy"), labeled_ind) # save initially labeled points that are common to each test
            
            # check if test already completed previously
            choices_run_savename = os.path.join(RESULTS_DIR, f"choices_s2_s2.npy")
            if os.path.exists(choices_run_savename):
                print(f"Found choices for s2 in s2")
                return
            
            # split up 1 vs all for each of the classes
            num_splits = np.unique(labels).size 
            iters_per = args.iters // num_splits + 1
            
            for c in np.unique(labels):
                labels_c = np.zeros_like(labels)
                labels_c[labels == c] = 1
                
                labeled_ind, _, _ = run_S2(Graph, labeled_ind, labels_c[labeled_ind], labels_c, budget=iters_per)
            
            
            # save labeled indices
            np.save(choices_run_savename, labeled_ind[:args.iters + num_init])
            
            return

        print("------Starting S2 Tests-------")

        Parallel(n_jobs=args.numcores)(delayed(s2_test)(seed, Graph) for seed, Graph \
                in zip(seeds, Graphs))
