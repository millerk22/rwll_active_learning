import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import os
import numpy as np
from glob import glob
from functools import reduce

if __name__ == "__main__":
    parser = ArgumentParser(description="Compile Summary Stats of Active Learning Tests")
    parser.add_argument("--dataset", type=str, default='mnist-mod3')
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--resultsdir", type=str, default="results")
    args = parser.parse_args()
    


    # Get average and std curves over all tests
    overall_results_dir = os.path.join(args.resultsdir, f"{args.dataset}_overall_{args.iters}")
    if not os.path.exists(overall_results_dir):
        os.makedirs(overall_results_dir)

    results_models_directories = glob(os.path.join(args.resultsdir, f"{args.dataset}_results_*_{args.iters}", "*/"))
    acc_model_names_list = np.unique([fpath.split("/")[-2] for fpath in results_models_directories])
    for acc_model_name in tqdm(acc_model_names_list, desc=f"Saving results over all runs to: {overall_results_dir}", total=len(acc_model_names_list)):
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats.csv")
        acc_files = glob(os.path.join(args.resultsdir, f"{args.dataset}_results_*_{args.iters}", f"{acc_model_name}", "accs.csv"))
        dfs = []
        err_string = ""
        for f in sorted(acc_files):
            try:
                dfs.append(pd.read_csv(f))
            except:
                err_string += f.split("/")[1].split("_")[-2] + ", "
        
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
        
        if len(err_string) > 0:
            print(f"Error with {acc_model_name} and seeds {err_string}")
        
        
        
        # Do metrics summary
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats_metrics.csv")
        metric_files = glob(os.path.join(args.resultsdir, f"{args.dataset}_results_*_{args.iters}", f"{acc_model_name}", "metrics.csv"))
        dfs = []
        err_string = ""
        for f in sorted(metric_files):
            try:
                dfs.append(pd.read_csv(f))
            except:
                err_string += f.split("/")[1].split("_")[-2] + ", "
                
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
    
    
    

    # Get average and std curves over all tests
    overall_results_dir = os.path.join(args.resultsdir, f"{args.dataset}_overall_{args.iters}")
    if not os.path.exists(overall_results_dir):
        os.makedirs(overall_results_dir)

    results_models_directories = glob(os.path.join(args.resultsdir, f"{args.dataset}_results_*_{args.iters}", "*/"))
    acc_model_names_list = np.unique([fpath.split("/")[-2] for fpath in results_models_directories])
    for acc_model_name in tqdm(acc_model_names_list, desc=f"Saving results over all runs to: {overall_results_dir}", total=len(acc_model_names_list)):
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats_metrics.csv")
        metric_files = glob(os.path.join(args.resultsdir, f"{args.dataset}_results_*_{args.iters}", f"{acc_model_name}", "metrics.csv"))
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