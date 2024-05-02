import numpy as np
from scipy.io import loadmat
from glob import glob
import os
from argparse import ArgumentParser
import pandas as pd


run2seed = {0:2, 1:3, 2:5, 3:6, 4:15, 5:16, 6:29, 7:30, 8:49, 9:50} # the seeds we chose for the experiments



def save_results(summary_dirs, suff="land"):
    if len(summary_dirs) > 0: # compile average statistics
        print(len(summary_dirs))
        for i, summ_dir in enumerate(summary_dirs):
            if i == 0:
                 results = np.load(os.path.join(summ_dir, f"acc_{suff}_{suff}.npy"))[np.newaxis, :]
            else:
                results = np.concatenate((results, np.load(os.path.join(summ_dir, f"acc_{suff}_{suff}.npy"))[np.newaxis, :]), axis=0)

        print(f"len of results = {results.shape}")
        columns = {f"{suff} : {suff} : avg" : np.mean(results, axis=0), f"{suff} : {suff} : std" : np.std(results, axis=0)}
        print(columns)
        columns["budget"]  = np.load(os.path.join(summary_dirs[0], f"budget_{suff}.npy"))
        df = pd.DataFrame.from_dict(columns)
        print(df.head())
        overall_dir = summary_dirs[0].split("_results_")[-2] + "_overall_" + str(args.n)
        if not os.path.exists(overall_dir):
            os.makedirs(overall_dir)
        df.to_csv(os.path.join(overall_dir, f"acc_{suff}.csv"))
    else:
        print(f"summary dirs is empty for suff = {suff}, skipping...")
    return 




if __name__ == "__main__":
    parser = ArgumentParser(description="Script for converting and organizing MATLAB results to our results directories")
    parser.add_argument("--resultsdir", type=str, default="results_other")
    parser.add_argument("--matlabdir", type=str, default="data_matlab")
    parser.add_argument("--n", type=int, default=100, help="how many total choices to include")
    parser.add_argument("--save_acc", type=int, default=0, help="bool flag. If 1, then try to record the accuracy of choices in the native SSL model (e.g. LAND)")
    parser.add_argument("--only", type=str, default="", help="specify dataset to only unload")
    args = parser.parse_args()
    
    
    if args.only != "":
        dirnames = glob(os.path.join(args.matlabdir, args.only))
    else:
        dirnames = glob(os.path.join(args.matlabdir, "*"))
    for dirname in dirnames:
        print(dirname)
        summary_dirs_land, summary_dirs_cal = [], []
        fnames = glob(os.path.join(dirname, f"*results*.mat"))
        for fname in sorted(fnames):
            dataset = fname.split("/")[-2]
            run = int(fname.split("/")[-1].split(".")[-2][-1])
            seed = run2seed[run]
            if "LAND" in fname:
                choices = loadmat(fname)["Queries_LAND"].reshape(-1) - 1
                savedir = os.path.join(args.resultsdir, f"{dataset}_results_{seed}_{args.n}")
                n_init = np.load(os.path.join(savedir, "init_labeled.npy")).size
                
                if not os.path.exists(savedir):
                    print(f"savedir = {savedir} DNE...")
                else:
                    savename = os.path.join(savedir, f"choices_land_land.npy")
                    np.save(savename, choices[:n_init+args.n])
                
                if args.save_acc:
                    res = loadmat(fname)
                    acc = res["OA_LAND"].flatten()
                    budget = res["Budget"].flatten()
                    
                    if not os.path.exists(os.path.join(savedir, "land")):
                        os.makedirs(os.path.join(savedir, "land"))
                    
                    np.save(os.path.join(savedir, "land", "acc_land_land.npy"), acc)
                    np.save(os.path.join(savedir, "land", "budget_land.npy"), budget)
                    
                    summary_dirs_land.append(os.path.join(savedir, "land"))
                    
            elif "CAL" in fname:
                choices = loadmat(fname)["Queries_CAL"].reshape(-1) - 1
                savedir = os.path.join(args.resultsdir, f"{dataset}_results_{seed}_{args.n}")
                n_init = np.load(os.path.join(savedir, "init_labeled.npy")).size
                
                if not os.path.exists(savedir):
                    print(f"savedir = {savedir} DNE...")
                else:
                    savename = os.path.join(savedir, f"choices_cal_cal.npy")
                    np.save(savename, choices[:n_init+args.n])
                
                if args.save_acc:
                    res = loadmat(fname)
                    acc = res["OA_CAL"].flatten()
                    budget = res["Budget"].flatten()
                    
                    if not os.path.exists(os.path.join(savedir, "cal")):
                        os.makedirs(os.path.join(savedir, "cal"))
                    
                    np.save(os.path.join(savedir, "cal", "acc_cal_cal.npy"), acc)
                    np.save(os.path.join(savedir, "cal", "budget_cal.npy"), budget)
                    
                    summary_dirs_cal.append(os.path.join(savedir, "cal"))
                    
                    
            else:
                print(f"Unexpected input fname = {fname}, skipping...")
                continue
        
        
        save_results(summary_dirs_land, suff="land")
        save_results(summary_dirs_cal, suff="cal")
            
            
        
