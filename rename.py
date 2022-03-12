import os
from glob import glob
import numpy as np
import pandas as pd


if __name__ == "__main__":
    stats_fnames = glob("*_overall_*/rwll*_stats.csv")
    for fname in stats_fnames:
        parent, stats = fname.splt("/")
        if stats.split("_")[0] in ["rwll01", "rwll1", "rwll001"]:
            print(fname)
            print(os.path.join(parent, "".join(["old", stats])))
            break
            # os.rename(fname, os.path.join(parent, "".join(["old", stats])))

    choices_fnames = glob("*_results_*/choices_*_rwll*.npy")
    for fname in choices_fnames:
        parent, choice = fname.splt("/")
        modelname = choice.split("_")[-1].split(".")[0]

        if modelname in ["rwll01", "rwll1", "rwll001"]:
            acq = choice.split("_")[1]
            old_modelname = "".join(["old", modelname])
            print(fname)
            print(os.path.join(parent, f"choices_{acq}_{old_modelname}.npy"))
            break
            # os.rename(fname, os.path.join(parent, f"choices_{acq}_{old_modelname}.npy")))

    accs_fnames = glob("*_results_*/*/acc_*_rwll*.npy")
    for fname in accs_fnames:
        parent, child, acc = fname.splt("/")
        modelname = acc.split("_")[-1].split(".")[0]
        if modelname in ["rwll01", "rwll1", "rwll001"]:
            acq = acc.split("_")[1]
            old_modelname = "".join(["old", modelname])
            print(fname)
            print(os.path.join(parent, child, f"acc_{acq}_{old_modelname}.npy"))
            break
            # os.rename(fname, os.path.join(parent, child, f"acc_{acq}_{old_modelname}.npy")))

    accdf_fnames = glob("*_results_*/*/accs.csv")
    for fname in accdf_fnames:
        df = pd.read_csv(fname)
        columns = [col for col in df.columns if col.split(" : ") in ["rwll01", "rwll1", "rwll001"]]
        newcolumns = [" : ".join([col.split(" : ")[0], "old" + col.split(" : ")[1]])]
        rename = {c:nc for c,nc in zip(columns, newcolumns)}
        df.rename(columns=rename, inplace=True)
        break
        # df.to_csv(fname, index=None)
