import argparse
import os
from plotutils import plot_acc

bbox_x = 1.01
resultsdir = "results"
savedir = os.path.join("imgs", "toy_figures")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acq", nargs="+", type=str, default=["unc : rwll"])
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-r", "--resultsdir", type=str, required=False, default=resultsdir)
    parser.add_argument("-s", "--savedir", type=str, required=False, default=savedir)
    args = parser.parse_args()

    plot_acc(dataset=args.dataset, modelname="rwll", resultsdir=args.resultsdir, savedir=args.savedir, ymin=70, bbox_to_anchor=(bbox_x,-0.16), acq_to_show=args.acq, xmax=50, ncol=3)
