import os
from glob import glob
import numpy as np
import pandas as pd


if __name__ == "__main__":
    results_dirs = glob("*_results_*/rwll*")
    for dir in results_dirs:
        parent, modelname = dir.split("/")
        if modelname in ["rwll01", "rwll1", "rwll001"]:
            os.rename(dir, os.path.join(parent, "".join(["old", modelname])))
        
