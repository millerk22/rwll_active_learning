# Poisson reWeighted Laplacian Uncertainty Sampling for Graph-based Active Learning

This repo contains the code used to perform the numerical experiments contained in the paper [Poisson reWeighted Laplacian Uncertainty Sampling for Graph-based Active Learning](https://arxiv link) by Kevin Miller (Oden Institute) and Jeff Calder (UMN, Twin Cities). We propose a computationally efficient type of uncertainty sampling criterion designed for the Poisson Reweighted Laplacian Learning (PWLL) graph-based semi-supervised model for classification. Experiments demonstrate the favorably explorative behavior of this acquisition function to query sample points from unexplored clusters, in the low-label rate regime of active learning. 

The experiments in this repository are built around Jeff Calder's [GraphLearning](https://github.com/jwcalder/GraphLearning) Python package. 

## Required packages

See the ``requirements.txt`` file for Python packages necessary to run the scripts in this repository.

## Experiment organization

Except for the similarity graphs for MNIST, FASHIONMNIST, and EMNIST datasets, we have provided the calculated similarity graphs and knn information necessary for running experiments. Running an experiment begins with a bash script (e.g., ``run_decay.sh``), wherein other Python scripts are called to
1. Perform an active learning test on a ``dataset`` of sequentially selecting ``iters=n`` query points ``x_1, x_2, ..., x_n`` via acquisition functions listed in ancialliary file (e.g., ``config.yaml``),
2. For each acquisition function, evaluate the accuracy of using each subset of query points ``{x_1, ..., x_j}`` for ``j=1,...,n`` in graph-based models listed in the ancilliary file (e.g., ``config.yaml``)
3. Compile a summary of computed accuracies into a ``results`` directory

For example, to perform the ``Box`` experiment one would run the following:
```
python test_al_gl.py --dataset box --config config.yaml --resultsdir results
python accuracy_al_gl.py --dataset box --config config.yaml --resultsdir results
python compile_summary.py --dataset box --resultsdir results
```

We have provided the bash script ``run_decay.sh`` with most experiments commented out to archive the input parameter settings for the different test and provide an example of usage.

__Note:__ In order to adapt the pipeline to the Isolet experiments and the VOpt/SigmaOpt "Full" acquisition functions, we created separate scripts ``test_al_gl_isolet.py``, ``accuracy_al_gl_isolet.py``, and ``test_al_gl_voptfull.py``. 
 

