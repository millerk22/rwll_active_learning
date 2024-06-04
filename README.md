# Poisson reWeighted Laplacian Uncertainty Sampling for Graph-based Active Learning

This repo contains the code used to perform the numerical experiments contained in the paper [Poisson reWeighted Laplacian Uncertainty Sampling for Graph-based Active Learning](https://arxiv link) by Kevin Miller (Oden Institute) and Jeff Calder (UMN, Twin Cities). We propose a computationally efficient type of uncertainty sampling criterion designed for the Poisson Reweighted Laplacian Learning (PWLL) graph-based semi-supervised model for classification. Experiments demonstrate the favorably explorative behavior of this acquisition function to query sample points from unexplored clusters, in the low-label rate regime of active learning. 

The experiments in this repository are built around Jeff Calder's [GraphLearning](https://github.com/jwcalder/GraphLearning) Python package. 

## Required packages

In addition to standard Python packages ``numpy, scipy, pandas,`` and ``matplotlib``, you will need 
* ``graphlearning``: ``pip install graphlearning``
* ``tqdm``
* ``pytorch`` (if looking to train the VAE embeddings for EMNIST locally)
* `yaml (pip install pyyaml)` to parse the configuration files

A working `requirements.[os].txt` file is available at the root of this repo which can be used to install the dependencies of this project as follows:

## Installation

For the latest python version (`3.12.3`):

```
cd [project-root]
# create a virtual env in the directory .venv
python -m venv .venv 

# activate the virtual environment
source .venv/bin/activate

# install the dependencies
pip install -r requirements.[os].txt
```

## Experiment organization

Except for the similarity graphs for MNIST, FASHIONMNIST, and EMNIST datasets, we have provided the calculated similarity graphs and knn information necessary for running experiments. Running an experiment begins with a bash script (e.g., ``run_decay.sh``), wherein other Python scripts are called to
1. Perform an active learning test on a ``dataset`` of sequentially selecting ``iters=n`` query points ``x_1, x_2, ..., x_n`` via acquisition functions listed in ancialliary file (e.g., ``config.yaml``),
2. For each acquisition function, evaluate the accuracy of using each subset of query points ``{x_1, ..., x_j}`` for ``j=1,...,n`` in graph-based models listed in the ancilliary file (e.g., ``config.yaml``)
3. Compile a summary of computed accuracies into a ``results`` directory

For example, to perform the ``Box`` experiment one would run the following:
```
python test_al_gl.py --dataset box --config config.yaml --resultsdir results --metric raw
python accuracy_al_gl.py --dataset box --config config.yaml --resultsdir results --metric raw
python compile_summary.py --dataset box --resultsdir results
```

You can also pass the flag `--use-load-graph` to use the loadgraph function in the GraphLearning package to load the similarity graph and knn information from the precomputed files. For example, in order to run the experiment on `pubmed` graph using `config-pubmed.yaml`, you can perform the following:

```
python test_al_gl.py --dataset pubmed --config config-pubmed.yaml --resultsdir results --metric raw --use-load-graph
python accuracy_al_gl.py --dataset pubmed --config config-pubmed.yaml --resultsdir results --metric raw --use-load-graph
python compile_summary.py --dataset pubmed --resultsdir results
```

We have provided the bash script ``run_decay.sh`` with most experiments commented out to archive the input parameter settings for the different test and provide an example of usage.

__Note:__ In order to adapt the pipeline to the Isolet experiments and the VOpt/SigmaOpt "Full" acquisition functions, we created separate scripts ``test_al_gl_isolet.py``, ``accuracy_al_gl_isolet.py``, and ``test_al_gl_voptfull.py``. 

## Viewing results

After running the experiments above, you can view the results by running the following:
```
python inference.py -d pubmed
```

A list of options are available for viewing the results, which can be viewed by running:
```
python inference.py -h
```

You can also find the details for the plots on `FinalPlots.ipynb` notebook.
 
## VAE Training for EMNIST

While the VAE embeddings are already precomputed in the GraphLearning package, we provide the script ``emnist_vae.py`` we used to train a standard VAE for the EMNIST dataset. The resulting embeddings are stored in the zipped file ``data/emnist_vae.npz``.


## MATLAB data for comparison

In order to compare against Cautious Active Learning (CAL) and Learning by Active Nonlinear Diffusion (LAND), we used the respective authors' code which was written in MATLAB. We have provided the outputs of their code in the directories contained in ``data_matlab``. Further questions about their code (and our experiments with incorporating their code) can be directed to ``ksmiller@utexas.edu``. 
