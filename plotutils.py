import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import os
import seaborn as sns
sns.set_style('whitegrid')
import graphlearning as gl
from test_al_gl import get_active_learner
import pickle
import acquisitions
from utils import get_active_learner


acq_color = {'random':'r', 'vopt':'cyan', 'voptfull':'grey',  'mcvopt':'k', 'sopt':'lime', 'soptfull':'magenta',
             'uncnorm':'blue',
             'uncnormkde' : 'blue', 
             'uncnormprop++':'blue',
             'uncnormprop++kde':'blue',
             'uncnormdecaytau':'gold', 
             'uncnormdecaytaukde':'gold',
             'uncnormprop++decaytau':'gold',
            'uncnormprop++decaytaukde':'gold',
             'uncnormprop':'purple',
             'uncnormpropdecaytau':'gold',
            'uncnormprop++switchk':'brown',
             'uncnormprop++switchkkde':'slategrey',
             'uncnormswitchk':'purple',
             'uncnormswitchkcheat':'green',
             'unc':'green',
             's2':'brown',
             'land' : 'purple',
             'cal' : 'orange'
          }

display_name_map = {'unc': 'Unc. (SM)', 'uncnorm' : 'Unc. (Norm)', 'uncnormdecaytau': r'Unc. (Norm, $\tau \rightarrow 0$)',
                   'vopt': 'VOpt (ST)', 'mcvopt':'MCVOpt', 'random':'Random', 'sopt':r'$\Sigma$-Opt (ST)', 'soptfull':r'$\Sigma$-Opt',
                    'uncnormkde':'Unc. (Norm) KDE', 
                   'uncnormdecaytaukde':r'Unc. (Norm, $\tau \rightarrow 0$) KDE', 'voptfull':'VOpt',
                   "uncnormprop++decaytaukde" : r'Unc. (Norm, $\tau \rightarrow 0$) KDE + Prop',
               "uncnormprop++kde" : "Unc. (Norm) KDE + Prop",
               "uncnormprop++switchkkde":r"Unc. (Norm, Switch $\tau$) KDE + Prop",
                   "uncnormprop++" : "Unc. (Norm) Prop",
               "uncnormprop++switchk":r"Unc. (Norm, Switch $\tau$) Prop + Switch",
                   "uncnormprop++decaytau": r"Unc. (Norm, $\tau \rightarrow 0$) Prop",
                   "s2":r"$S^2$",
                   "land": r"LAND",
                   "cal" : r"CAL" }

marker = {'unc': '*', 'vopt': 'v', 'mcvopt':'>', 'random':'.','voptfull': 'v', 'sopt':'s', 'soptfull':'s',
          'uncnorm' : 'o', 
          'uncnormdecaytau': '^',
           'uncnormkde':'+', 
           'uncnormdecaytaukde':'x', 
         "uncnormprop++decaytaukde" : 'x', 
          'uncnormprop++':'+',
            "uncnormprop++kde" : "+", 
          "uncnormprop++decaytau" : "x",
            "uncnormprop++switchkkde":"+",
         "s2":"h",
         "land":"x",
         "cal" : "d"}
 
    

class dummy_args(object):
    def __init__(self):
        self.gamma = 0.1

def plot_acc_toy(dataset="box", acq_to_show=["unc : rwll", "uncnorm : rwll0010", "uncnorm : rwll0010", 
                 "uncnormdecaytau : rwll0010"], resultsdir="results", savedir=None, ymin=80, bbox_to_anchor=(1.15,-0.16), 
                  idx_heatmap=[0, 15, 50], showbinned=False, seed=2, tot_iters=100, simplex=False, knn=20, 
                 eig_normalization='combinatorial'):
    nstart = np.load(f"{resultsdir}/{dataset}_results_2_{tot_iters}/init_labeled.npy").size
    
    dataset_data = np.load(f"data/{dataset}_raw.npz")
    X, labels = dataset_data['data'], dataset_data['labels']
    G = gl.graph.load(f"data/{dataset}_{knn}")
    
    for acq in acq_to_show:
        acq_name, modelname = acq.split(" : ")
        
        if modelname == 'rwll':
            tau = 0.0
        elif modelname == 'rwll0100':
            tau = 0.01
        elif modelname == 'rwll0010':
            tau = 0.001
        else:
            raise NotImplementedError(f"accmodelname = {modelname} not yet implemented")
        model = gl.ssl.laplace(G, reweighting='poisson', tau=tau)
        
        
        fnames = sorted(glob(f"{resultsdir}/{dataset}_results_{seed}_{tot_iters}/choices_{acq_name}_{modelname}.npy"))
        print(acq, fnames)
        if len(fnames) == 0:
            continue
        choices = np.load(fnames[0])
        
        for idx in idx_heatmap:
            train_ind = choices[:idx+nstart]
            args = dummy_args()
            # fetch active_learning object
            AL = get_active_learner(acq_name, model, train_ind, labels[train_ind], eig_normalization, args)
            
            if modelname[-1] == '0' and idx > 8 and acq_name[-8:] == 'decaytau':
                model.tau = np.zeros_like(model.tau)
            query_point, af_vals = AL.select_queries(return_acq_vals=True)
            
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(X[AL.unlabeled_ind,0], X[AL.unlabeled_ind,1], c=af_vals, cmap='viridis')
            ax.scatter(X[train_ind,0], X[train_ind,1], c='r', marker='*', s=80)
            ax.set_title(f"{dataset}, {acq_name} values at iter = {idx}")
            plt.axis('equal')
            plt.axis('off')
            if savedir is not None:
                ax.set_title("")
                plt.savefig(f"{savedir}/{acq_name}_{modelname}_afvals_{idx}.jpg", format="jpeg", bbox_inches='tight')
            plt.show()
            
            
            if simplex:
                simplex_dom = np.linspace(0, 1, 11)
                fig, ax = plt.subplots(figsize=(4,4))
                u0 = AL.u[:,0].flatten()[::10]
                u1 = AL.u[:,1].flatten()[::10]
                ax.scatter(u0, u1, marker='x', s=50, zorder=7, alpha=0.5)
                ax.scatter(AL.u[AL.unlabeled_ind,0], AL.u[AL.labeled_ind,1], marker='*',c='r', zorder=8, s=50)
                ax.plot(simplex_dom, 1. - simplex_dom, "--", linewidth=2.5, color='k', alpha=0.3) 
                ax.axis('square')
                ax.set_xticks(np.linspace(0,1,6))
                ax.set_xlabel(r"$u_0(x)$", fontsize=16)
                ax.set_ylabel(r"$u_1(x)$", fontsize=16)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                plt.show()
    
    return
 




    
def plot_acc(dataset="mnist-mod3", modelname="rwll", resultsdir="results", 
                      savedir=None, ymin=80, bbox_to_anchor=(1.15,-0.16), acq_to_show=None,
                 plot_qualifier='', xmax=None, ncol=2):
    
    if dataset[:6] != 'emnist':
        nc = np.load(f"{resultsdir}/{dataset}_results_2_100/init_labeled.npy").size
    else:
        nc = np.load(f"{resultsdir}/{dataset}_results_2_400/init_labeled.npy").size
    
    if acq_to_show is None:
        acq_to_show = [ f"random : rwll", 
                   "unc : rwll",
                   "uncnorm : rwll0010",
                   "uncnormdecaytau : rwll0010", 
                   "vopt : rwll0100", "mcvopt : rwll"
                  ]

    print(f"Num classes = {nc}")
    if dataset in ['emnist-mod5', 'emnistvcd']:
        fnames = glob(f"{resultsdir}/{dataset}_overall_400/*_stats.csv")
    else:
        fnames = glob(f"{resultsdir}/{dataset}_overall_100/*_stats.csv")
    fname = sorted([fname for fname in fnames if fname.split("/")[-1][:len(modelname)+1] == modelname + "_"])[0]
    print(fname)
    df = pd.read_csv(fname)
    fig, ax = plt.subplots(figsize=(8,5))
#     modelname = fname.split("/")[-1].split("_")[0]
    to_plot_cols = [col for col in df.columns if col.split(" : ")[-1] == 'avg']
    to_plot_cols = sorted([col for col in to_plot_cols if " : ".join(col.split(" : ")[:2]) in acq_to_show])
    rename = {col : [s.strip().capitalize() for s in col.split(" :")[:3]] for col in to_plot_cols}
    for col,val in rename.items():
        rename[col] = val[0]
    print(to_plot_cols)
    df = df[to_plot_cols].rename(columns=rename)
    df = df.shift(periods=nc)
    colormap = {col_name:acq_color[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns }
    markermap = {col_name:marker[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns}
    
    title = f"Accuracy in {modelname.upper()}"
    if savedir is not None:
        title = None
    
    for col in df:
#         df[col].plot(ax=ax, title=title, linewidth=2.0,  marker=markermap[col], color=colormap[col], alpha=0.8)
        column = df[col]
        values = column.iloc[::2]
#         ax.plot(column.index, column.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        ax.plot(values.index, values.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        
        x = values.index
        ax.scatter(values.index, values.values, linewidth=2.0,  marker=markermap[col],color=colormap[col],
                   alpha=0.8, label=col)
    
    
    if xmax is None:
        ax.set_xlim(0, 100)
        if dataset in ['emnist-mod5', 'emnistvcd']:
            ax.set_xlim(0,150)
    else:
        ax.set_xlim(0, xmax)
    
    
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = np.array(handles), np.array(labels)
    labels = np.array([display_name_map[l.lower()] for l in labels])
    ordering = np.argsort([l for l in labels])
    lgd = ax.legend(handles[ordering], labels[ordering], bbox_to_anchor=bbox_to_anchor,ncol=ncol,
                   fontsize=14)
    ax.set_xlabel("Size of Labeled Set", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if savedir is None:
        plt.suptitle(dataset.upper())
    else:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(f"{savedir}/{dataset}_{modelname}_accplot{plot_qualifier}.jpg", 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', format="jpeg")
    plt.show()
    

def plot_cluster_exploration(dataset="mnist", problem="mod3", resultsdir="results", savedir=None,
                            xmax=100, plot_qualifier='', cols_to_plot=None, ncol=2,
                            bbox_to_anchor=(1.01, -0.16)):
    if cols_to_plot is None:
        cols_to_plot = ["unc_rwll", 
                    "uncnorm_rwll0010", "uncnormdecaytau_rwll0010", 
                    'random_rwll', 
                    'vopt_rwll0100', 
                    'mcvopt_rwll']
    
    if dataset == 'cifar' or dataset == "cifarimb" or dataset == "cifarsmall":
        X, clusters = gl.datasets.load(dataset, metric='aet')
    elif dataset in ["pavia", "salinas"]:
        X, clusters = gl.datasets.load(dataset, metric='hsi')
    elif dataset in ['isolet', 'box', 'blobs']:
        X, clusters = gl.datasets.load(dataset, metric='raw')
    else:
        if dataset == 'emnistvcd':
            X, clusters = gl.datasets.load('emnist', metric='vae')
        else:
            X, clusters = gl.datasets.load(dataset, metric='vae')
        
    print(np.unique(clusters))
    
    if problem == 'evenodd':
        nc = 2 
    elif problem == 'mod3' or dataset == 'emnistvcd':
        nc = 3
    elif problem == 'mod5':
        nc = 5
    elif problem == "hsi":
        nc = np.unique(clusters).size
    elif dataset == 'isolet':
        nc = 1
    

    choice_cutoff = xmax+10

    if problem is None:
        if 'emnist' in dataset:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}_results_*_400", "choices_*.npy"))
        else:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}_results_*_100", "choices_*.npy"))
    else:
        if 'emnist' in dataset:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}-{problem}_results_*_400", "choices_*.npy"))
        else:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}-{problem}_results_*_100", "choices_*.npy"))
    
    
    print(len(choices_fnames))

    fracs_clusters = {}

    num_clusters = np.unique(clusters).size

    for fname in choices_fnames:
        acq_func = fname.split("choices_")[-1].split(".")[0]
        seed = fname.split("_results_")[-1].split("_")[0]

        if acq_func not in fracs_clusters:
            fracs_clusters[acq_func] = {}

        choices = np.load(fname)
        fracs = np.array([])
        for i in range(nc, choice_cutoff + 1):
            choices_i_clusters = clusters[choices[:i]]
            fracs = np.append(fracs, np.unique(choices_i_clusters).size/num_clusters)

        fracs_clusters[acq_func][seed] = fracs

    avgs = {acq_func : 0.0 for acq_func in fracs_clusters}
    for acq_func, seed_dicts in fracs_clusters.items():
        vals = np.array([row for s, row in seed_dicts.items()])
        avgs[acq_func] = np.average(vals, axis=0)

    df = pd.DataFrame(avgs)

    
    cols_to_plot = sorted([col for col in cols_to_plot if col in df.columns])
    # rename = {col: ", ".join([s.capitalize() for s in col.split("_")]) for col in cols_to_plot}
    rename = {col: col.split("_")[0].capitalize() for col in cols_to_plot}
    df = df[cols_to_plot].rename(columns=rename)
    colormap = {col_name:acq_color[col_name.split("_")[-1].lower()] for col_name in df.columns }
    markermap = {col_name:marker[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns}

    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel("Active Learning Iteration", fontsize=14)
    ax.set_ylabel("Fraction of Clusters Sampled", fontsize=14)
    ax.set_xlim(0, xmax)
    title_str = dataset.upper()
    if problem is not None:
        title_str += "-" + problem
    
    title = f"Clustering Fractions"
    if savedir is not None:
        title = None
    
    for col in df:
#         df[col].plot(ax=ax, title=title, linewidth=2.0,  marker=markermap[col], color=colormap[col], alpha=0.8)
        column = df[col]
        values = column.iloc[::2]
        # ax.plot(column.index, column.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        ax.plot(values.index, values.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        x = values.index
        ax.scatter(values.index, values.values, linewidth=2.0,  marker=markermap[col], color=colormap[col],
                   alpha=0.8, label=col, s=50)
    
    
    
    
    ax.set_xlim(0, xmax)
    if dataset in ['emnist-mod5', 'emnistvcd']:
        ax.set_xlim(0,150)
    
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = np.array(handles), np.array(labels)
    labels = np.array([display_name_map[l.lower()] for l in labels])
                 
    
    ordering = np.argsort([l for l in labels])
    lgd = ax.legend(handles[ordering], labels[ordering], bbox_to_anchor=bbox_to_anchor,ncol=ncol,
                   fontsize=14)
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if savedir is None:
        plt.suptitle(dataset.upper())
    else:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(f"{savedir}/{dataset}_clusterplot{plot_qualifier}.jpg", 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', format='jpeg')
    plt.show()
    return

   













###############################
    
def plot_acc_multidir(dataset="mnist-mod3", modelname="rwll", resultsdir="results", resultsdir_other="results_kde", 
                      savedir=None, ymin=80, bbox_to_anchor=(1.15,-0.16), acq_to_show=None,
                     acq_to_show_other=None, plot_qualifier='', xmax=None, ncol=2):
    
    if dataset[:6] != 'emnist':
        nc = np.load(f"{resultsdir}/{dataset}_results_2_100/init_labeled.npy").size
    else:
        nc = np.load(f"{resultsdir}/{dataset}_results_2_400/init_labeled.npy").size
    
    if acq_to_show is None:
        acq_to_show = [ f"random : rwll", 
                   "unc : rwll",
                   "uncnorm : rwll0010",
                   "uncnormdecaytau : rwll0010", 
                   "vopt : rwll0100", "mcvopt : rwll"
                  ]
    if acq_to_show_other is None:
        acq_to_show_other = []

    print(f"Num classes = {nc}")
    if dataset in ['emnist-mod5', 'emnistvcd']:
        fnames = glob(f"{resultsdir}/{dataset}_overall_400/*_stats.csv")
    else:
        fnames = glob(f"{resultsdir}/{dataset}_overall_100/*_stats.csv")
    fname = sorted([fname for fname in fnames if fname.split("/")[-1][:len(modelname)+1] == modelname + "_"])[0]
    print(fname)
    df = pd.read_csv(fname)
    fig, ax = plt.subplots(figsize=(8,5))
#     modelname = fname.split("/")[-1].split("_")[0]
    to_plot_cols = [col for col in df.columns if col.split(" : ")[-1] == 'avg']
    to_plot_cols = sorted([col for col in to_plot_cols if " : ".join(col.split(" : ")[:2]) in acq_to_show])
    rename = {col : [s.strip().capitalize() for s in col.split(" :")[:3]] for col in to_plot_cols}
    for col,val in rename.items():
        rename[col] = val[0]
    print(to_plot_cols)
    df = df[to_plot_cols].rename(columns=rename)
    df = df.shift(periods=nc)
    colormap = {col_name:acq_color[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns }
    markermap = {col_name:marker[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns}
    
    title = f"Accuracy in {modelname.upper()}"
    if savedir is not None:
        title = None
    
    for col in df:
#         df[col].plot(ax=ax, title=title, linewidth=2.0,  marker=markermap[col], color=colormap[col], alpha=0.8)
        column = df[col]
        values = column.iloc[::2]
        # ax.plot(column.index, column.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        ax.plot(values.index, values.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        x = values.index
        ax.scatter(values.index, values.values, linewidth=2.0,  marker=markermap[col],color=colormap[col],
                   alpha=0.8, label=col)
    
    if len(acq_to_show_other) > 0:
        if dataset in ['emnist-mod5', 'emnistvcd']:
            fnames = glob(f"{resultsdir_other}/{dataset}_overall_400/*_stats.csv")
        else:
            fnames = glob(f"{resultsdir_other}/{dataset}_overall_100/*_stats.csv")
        fname = sorted([fname for fname in fnames if fname.split("/")[-1][:len(modelname)+1] == modelname + "_"])[0]
        print(fname)
        df = pd.read_csv(fname)
        to_plot_cols = [col for col in df.columns if col.split(" : ")[-1] == 'avg']
        to_plot_cols = sorted([col for col in to_plot_cols if " : ".join(col.split(" : ")[:2]) in acq_to_show_other])
        rename = {col : [s.strip().capitalize() for s in col.split(" :")[:3]] for col in to_plot_cols}
        for col,val in rename.items():
            rename[col] = val[0]
        print(to_plot_cols)
        df = df[to_plot_cols].rename(columns=rename)
        df = df.shift(periods=nc)
        colormap = {col_name:acq_color[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns }
        markermap = {col_name:marker[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns}


        for col in df:
            column = df[col]
            values = column.iloc[::2]
            # ax.plot(column.index, column.values, linewidth=2.0, color=colormap[col], alpha=0.8)
            ax.plot(values.index, values.values, linewidth=2.0, color=colormap[col], alpha=0.8)
            x = values.index
            ax.scatter(values.index, values.values, linewidth=2.0,  marker=markermap[col], color=colormap[col],
                       alpha=0.8, label=col)
    
    if xmax is None:
        ax.set_xlim(0, 100)
        if dataset in ['emnist-mod5', 'emnistvcd']:
            ax.set_xlim(0,150)
    else:
        ax.set_xlim(0, xmax)
    
    
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = np.array(handles), np.array(labels)
    labels = np.array([display_name_map[l.lower()] for l in labels])
    ordering = np.argsort([l for l in labels])
    lgd = ax.legend(handles[ordering], labels[ordering], bbox_to_anchor=bbox_to_anchor,ncol=ncol,
                   fontsize=14)
    ax.set_xlabel("Size of Labeled Set", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if savedir is None:
        plt.suptitle(dataset.upper())
    else:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(f"{savedir}/{dataset}_{modelname}_accplot{plot_qualifier}.jpg", 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', format="jpeg")
    plt.show()   
    
    
def plot_cluster_exploration_multidir(dataset="mnist", problem="mod3", resultsdir="results", resultsdir_other="results_kde", 
                                      savedir=None,xmax=100, plot_qualifier='', cols_to_plot=None,
                                        cols_to_plot_other=None, ncol=2,
                                        bbox_to_anchor=(1.01, -0.16)):
    if cols_to_plot is None:
        cols_to_plot = ["unc_rwll", 
                    "uncnorm_rwll0010", "uncnormdecaytau_rwll0010", 
                    'random_rwll', 
                    'vopt_rwll0100', 
                    'mcvopt_rwll']
    if cols_to_plot_other is None:
        cols_to_plot_other = [] #["uncnormkde_rwll0010", "uncnormdecaytaukde_rwll0010", ]
    
    if dataset == 'cifar' or dataset == "cifarimb" or dataset == "cifarsmall":
        X, clusters = gl.datasets.load(dataset, metric='aet')
    elif dataset in ["pavia", "salinas"]:
        X, clusters = gl.datasets.load(dataset, metric='hsi')
    elif dataset in ['isolet', 'box', 'blobs']:
        X, clusters = gl.datasets.load(dataset, metric='raw')
    else:
        if dataset == 'emnistvcd':
            X, clusters = gl.datasets.load('emnist', metric='vae')
        else:
            X, clusters = gl.datasets.load(dataset, metric='vae')
        
    print(np.unique(clusters))
    
    if problem == 'evenodd':
        nc = 2 
    elif problem == 'mod3' or dataset == 'emnistvcd':
        nc = 3
    elif problem == 'mod5':
        nc = 5
    elif problem == "hsi":
        nc = np.unique(clusters).size
    elif dataset == 'isolet':
        nc = 1
    else:
        nc = 2
    

    choice_cutoff = xmax+10

    if problem is None:
        if 'emnist' in dataset:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}_results_*_400", "choices_*.npy"))
        else:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}_results_*_100", "choices_*.npy"))
    else:
        if 'emnist' in dataset:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}-{problem}_results_*_400", "choices_*.npy"))
        else:
            choices_fnames = glob(os.path.join(resultsdir, f"{dataset}-{problem}_results_*_100", "choices_*.npy"))
    
    
    print(len(choices_fnames))

    fracs_clusters = {}

    num_clusters = np.unique(clusters).size

    for fname in choices_fnames:
        acq_func = fname.split("choices_")[-1].split(".")[0]
        seed = fname.split("_results_")[-1].split("_")[0]

        if acq_func not in fracs_clusters:
            fracs_clusters[acq_func] = {}

        choices = np.load(fname)
        fracs = np.array([])
        for i in range(nc, choice_cutoff + 1):
            choices_i_clusters = clusters[choices[:i]]
            fracs = np.append(fracs, np.unique(choices_i_clusters).size/num_clusters)

        fracs_clusters[acq_func][seed] = fracs

    avgs = {acq_func : 0.0 for acq_func in fracs_clusters}
    for acq_func, seed_dicts in fracs_clusters.items():
        vals = np.array([row for s, row in seed_dicts.items()])
        avgs[acq_func] = np.average(vals, axis=0)

    df = pd.DataFrame(avgs)

    
    cols_to_plot = sorted([col for col in cols_to_plot if col in df.columns])
    rename = {col: col.split("_")[0].capitalize() for col in cols_to_plot}
    df = df[cols_to_plot].rename(columns=rename)
    colormap = {col_name:acq_color[col_name.split("_")[-1].lower()] for col_name in df.columns }
    markermap = {col_name:marker[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns}

    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel("Active Learning Iteration", fontsize=14)
    ax.set_ylabel("Fraction of Clusters Sampled", fontsize=14)
    ax.set_xlim(0, xmax)
    title_str = dataset.upper()
    if problem is not None:
        title_str += "-" + problem
    
    title = f"Clustering Fractions"
    if savedir is not None:
        title = None
    
    for col in df:
#         df[col].plot(ax=ax, title=title, linewidth=2.0,  marker=markermap[col], color=colormap[col], alpha=0.8)
        column = df[col]
        values = column.iloc[::2]
        # ax.plot(column.index, column.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        ax.plot(values.index, values.values, linewidth=2.0, color=colormap[col], alpha=0.8)
        x = values.index
        ax.scatter(values.index, values.values, linewidth=2.0,  marker=markermap[col], color=colormap[col],
                   alpha=0.8, label=col, s=50)
    
    
    if cols_to_plot_other is not None:
        if problem is None:
            if 'emnist' in dataset:
                choices_fnames = glob(os.path.join(resultsdir_other, f"{dataset}_results_*_400", "choices_*.npy"))
            else:
                choices_fnames = glob(os.path.join(resultsdir_other, f"{dataset}_results_*_100", "choices_*.npy"))
        else:
            if 'emnist' in dataset:
                choices_fnames = glob(os.path.join(resultsdir_other, f"{dataset}-{problem}_results_*_400", "choices_*.npy"))
            else:
                choices_fnames = glob(os.path.join(resultsdir_other, f"{dataset}-{problem}_results_*_100", "choices_*.npy"))


        print(len(choices_fnames))

        fracs_clusters = {}

        num_clusters = np.unique(clusters).size

        for fname in choices_fnames:
            acq_func = fname.split("choices_")[-1].split(".")[0]
            seed = fname.split("_results_")[-1].split("_")[0]

            if acq_func not in fracs_clusters:
                fracs_clusters[acq_func] = {}

            choices = np.load(fname)
            fracs = np.array([])
            for i in range(nc, choice_cutoff + 1):
                choices_i_clusters = clusters[choices[:i]]
                fracs = np.append(fracs, np.unique(choices_i_clusters).size/num_clusters)

            fracs_clusters[acq_func][seed] = fracs

        avgs = {acq_func : 0.0 for acq_func in fracs_clusters}
        for acq_func, seed_dicts in fracs_clusters.items():
            vals = np.array([row for s, row in seed_dicts.items()])
            avgs[acq_func] = np.average(vals, axis=0)

        df = pd.DataFrame(avgs)


        cols_to_plot_kde = sorted([col for col in cols_to_plot_other if col in df.columns])
        rename = {col: col.split("_")[0].capitalize() for col in cols_to_plot_other}
        df = df[cols_to_plot_other].rename(columns=rename)
        colormap = {col_name:acq_color[col_name.split("_")[-1].lower()] for col_name in df.columns }
        markermap = {col_name:marker[col_name.split(", ")[-1].lower().strip()] for col_name in df.columns}

        for col in df:
            column = df[col]
            values = column.iloc[::2]
            # ax.plot(column.index, column.values, linewidth=2.0, color=colormap[col], alpha=0.8)
            ax.plot(values.index, values.values, linewidth=2.0, color=colormap[col], alpha=0.8)
            x = values.index
            ax.scatter(values.index, values.values, linewidth=2.0,  marker=markermap[col],  color=colormap[col],
                       alpha=0.8, label=col, s=50)
    
    
    
    ax.set_xlim(0, xmax)
    if dataset in ['emnist-mod5', 'emnistvcd']:
        ax.set_xlim(0,150)
    
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = np.array(handles), np.array(labels)
    labels = np.array([display_name_map[l.lower()] for l in labels])
                 
    
    ordering = np.argsort([l for l in labels])
    lgd = ax.legend(handles[ordering], labels[ordering], bbox_to_anchor=bbox_to_anchor,ncol=ncol,
                   fontsize=14)
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if savedir is None:
        plt.suptitle(dataset.upper())
    else:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(f"{savedir}/{dataset}_clusterplot{plot_qualifier}.jpg", 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', format='jpeg')
    plt.show()