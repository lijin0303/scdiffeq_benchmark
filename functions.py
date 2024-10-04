'''
Utils for running cell dynamics inference
'''
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import anndata as ann
import cell_fate as cf
import torch
import pickle
from os.path import exists
from IPython.display import set_matplotlib_formats
warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "FALSE" 

set_matplotlib_formats('png')
plt.rcParams['figure.dpi'] = 70
plt.rcParams['savefig.dpi'] = 300
device = "cpu"


# Read in h5ad and pp
def h5adpp(h5ad_path,seed=123,split=2):
    adata = cf.io.read_h5ad(h5ad_path)
    cf.pp.annotate_train_test(adata)
    cf.pp.annnotate_unique_test_train_lineages(adata)
    adata_train = adata[adata.obs["train"]].copy()
    np.random.seed(seed)
    repSample = [np.random.choice(adata_train.obs.index, 20_000, replace=False) for i in range(10)]
    sample_idx = repSample[split]
    adata_T1 = adata_train[sample_idx].copy()
    adata_T1.obs = adata_T1.obs.reset_index()
    # Standard qc metrics and filtering, select 2000 most variable genes
    sc.pp.calculate_qc_metrics(adata_T1, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_genes(adata_T1, min_cells=20)
    sc.pp.highly_variable_genes(adata_T1, flavor='seurat_v3', n_top_genes=2000, subset=False)
    # Log transformation for negative binomial distribution
    adata_T1.X = np.log1p(adata_T1.X)
    return adata_T1


def quickHist(S):
    S.plot.hist(grid=True, bins=80, rwidth=1.2,color='#607c8e')
    plt.title('Distribution of input series')
    plt.xlabel('Counts')
    plt.ylabel('Pandas Series')
    plt.grid(axis='y', alpha=0.75)
    
    
    
def DR(ad,nn=50,res=0.1,rep="X_pca"):
    if not ('X_pca' in ad.obsm): sc.pp.pca(ad)
    sc.pp.neighbors(ad, n_neighbors=nn,use_rep=rep) ### build KNN graph
    sc.tl.umap(ad, min_dist=res)### Generate UMAP embedding with latent space