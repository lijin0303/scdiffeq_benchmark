'''
Utils for perturb & visualize scDiffEq
'''
from neural_diffeqs import neural_diffeq
from os.path import exists
import torch
import matplotlib.pyplot as plt
from anndata import read_h5ad
import vinplots
import umap
import numpy as np
from torchsde import sdeint
import pickle 
import sklearn
import warnings
import pandas as pd
import nmslib
from collections import Counter
from operator import itemgetter
import seaborn as sns
from torchsde import BrownianInterval
from tqdm import tqdm
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 200

def mk_plot(c=3):
    fig = vinplots.Plot()
    fig.construct(nplots=c, ncols=c)
    fig.modify_spines(ax="all", spines_to_delete=['top', 'right', 'bottom', 'left'])
    axes =[ fig.AxesDict[0][i] for i in range(c)]
    return fig, axes

def zperturb(ann,idx,pca,zs,perturb_genes="Klf4;Nr4a1"): 
    # perturb expression at original space
    x = ann[idx].X
    scaler=sklearn.preprocessing.StandardScaler()
    x_ =scaler.fit_transform(x.toarray())
    PGenes=perturb_genes.split(";")
    x_[:,ann.var.gene_id.isin(PGenes)] = zs
    # embed perturbed cells into fitted PCA space
    xp = pca.transform(x_)
    return xp

def prog_umap(umap_model,adata,idx,diffeq,perturb=False,pcamodel= None,zs=None,pg="Klf4;Nr4a1"):
    device = "cpu"
    X_umap = umap_model.transform(adata.obsm['X_pca'])
    if not perturb:
        startC = adata[idx].obsm['X_pca']
    else:
        message="Running perturbations of " + pg + " " + "w/ z-score of " + str(zs)
        print(message)
        startC = zperturb(adata,idx,pcamodel,zs,perturb_genes=pg)
    X0 = torch.Tensor(startC).to(device)
    t = torch.Tensor([0, 0.01, 0.02]).to(device)
    diffeq = diffeq.to(device)
    with torch.no_grad():
        X_pred = sdeint(diffeq, X0, t)
    X_pred_ = X_pred.detach().cpu().numpy()
    X_pred_umap = [umap_model.transform(Xp) for Xp in X_pred_]
    ### Plotting
    fig,axes = mk_plot()
    cs = "darkgreen" if perturb else "darkred"
    for n, xpu in enumerate(X_pred_umap):
        axes[n].scatter(X_umap[:,0], X_umap[:,1], c="lightgrey", zorder=0)
        axes[n].scatter(xpu[:,0], xpu[:,1], zorder=n+2, c=cs, s=10)
        axes[n].set_xticks([])
        axes[n].set_yticks([])
        axes[n].set_title("Day: {}".format(int(2 + 2*n)))

def sde_simu(adata,idx,pcamodel,zs,diffeq,perturb,pg="Klf4;Nr4a1",device="cpu",ts =[0, 0.01, 0.02]):
    if not perturb:
        startC = adata[idx].obsm['X_pca']
    else:
        startC = zperturb(adata,idx,pcamodel,zs,perturb_genes=pg)
    X0 = torch.Tensor(startC).to(device)
    t = torch.Tensor(ts).to(device)
    diffeq = diffeq.to(device)
    with torch.no_grad():
        X_pred = sdeint(diffeq, X0, t)
    X_pred_ = X_pred.detach().cpu().numpy()
    return X_pred_

def NN_Build(adata):
    NN_buildD = adata.obsm['X_pca']
    # Initialize a new index
    # Use HNSW index on Cosine Similarity based on original data
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(NN_buildD)
    index.createIndex({'post': 2}, print_progress=False)
    # Use the annotation for cell type classification
    NN_annotD = adata.obs
    NN_annotD["NewAnnot"] = NN_annotD.Annotation.astype(str)
    NN_annotD.loc[~NN_annotD.Annotation.isin(
    ["undiff", "Neutrophil", "Monocyte"]),"NewAnnot"] = "Others"
    NN_buildDf = pd.DataFrame(NN_buildD,NN_annotD.NewAnnot)
    return index,NN_buildD,NN_annotD,NN_buildDf

def NN_Classify(index,NN_buildDf,NN_findD,n_neighbor):
    CountList = []
    for snap in NN_findD: 
        # Batch query nearest neighbours for simulated data at each tp
        neighbours = index.knnQueryBatch(snap,k=n_neighbor,num_threads=4)
        LabelCount = [Counter(NN_buildDf.iloc[i].index).most_common(2)[0][0] for i in neighbours]
        CountList.append(Counter(LabelCount))
    return CountList

def CellType_Stack(countl,pt = 'Unperturbed cells'):
    plt.figure(figsize=(8,8))
    sns.set_context("talk")
    labels=["undiff", "Neutrophil", "Monocyte", "Others"]
    colors=["darkgrey", "orange", "green", "crimson"]
    udC,NC,MonoC,OthrC = [list(map(itemgetter(l), countl)) for l in labels]
    plt.stackplot(np.arange(len(udC)),
                  udC,NC,MonoC,OthrC,
                  labels=labels, colors=colors)
    plt.legend(loc='lower left', prop={'size': 14})
    plt.margins(0,0)
    plt.title(pt)
    plt.xlabel(f"Time from day 2 to 6 (Steps = {len(countl)})")
    plt.ylabel("Count of cell type")
    plt.show()
    
    
def continum_celltype(adata,idx,diffeq,steps = 20,perturb=False,pg="Klf4;Nr4a1",pcamodel= None,zs=None,nn=10):
    if perturb:
        message="Running perturbations of " + pg + " " + "w/ z-score of " + str(zs)
        print(message)
    NN_findD = sde_simu(adata,idx,pcamodel,zs,diffeq,perturb,pg,
                        ts = np.linspace(0,0.02,steps))
    indexNN,_,_,NN_buildDf = NN_Build(adata)
    classedCT = NN_Classify(indexNN,NN_buildDf,NN_findD,nn)
    if perturb:
        pltTitle = "Perturbed " + pg + " " + "w/ z-score of " + str(zs)
        CellType_Stack(classedCT,pt=pltTitle)
    else:
        CellType_Stack(classedCT)
    return classedCT


def batch_mvp(m, v):
    return torch.bmm(m, v.unsqueeze(-1)).squeeze(dim=-1)

def sdeCompare(sde,y0,t,r=20,a=0.5):
    with torch.no_grad():
        batch_size, state_size, brownian_size = sde.g(None,y0).shape
    yp_simu = []
    for _ in tqdm(range(r)):
        with torch.no_grad():
            yp = sdeint(sde, y0, t)
            yp_simu.append(yp.reshape(-1,state_size).detach().numpy())
    ypi = np.stack(yp_simu,1)
    ypi2 = sde_decomp(sde,y0,t,s="fdt+gW",r=r)
    ypi3 = sde_decomp(sde,y0,t,s="fdt+W",r=r,a=a)
    ypi4 = sde_decomp(sde,y0,t,r=r)
    return [ypi,ypi2,ypi3,ypi4]

def sde_decomp(sde,y0,t,s="fdt",r=20,a=0.5):
    with torch.no_grad():
        batch_size, state_size, brownian_size = sde.g(None,y0).shape
    yp_simu = []
    for _ in tqdm(range(r)):
        yl = [y0]
        yc = y0
        with torch.no_grad():
            for i in range(len(t)-1):
                bm = BrownianInterval(t0=t[0],
                                      t1=t[-1], 
                                      size=(batch_size, brownian_size),
                                      device='cpu')
                if s=="fdt+gW":
                    noise = batch_mvp(sde.g(None,yc), bm(t[i],t[i+1]))
                elif s=="fdt+W":
                    noise = torch.tensor(np.random.randn(batch_size, state_size)*a*np.sqrt((t[i+1]-t[i]).numpy()))
                elif s=="fdt":
                    noise = torch.tensor(np.zeros((state_size)))   
                yn = sde.f(None,yc)*(t[i+1]-t[i]) + noise.float() + yc
                yc = yn
                yl.append(yn)
            yp_simu.append(np.stack(yl).reshape(-1,state_size))
    ypi = np.stack(yp_simu,1)
    return ypi


def UampSDE(adata,umap_model,l,labels):
    device = "cpu"
    X_umap = umap_model.transform(adata.obsm['X_pca'])
    _,r,state_size = l[0].shape
    X_pred_umap = [umap_model.transform(yp.reshape(-1,state_size)) for yp in l]
    fig,axes = mk_plot(c = len(l))
    for n, xpu in enumerate(X_pred_umap):
        axes[n].scatter(X_umap[:,0], X_umap[:,1], c="lightgrey", zorder=0,alpha=0.22)
        axes[n].scatter(xpu[:,0],xpu[:,1], c=np.arange(xpu.shape[0]), 
            marker="*",cmap="viridis",s=45)
        axes[n].scatter(xpu[:r, 0],xpu[:r, 1],s=45,marker="*",color="darkred")
        axes[n].set_xticks([])
        axes[n].set_yticks([])
        axes[n].set_title(labels[n])
    return axes
 