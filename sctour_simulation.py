'''
Utils for testing sctour
'''
from scipy.spatial import distance
from torchdiffeq import odeint
import umap
import numpy as np
import torch
import matplotlib.pyplot as plt
import anndata as ann
import pickle
import sys  
import sctour as sct
sys.path.insert(0, '/Users/ruitong/scDiffEq')
from functions import *

device = "cpu"

def sctour_train(annPath,seed = 123,subset = None,tier = "Tier1",latentN = 30):
    ### 20k sampling
    adata = ann.read_h5ad(annPath)
    if subset is not None:
        np.random.seed(seed)
        repSample = [np.random.choice(adata.obs[adata.obs["train"]].index,
                                      20_000, replace=False) for i in range(10)]
        sample_idx = repSample[subset]
        adata_T1 = adata[sample_idx].copy()
        adata_T1.obs = adata_T1.obs.reset_index()
    else:
        adata_T1 = adata
    adata_T1 = adata_T1[:,adata_T1.var[f"highly_variable_{tier}"]]
    if subset is not None:
        adata_T1.write_h5ad(f"../ProcData/Weinreb2020_TrainSet{subset}.h5ad")
    tnode = sct.train.Trainer(adata_T1, loss_mode='nb',nepoch=400,n_latent=latentN)
    tnode.train()
    pickle.dump(tnode, file = open(f"tnode_l{latentN}.pickle", "wb"))
    

def postTrain(annPath,model,alpha=0.8):
    adata_T1 = ann.read_h5ad(annPath)
    tnode = pickle.load(open(model, "rb"))
    adata_T1.obs['ptime'] = sct.train.reverse_time(tnode.get_time())
    mix_zs,zs,pred_zs = tnode.get_latentsp(alpha_z=(1-alpha), alpha_predz=alpha)
    adata_T1.obsm['X_TNODE'] = mix_zs
    DR(adata_T1)
    adata_T1.obsm['X_VF'] = tnode.get_vector_field(adata_T1.obs['ptime'].values, adata_T1.obsm['X_TNODE'])
    adata_T1_sorted = adata_T1[np.argsort(adata_T1.obs['ptime']),:]
    DR(adata_T1_sorted,rep="X_TNODE")
    adata_T1_sorted.obsm['X_VF'] = tnode.get_vector_field(adata_T1_sorted.obs['ptime'].values,
                                                          adata_T1_sorted.obsm['X_TNODE'])
    
    return adata_T1,adata_T1_sorted

def umapSeries(ann):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))
    sc.pl.umap(ann, color='Annotation', size=10, ax=axs[0], legend_loc='on data', show=False,legend_fontoutline=3)
    sc.pl.umap(ann, color='ptime', size=20, ax=axs[1], show=False,legend_fontoutline=3)
    sct.vf.plot_vector_field(ann, zs_key='TNODE', vf_key='VF', use_rep_neigh='TNODE',reverse=True,
                             color='Annotation', ax=axs[2], legend_loc='none', frameon=False, size=100, alpha=0.2)
    plt.show()
    
def sctour_predict(annPath,model,alpha=0.8):
    adata = ann.read_h5ad(annPath)
    adata_test = adata[adata.obs["test"]].copy()
    if isinstance(model, str): model = pickle.load(open(model, "rb"))
    test_pred_t, test_pred_ltsp, _, _ = model.predict_time(adata_test, alpha_z=(1-alpha), 
                                                         alpha_predz=alpha, mode='coarse')
    adata_test.obs['ptime'] = 1-test_pred_t
    adata_test.obsm['X_TNODE'] = test_pred_ltsp
    DR(adata_test)

    adata_test.obsm['X_VF'] = model.get_vector_field(adata_test.obs['ptime'].values, adata_test.obsm['X_TNODE'])
    adata_test_sorted = adata_test[np.argsort(adata_test.obs['ptime']),:]
    DR(adata_test_sorted,rep="X_TNODE")
    adata_test_sorted.obsm['X_VF'] = model.get_vector_field(adata_test_sorted.obs['ptime'].values,
                                                          adata_test_sorted.obsm['X_TNODE'])
    
    return adata_test,adata_test_sorted

def sctour_test(testPath,anntest,model,prefix="l30",ns=80):
    TrueTest = pickle.load(open(testPath, "rb"))
    TestIndex = TrueTest[TrueTest].index
    if isinstance(model, str): model = pickle.load(open(model, "rb"))
    Fates = [chain_simu(anntest,l,model,ns=ns,plotmute=False) for l in TestIndex]
    pickle.dump(Fates, file = open(f"TestFate_Prediction_{prefix}.pickle", "wb"))
    SubTest = anntest.obs.loc[TestIndex].copy()
    SubTest["fate_sctour"] = Fates
    fbcor = SubTest[["fate_sctour","neu_vs_mo_percent"]].corr()
    TestClone = anntest.obs.loc[TestIndex,"clone_idx"]
    dfTest = anntest.obs[anntest.obs.clone_idx.isin(TestClone)]
    sumDF = dfTest.groupby(["clone_idx","Time point","Annotation"]).size()
    ByCloneCor = dfTest.groupby("clone_idx")[['Time point','ptime']].corr().unstack().iloc[:,1].dropna()
    return fbcor,ByCloneCor,SubTest,sumDF
    
     
def vec_dist(a,vec):
    return distance.euclidean(a,vec[-1])
def Simulate(D,trainedM,ind,n):
    z0 = D.obsm['X_TNODE'][ind]
    model = trainedM.get_model(None)
    t_init = D.obs["ptime"][ind]
    tstep = D.obs.ptime[D.obs.ptime<t_init].to_numpy()
    t_np = np.random.choice(tstep,replace=False,size=n)
    t_np = np.flip(np.sort(t_np)).copy()
    t = torch.Tensor(t_np).to(device)
    if not ((t[1:] > t[:-1]).all())|((t[1:] < t[:-1]).all()):
        t_np = np.linspace(t_init,0,n)
        t = torch.Tensor(t_np).to(device)
    z0 = torch.Tensor(z0).to(device)
    pred_z = odeint(model.lode_func,z0,t,method = model.ode_method).view(-1, model.n_latent)
    x_pred_ = pred_z.detach().cpu().numpy()
    euDist = np.apply_along_axis(vec_dist,1,D.obsm['X_TNODE'],vec=x_pred_)
    idx = np.argpartition(euDist, 50)
    FateSeries = D.obs[idx<50].Annotation.value_counts()[["Neutrophil","Monocyte"]]
    return x_pred_,FateSeries
def fateBias(D,ind,trainedM,n,r=25):
    x_pred_ = Simulate(D,trainedM,ind,n)[0]
    FateSeries = sum([Simulate(D,trainedM,ind,n)[1] for k in range(r)])
    fateBias = ((FateSeries[0])/(FateSeries[0]+FateSeries[1]+1)).round(3)
    return fateBias,FateSeries,x_pred_
def trajectory_simu(D,vec,pickelUMAP=None,sort=False):
    if sort:
        D = D[np.argsort(D.obs['ptime'].values), :]
    if pickelUMAP is None:
        testUMAP = umap.UMAP(random_state=42)
        vae_embedding_umap = testUMAP.fit_transform(D.obsm['X_TNODE'])
    else:
        testUMAP = pickle.load(open(pickelUMAP, "rb"))
        vae_embedding_umap = testUMAP.transform(D.obsm['X_TNODE'])
    vae_embedding_progress_umap = testUMAP.transform(vec)
    plt.figure(figsize=(12, 6.4))
    plt.scatter(vae_embedding_umap[:, 0], 
                vae_embedding_umap[:, 1], 
                c=np.array(D.uns["Annotation_colors"])[D.obs.Annotation.cat.codes.to_numpy()],
                s=1, 
                marker="*",
                alpha=0.22)
    plt.scatter(vae_embedding_progress_umap[:, 0], 
                vae_embedding_progress_umap[:, 1], 
                c=np.arange(vae_embedding_progress_umap.shape[0]), 
                marker="*",
                cmap="viridis",
                s=45
               )
    plt.colorbar()
    # Plot simulated path
    plt.scatter(vae_embedding_progress_umap[0, 0], 
                vae_embedding_progress_umap[0, 1], 
                s=45, 
                marker="*",
                color="red")
    plt.gca().set_aspect('equal', 'datalim')
    plt.xticks([]),plt.yticks([])
    plt.show()
    
def chain_simu(D,label,trainM,ns,plotmute=True):
    print("Run for test cell: "+label)
    ind_check = D.obs.index.get_loc(label) 
    fateB,fateo,X_pred = fateBias(D,ind_check,trainM,ns)
    print("Steps simulated: "+"{:.1f}".format(len(X_pred)))
    if plotmute:
        formatted_float = "{:.3f}".format(fateB)
        print('Fate Bias:' + formatted_float)
        display(fateo)
        trajectory_simu(D,X_pred,sort=False)
    else:
        return fateB