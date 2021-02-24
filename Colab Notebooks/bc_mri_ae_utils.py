import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import umap
from sklearn.decomposition import PCA

def getCaseMatch(dx_idx,dfs,do_plot=False,control='non-familial-control'):
    ''' takes in IDs of ASD subjects, gives back IDs of TDs matched on age and gender'''
    ii = list()
    dfsc = dfs.copy();
    for dx_ind in range(sum(dx_idx)):
        dfsc = dfsc.set_index(np.arange(len(dfsc)))
        dx_age = dfsc['age_years'].values[dx_idx]
        dx_sex = dfsc['sex'].values[dx_idx]
        idxs = np.arange(len(dfs))
        v1 = dfsc['family_type'].values==control
        v2 = dfsc['sex'].values == dx_sex[dx_ind]
        v_dx = dfsc['clinical_asd_dx'].values !='1'
        
        v3 = abs(dfsc['age_years'].values[v1*v2*v_dx]-dx_age[dx_ind])

        match_arr = idxs[v1*v2][np.argsort(v3)]
        match_arr = np.array(match_arr)
        match_arr = match_arr[np.array([m not in ii for m in match_arr])]
        i = match_arr[0]
        ii.append(i)

    caseMatch_idx = ii
    caseMatch_idx.sort()
    caseMatch_idx
    assert len(caseMatch_idx)==len(np.unique(caseMatch_idx)),'non unique elements'
    if do_plot==True:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.hist(dfs['sex'].values[caseMatch_idx],alpha=.3)
        plt.hist(dfs['sex'].values[dx_idx],alpha=.3)

        plt.subplot(1,3,2)
        plt.hist(dfs['age_years'].values[caseMatch_idx],alpha=.3)
        plt.hist(dfs['age_years'].values[dx_idx],alpha=.3)

        plt.subplot(1,3,3)
        plt.hist(dfs['family_type'].values[caseMatch_idx],alpha=.3)
        plt.hist(dfs['family_type'].values[dx_idx],alpha=.3)
        
    return caseMatch_idx

def dim_reduce(z,method='UMAP'):
    
    if method=='UMAP':
        reducer = umap.UMAP()
    else:
        reducer = PCA(n_components=2)
        
    tiny = reducer.fit_transform(z)
    
    return tiny



def project_data(data,dxArr,lbls,legend=True):
    plt.figure(figsize=(12,4));
    plt.subplot(1,3,1);
    
    z_mean, z_log_var, z = z_encoder(data[:,:,:,:]);
    s_mean, s_log_var, s = s_encoder(data[:,:,:,:]);
    v_mean, v_log_var, v = encoder(data[:,:,:,:]);
    
    if z.shape[1]>2:
        print('reducing dimensionality')
        z = dim_reduce(z)
        s = dim_reduce(s)
        v = dim_reduce(v)
    
    x = z;x = np.array(x);
    mark = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.','b*', 'g*', 'r*', 'c*', 'm*', 'y*', 'k*','bx', 'gx', 'rx', 'cx', 'mx', 'yx', 'kx']
    for i in range(len(np.unique(dxArr))):
        ii = np.unique(dxArr)[i];
        plt.plot(x[dxArr==ii,0],x[dxArr==ii,1],mark[int(i)],markersize=15,alpha=.3);
        
    if legend:
        plt.legend(lbls);
    plt.title('CAE background');

    plt.subplot(1,3,2)
    x = s;
    x = np.array(x);

    for i in range(len(np.unique(dxArr))):
        ii = np.unique(dxArr)[i];
        plt.plot(x[dxArr==ii,0],x[dxArr==ii,1],mark[int(i)],markersize=15,alpha=.3);
    if legend:
        plt.legend(lbls);
    plt.title('CAE salient');

    plt.subplot(1,3,3);
    
    x = v;
    x = np.array(x);

    for i in range(len(np.unique(dxArr))):
        ii = np.unique(dxArr)[i];
        plt.plot(x[dxArr==ii,0],x[dxArr==ii,1],mark[int(i)],markersize=15,alpha=.3);
    if legend:
        plt.legend(lbls);
    plt.title('VAE');
    
    if len(np.unique(dxArr))>1:
        plt.figure(figsize=(6,4));
        plt.bar([0,1,2],[silhouette_score(z,dxArr),silhouette_score(s,dxArr),silhouette_score(v,dxArr)]);
        plt.xticks([0,1,2],labels=['CAE background','CAE salient','Vae']);
        plt.title('Silhouette score')
        plt.ylim(0,1)
        
        
def plot_cscatter(dxArr,v=None,clbl=None,clim=None):
    
    if type(v)==type(None):
        v = np.arange(len(df))
    
    space_lbls = ['background','salient','VAE']
    spaces = [space_bg_abide, space_sl_abide, space_vae_abide]
    sub_slices = [patients,patients,patients]
    
    plt.figure(figsize=(15,5))
    for i in range(3):
        plt.subplot(1,3,i+1)
        space = spaces[i]
        v = sub_slices[i]
        plt.scatter(space[v,0],space[v,1],c=dxArr[v])
        if clim:
            plt.clim(clim[0],clim[1])
        cbar = plt.colorbar()
        #cbar.ax.set_ylabel(clbl, rotation=270)
        cbar.ax.set_ylabel(clbl, rotation=270,labelpad=20,fontsize=16,fontweight='bold')
        plt.title(space_lbls[i],fontsize=16,fontweight='bold')

    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=.3,
        hspace=None,)        
    
    
def numerize(vec):
    vec = np.array(vec)
    numVec = np.zeros(vec.shape[0])
    u = np.unique(vec)
    n_u = len(u)
    for i in range(n_u):
        numVec[vec==u[i]]=i

    return numVec