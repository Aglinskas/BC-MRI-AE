import os
import numpy as np
from matplotlib import pyplot as plt
import umap
from IPython import display
import time

def get_weights(fdir=None):
    if not fdir:
        fdir = '/mmfs1/data/aglinska/tf_outputs/CVAE/'
    items = os.listdir(fdir)
    items = [item.split('.')[0] for item in items]
    items = [item.replace('_loss','') for item in items]
    items = np.unique(items)
    items=items[items!='']
    items.sort()

    for i in range(len(items)):
        print(f"{i:02d} | {items[i]}")
        
    return items


def cscatter(spaces,v=None,c=None,clim=None,clbl=None,legend=None):
    space_lbls = ['Background','Salient','VAE']

    if type(v)==type(None):
        v = np.repeat(True,len(spaces[0]))
        
    plt.figure(figsize=(12,4))
    for i in range(len(spaces)):
        plt.subplot(1,3,i+1)
        
        if type(c)!=type(None) and len(np.unique(c)) > 10: # continus colourbar
            #print('continuues colourbar')
            
            plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
            if type(clim)==type(None): #if clim not passed, 
                clim = (min(c),max(c)) # calc min max
            plt.clim(clim[0],clim[1]) # do clim regardless
                
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(clbl,rotation=270,labelpad=20,fontsize=16,fontweight='bold')    
                
        elif type(c)!=type(None) and len(np.unique(c)) < 10: # categorical colourbar
            #print('categorical colourbar')
            for j in np.unique(c):
                plt.scatter(spaces[i][c[v]==j,0],spaces[i][c[v]==j,1],alpha=.5)
                    
            if type(legend)==type(None):
                legend = [str(i) for i in np.unique(c)]    
            plt.legend(legend)

        else:
           #print('else')
            plt.scatter(spaces[i][v,0],spaces[i][v,1])
            
        
        #plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
        plt.xlabel('latent dim. 1');plt.ylabel('latent dim. 2')
        plt.title(space_lbls[i])

    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=.3,hspace=None,) 
    print(sum(v))
    
    
def get_batch_idx(df,batch_size = 64):

    sub_scan_site = df['ScanSite'].values
    scanning_sites = np.unique(sub_scan_site)

    nsites = len(scanning_sites)

    this_site = np.random.randint(low=0,high=nsites)


    site_asd = (sub_scan_site==scanning_sites[this_site]) * (df['DxGroup'].values==1)
    site_td = (sub_scan_site==scanning_sites[this_site]) * (df['DxGroup'].values==2)

    asd_idx = np.nonzero(site_asd)[0]
    td_idx = np.nonzero(site_td)[0]

    while len(asd_idx) < batch_size: #if not enough copy over
        asd_idx = np.hstack((asd_idx,asd_idx))

    while len(td_idx) < batch_size: #if not enough copy over
        td_idx = np.hstack((td_idx,td_idx))

    assert len(np.unique(df.iloc[asd_idx]['Subject Type'].values)),'subject batch selection messed up'
    assert len(np.unique(df.iloc[asd_idx]['ScanSite'].values)),'subject batch selection messed up'
    assert len(np.unique(df.iloc[asd_idx]['ScannerType'].values)),'subject batch selection messed up'

    assert len(np.unique(df.iloc[td_idx]['Subject Type'].values)),'subject batch selection messed up'
    assert len(np.unique(df.iloc[td_idx]['ScanSite'].values)),'subject batch selection messed up'
    assert len(np.unique(df.iloc[td_idx]['ScannerType'].values)),'subject batch selection messed up'
    
    assert ~any([a in td_idx for a in asd_idx]),'you f***ed up'
    assert ~any([t in asd_idx for t in td_idx]),'you f***ed up'
    
    np.random.shuffle(asd_idx)
    np.random.shuffle(td_idx)

    asd_idx = asd_idx[0:batch_size]
    td_idx = td_idx[0:batch_size]

    return asd_idx,td_idx


def dim_reduce(z,method='UMAP'):
    
    if method=='UMAP':
        reducer = umap.UMAP()
        #reducer = ParametricUMAP()
    else:
        reducer = PCA(n_components=2)
        
    tiny = reducer.fit_transform(z)
    
    return tiny

def cscatter(spaces,v=None,c=None,clim=None,clbl=None,legend=None):
    space_lbls = ['Background','Salient','VAE']

    if type(v)==type(None):
        v = np.repeat(True,len(spaces[0]))
        
    plt.figure(figsize=(12,4))
    for i in range(len(spaces)):
        plt.subplot(1,3,i+1)
        
        if type(c)!=type(None) and len(np.unique(c)) > 10: # continus colourbar
            #print('continuues colourbar')
            
            plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
            if type(clim)==type(None): #if clim not passed, 
                clim = (min(c),max(c)) # calc min max
            plt.clim(clim[0],clim[1]) # do clim regardless
                
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(clbl,rotation=270,labelpad=20,fontsize=16,fontweight='bold')    
                
        elif type(c)!=type(None) and len(np.unique(c)) < 10: # categorical colourbar
            #print('categorical colourbar')
            for j in np.unique(c):
                plt.scatter(spaces[i][c[v]==j,0],spaces[i][c[v]==j,1],alpha=.5)
                    
            if type(legend)==type(None):
                legend = [str(i) for i in np.unique(c)]    
            plt.legend(legend)

        else:
           #print('else')
            plt.scatter(spaces[i][v,0],spaces[i][v,1])
            
        
        #plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
        plt.xlabel('latent dim. 1');plt.ylabel('latent dim. 2')
        plt.title(space_lbls[i])

    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=.3,hspace=None,) 
    #print(sum(v))
    plt.show()
    
    
def get_spaces(ABIDE_data,z_encoder,s_encoder,w=2):
    
    encs = [z_encoder.predict, s_encoder.predict]
    bg_space = np.array(encs[0](ABIDE_data)[w])
    sl_space = np.array(encs[1](ABIDE_data)[w])

    if bg_space.shape[1]>2:
        bg_space = dim_reduce(bg_space,method='UMAP')
        sl_space = dim_reduce(sl_space,method='UMAP')
    return bg_space,sl_space


def plot_sweep(ABIDE_data,z_encoder,s_encoder,cvae_decoder,wspace='z',l=5,w=2):

    z = z_encoder.predict(ABIDE_data)
    s = s_encoder.predict(ABIDE_data)

    z_lin = np.linspace(z[2].min(axis=0),z[2].max(axis=0),l)
    s_lin = np.linspace(s[2].min(axis=0),s[2].max(axis=0),l)
    
    nrows = l;ncols = l;c = 0
    
    for i in range(l):
        for j in range(l):
            c+=1
            plt.subplot(nrows,ncols,c)
            vec_z = z_lin[i,:]
            vec_s = s_lin[i,:]
            vec_0 = np.zeros(vec_s.shape)

            if wspace=='z':
                vec3 = np.hstack((vec_z,vec_0))
            elif wspace=='s':
                vec3 = np.hstack((vec_0,vec_s))
            else:
                #vec3 = np.hstack((vec_z,vec_s))
                vec3 = np.hstack((vec_z,vec_s))

            plt.imshow(cvae_decoder.predict(np.vstack((vec3,vec3)))[0,:,:,32,0])
            plt.xticks([]);plt.yticks([]);
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    
def plot_four(DX_batch,TD_batch,z_encoder,s_encoder,cvae_decoder,cvae,idx=0):
    
    #im_in = [DX_batch,TD_batch][np.random.choice([0,1])]
    im_in = [DX_batch,TD_batch][idx]
    _zeros = np.zeros(s_encoder(im_in)[2].shape)

    v = 2
    cvae_sal_vec = np.hstack((_zeros,s_encoder(im_in)[v]))
    cvae_bg_vec = np.hstack((z_encoder(im_in)[v],_zeros))
        
    plt.figure(figsize=np.array((4*4,4))*.5)
    s = 11;k=32
    plt.subplot(1,4,1)
    plt.imshow(im_in[s,:,:,k]);plt.xticks([]);plt.yticks([]);plt.title('input')

    plt.subplot(1,4,2)
    #plt.imshow(cvae_decoder(cvae_full_vec)[s,:,:,k,0]);plt.xticks([]);plt.yticks([]);plt.title('reconstruction')
    plt.imshow(cvae.predict([DX_batch,TD_batch])[idx][s,:,:,k,0]);plt.xticks([]);plt.yticks([]);plt.title('reconstruction')

    plt.subplot(1,4,3)
    plt.imshow(cvae_decoder(cvae_sal_vec)[s,:,:,k,0]);plt.xticks([]);plt.yticks([]);plt.title('salient')

    plt.subplot(1,4,4)
    plt.imshow(cvae_decoder(cvae_bg_vec)[s,:,:,k,0]);plt.xticks([]);plt.yticks([]);plt.title('background')
    
    plt.show()
    
    
def str_to_ordinal(inVec):
    #inVec = df['ScannerType'].values
    u = np.unique(inVec)
    evec = np.zeros(inVec.shape)
    for i in range(len(u)):
        evec[inVec==u[i]]=i
        
    return evec


def plot_cvae_silhouettes(ABIDE_data,z_encoder,s_encoder,sub_slice,keys=None,l=5,w=2):
    from sklearn.metrics import silhouette_score
    
    z = z_encoder.predict(ABIDE_data)[w]
    s = s_encoder.predict(ABIDE_data)[w]
    
    
    if not keys:
        keys = ['ADOS_Total','ADOS_Social','DSMIVTR','AgeAtScan','ScannerID','ScanSiteID','FIQ']
    
    #l = 5 # How many bings
    arr = np.zeros((len(keys),2))
    mat = np.zeros((len(keys),l))
    for i in range(len(keys)):
        vec = df[keys[i]].values
        v = (vec!=-9.999e+03) * ~pd.isnull(vec) * sub_slice
        vecv = vec[v]
        vecv = np.digitize(vec[v],np.linspace(vec[v].min(),vec[v].max(),5))
        arr[i,0]= silhouette_score(z[v],vecv)
        arr[i,1]= silhouette_score(s[v],vecv)

    plot_df = pd.DataFrame(arr,index=keys,columns=['BG','SL'])
    ax = plot_df.plot.bar(rot=45)
    ax.legend()

    plt.show()
    
    
def get_sil_diff(ABIDE_data,z_encoder,s_encoder,sub_slice,keys=None,w=2):
    
    z = z_encoder.predict(ABIDE_data)[w]
    s = s_encoder.predict(ABIDE_data)[w]

    if not keys:
        keys = ['ADOS_Total','ADOS_Social','DSMIVTR','AgeAtScan','ScannerID','ScanSiteID','FIQ']
        #keys = ['AgeAtScan','ScannerID','ScanSiteID','FIQ']

    arr = np.zeros((len(keys),2))
    #mat = np.zeros((len(keys),l))
    for i in range(len(keys)):
        vec = df[keys[i]].values
        v = (vec!=-9.999e+03) * ~pd.isnull(vec) * patients
        vecv = vec[v]
        vecv = np.digitize(vec[v],np.linspace(vec[v].min(),vec[v].max(),5))
        arr[i,0]= silhouette_score(z[v],vecv)
        arr[i,1]= silhouette_score(s[v],vecv)

    dif = arr[:,1]-arr[:,0]
    
    return dif


def plot_cvae_dif_mat(ABIDE_data,z_encoder,s_encoder,sub_slice,keys=None):
    
    if not keys:
        keys = ['ADOS_Total','ADOS_Social','DSMIVTR','AgeAtScan','ScannerID','ScanSiteID','FIQ']
        
    dmat = np.array([get_sil_diff(ABIDE_data,z_encoder,s_encoder,sub_slice,keys=keys) for i in range(10)])

    ys = dmat.mean(axis=0)
    yerr = dmat.std(axis=0)
    xs = np.arange(dmat.shape[1])

    plt.bar(xs,ys);
    plt.errorbar(xs,ys,yerr,fmt='r.');
    plt.xticks(xs,labels=keys,rotation=45);
    plt.ylabel('SL-BG Silhouette Difference')
    plt.show()
    
    
# Progress Plotting Functions
def net_query():
    i = 0
    n = 50
    v_sl = s_encoder.predict(ABIDE_data[0:n,:,:,:])[i]#[0,:]
    v_bg = z_encoder.predict(ABIDE_data[0:n,:,:,:])[i]#[0,:]
    v = np.hstack((v_sl,v_bg))
    latent_vec = v;
    out = cvae_decoder.predict(latent_vec)

    im = out[:,:,:,:,0]
    im1 = ABIDE_data[0:n,:,:,:]
    ss = ((im-im1)**2).sum()

    return im[0,:,:,40],im1[0,:,:,40],ss

def net_plot(im,im1):
    plt.subplot(1,2,1);
    plt.imshow(im1);
    plt.subplot(1,2,2);
    plt.imshow(im);

def plot_trainProgress(loss,im,im1):

    display.clear_output(wait=True);
    display.display(plt.gcf());
    #time.sleep(1.0)

    plt.figure(figsize=np.array((7,5)) );

    plt.subplot(2,2,1);
    plt.imshow(im1);plt.xticks([]);plt.yticks([]);
    plt.title('image')

    plt.subplot(2,2,3);
    plt.imshow(im);plt.xticks([]);plt.yticks([]);
    plt.title('reconstruction')

    # Last 1000
    plt.subplot(2,2,2);
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    ss = int(len(loss)*1/3)
    plt.plot(loss[ss::],alpha=.3)
    plt.plot(moving_average(loss[ss::], 100))

    # Last 100
    plt.subplot(2,2,4);
    n = 1000
    if len(loss)>n:
        plt.plot(loss[-n::]);plt.title(f'loss: last {n} iteration');
        plt.plot(moving_average(loss[-n::], 10))
    else:
        plt.plot(loss);plt.title('overall loss');

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.15, hspace=.45);

    plt.show();