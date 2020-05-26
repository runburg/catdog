import math as m
import numpy as np

import urllib.request as urlreq

deg = np.pi/180.0
verbose = False

def linkskysep(qas,qb,linkp):
    dang = linkp[0]
    aa, da = qas[:,0], qas[:,1]
    ab, db = qb[0],qb[1]
    cosr = np.sin(da)*np.sin(db)+np.cos(da)*np.cos(db)*np.cos(aa-ab)
    return (cosr>=np.cos(dang))

def nei(q,islink,linkp):
    dr = linkp[0]
    #idxn = np.searchsorted(q[:,didx],q[0,didx]+dr,side='right')
    # qi = q[:idxn] # potential neighbors # use qi x 2 next line
    idxn = islink(q,q[0],linkp) # bool array, neighbor if true...
    return idxn

def fof(q,islink,linkp):
    nq = len(q)
    glis = -np.ones(nq)
    gid = -1
    for i in range(nq-1):
        idxn = nei(q[i:],islink,linkp)
        j = len(idxn)
        if j<2: continue
        g = glis[i:i+j]
        gn = g[idxn]
        if len(gn)<2: continue       
        gnuniq = np.unique(np.copy(gn))
        if gnuniq[-1] < 0:
            gid += 1
            gidthis = gid # new group 
        else:
            if gnuniq[0]<0: gnuniq = gnuniq[1:]
            gidthis = gnuniq[0]
            for gi in gnuniq[1:]:
                msk = (glis==gi)
                glis[msk] = gidthis
        glis[i:i+j] = np.where(idxn==True,gidthis,glis[i:i+j])
    gnuniq = np.unique(np.copy(glis))
    if gnuniq[0]<0: gnuniq = gnuniq[1:]
    i = 0
    for j in gnuniq: 
        glis[glis==j] = i
        i += 1
    return glis,len(gnuniq)


#"https://raw.githubusercontent.com/runburg/catdog/master/candidates/trial3333_rad100_small_pm_range3/successful_candidates_with_overlap_gte10_withcounts.txt")

def readurl(myurl):
    # reads URL txt file, expecting decimal degrees RA, dec, and N
    data = urlreq.urlopen(myurl)
    RAdecN = np.array([[float(li) for li in l.split()] for l in data])
    if (RAdecN.shape[1] != 3): 
        print('url misread?')
        quit()
    return RAdecN

def fofurl(myurl,skysep_degrees):
    skysep = 1*deg
    RAdecN = readurl(myurl)
    # convert to radians...
    RAdecN[:,0] *= deg; RAdecN[:,1] *= deg
    gidlis,ngroups = fof(RAdecN,linkskysep,[skysep])
    for rdn,g in zip(RAdecN,gidlis):
        print('%18.15f %18.15f %5d %4d'%(rdn[0]/deg,rdn[1]/deg,rdn[2],g))
#test()


def test():
    # jack's data
    myurl = "https://raw.githubusercontent.com/runburg/catdog/master/candidates/trial3333_rad100_small_pm_range3/successful_candidates_with_overlap_gte10_withcounts.txt"
    RAdecN = readurl(myurl)
    RAdecN[:,0] *= deg; RAdecN[:,1] *= deg
    skysep = 1*deg
    gid,ngroups = fof(RAdecN,linkskysep,[skysep])
    print(ngroups,'groups found.')
    print(np.sum(gid>=0),'grouped.')
    print(np.sum(gid==-1),'ungrouped.')
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl
    from matplotlib import cm
    pl.scatter(RAdecN[:,0],RAdecN[:,1],s=1,c=gid,cmap=cm.rainbow)
    plotfile = 'fofsky_test.png'
    pl.savefig(plotfile)
    #import os
    #os.system('convert '+plotfile+' ~/www/tmp.jpg')

# test()

