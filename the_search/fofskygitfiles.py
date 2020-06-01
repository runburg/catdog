import math as m
import numpy as np
import the_search.fofsky as fofsky

myurl="https://raw.githubusercontent.com/runburg/catdog/master/candidates/trial3333_rad100_small_pm_range3/successful_candidates_with_overlap_gte10_withcounts.txt"


skysep_degrees = 1.0
deg = np.pi/180.0

# to run a test just uncomment this:
# fofsky.test() ; quit()
# in just plots something to fofsky_test.png

# this next code lists out all groups, picking out ungrouped and the
# ra dec of object in a group w/most members...
if False:
    data = fofsky.readurl(myurl) # read in RA, dec, N; data[:,0] == RA
    # convert to radians...
    data[:,0] *= deg; data[:,1] *= deg  # RA, dec, to radians...
    skysep = skysep_degrees * deg
    gidlis, ngroups = fofsky.fof(data,fofsky.linkskysep,[skysep]) 
    # Note: gidlis has group_ids...
    newdata = np.array([]) # flattened...
    for gid in np.unique(gidlis):
        if (gid < 0): # ungrouped
            newdata = np.append(newdata,data[gidlis==gid])
        else:
            dg = data[gidlis==gid]
            i = np.argmax(dg[:,2])
            newdata = np.append(newdata,dg[i])
    print(newdata.shape[0])
    newdata = np.reshape(newdata,(int(newdata.shape[0]/3),3))
    for rdn in newdata:
        print('%18.15f %18.15f %5d'%(rdn[0]/deg,rdn[1]/deg,rdn[2]))


def group_with_fof(coord_list, counts=[], skysep=np.pi/180):
    data = coord_list # read in RA, dec, N; data[:,0] == RA
    # if len(counts) > 0:
        # data = np.append(data.T, counts).T
    # convert to radians...
    data[:,0] *= deg; data[:,1] *= deg  # RA, dec, to radians...
    skysep = skysep_degrees * deg
    gidlis, ngroups = fofsky.fof(data,fofsky.linkskysep,[skysep]) 
    # Note: gidlis has group_ids...
    newdata = np.array([]) # flattened...
    for gid in np.unique(gidlis):
        if (gid < 0): # ungrouped
            newdata = np.append(newdata,data[gidlis==gid])
        else:
            dg = data[gidlis==gid]
            i = np.argmax(dg[:,2])
            newdata = np.append(newdata,dg[i])
    newdata = np.reshape(newdata,(int(newdata.shape[0]/3),3))
    newdata[:, :2] *= 180/np.pi
    return newdata
 
