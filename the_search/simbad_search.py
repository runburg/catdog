import numpy as np
import sys
import astroquery
from astroquery.simbad import Simbad
import warnings
warnings.filterwarnings("ignore")

def  angdist(ra,dec,ra0,dec0): # degrees, returns dist to ra0,dec0, also degrees
    deg = np.pi/180.0
    a = ra*deg; d = dec*deg
    a0 = ra0*deg; d0 = dec0*deg
    cosr = np.sin(d)*np.sin(d0)+np.cos(d)*np.cos(d0)*np.cos(a-a0)
    if cosr>1.0: cosr=1.0 # don't worry about < -1..
    drang = np.arccos(cosr)
    return drang/deg

def parse_searchrad_string(dmax):
    searchraddeg = 0
    tmp = dmax
    tmptmp = ''.join([i for i in tmp if not i.isdigit()])
    if (len(tmptmp) != 1) or (tmptmp not in 'dms'):
        print(' choked on search radius. use only one unit, 20m, not 0d20m0s')
        quit()
    if 'd' in tmp: searchraddeg += float(tmp.split('d')[0])
    if 'm' in tmp: searchraddeg += float(tmp.split('m')[0])/60.
    if 's' in tmp: searchraddeg += float(tmp.split('s')[0])/3600.
    return searchraddeg

simbad_initialized = False

def simbadquery_init(verbose=False):
    global simbad_initialized
    # you can get what you need
    Simbad.add_votable_fields('ra(d)','dec(d)','otype','pm','rv_value')
    Simbad.remove_votable_fields('coordinates')
    simbad_initialized = True
    
    if verbose:
        print('# looking for star clusters, GCs and named dwarf galaxies...')

def simbadquery_reset(verbose=False):
    global simbad_initialized
    # you can get what you need
    Simbad.reset_votable_fields()
    simbad_initialized = False

skipnamelist = ['LEDA','LCRS','2MASS','2MASX','2SLAQ','SDSS','dFGS','MGC','MCG','LMC','NGP9']
skipotypelist = ['PairG','GinGroup','Seyfert']

def simbadquery(ra,dec,searchrad, verbose=False, rv_cutoff=1000.0): 
    global simbad_initialized, skipnamelist, skipotypelist 
    if simbad_initialized == False: # in the sense of a directive...
        print('you must initialize simbad query!!!')
        quit()
    # deg, deg, "15m" (for example) 
    head = '# OBJID : OTYPE : RA DEC PMRA PMDEC RV [SEARCH_DISTANCE]'
    idinfo = []
    dist = []
    a,d = ra,dec
    sign = "+" if d>=0 else "-"
    radec = str(a)+" "+sign+str(np.abs(d))
    # this prints out Simbad "query criteria" search string...
    crit =  "region(CIRCLE, "+radec+", "+searchrad+")"
    crit += " & (otype='Cl*' | otype='GlC' | otype='OpC' | otype='G' "
    crit += " | maintype='Gl?' | maintype='G?'"
    crit += ")"
    #print crit
    result = Simbad.query_criteria(crit)
    if result is None:
        if verbose:
            print('# no objects found')
        return head,[]
    for entry in result:
        name,otype=entry[0],entry[3]
        name = name.decode('UTF-8')
        otype = otype.decode('UTF-8')
        if '[' in name: continue
        skipthis = False
        for skip in skipnamelist:
            if skip in name:
                skipthis = True
                break
        if skipthis: continue
        for skip in skipotypelist:
            if skip in otype:
                skipthis = True
                break
        if skipthis: continue
        rax,decx = float(entry[1]),float(entry[2])        
        pmrax,pmdecx,rvx = entry[4],entry[5],entry[-1]
        if rvx>rv_cutoff: continue
        ddeg = angdist(rax,decx,float(ra),float(dec))
        if ddeg >= 1: dstr = '%1.4fd'%(ddeg)
        elif ddeg >= 1./60.: dstr = '%1.4fm '%(ddeg*60.)
        else: dstr = '%1.2fs '%(ddeg*3600.)
        idthis = "%28s : %8s : %6.2f %5.2f %6s %6s %5s [%s]" % \
                    (name,otype,rax,decx,pmrax,pmdecx,rvx,dstr)
        idinfo.append(idthis)
        dist.append(ddeg)
    if (len(dist)):
        iddistzip = zip(idinfo,dist)
        iddistsort = sorted(iddistzip, key = lambda x: x[1])
        idinfo, dist = zip(*iddistsort)
    return head,idinfo
