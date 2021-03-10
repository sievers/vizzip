import numpy as np
import numba as nb
from scipy import linalg



@nb.njit
def reconstruct_channel(ivals,coeffs):
    out=np.zeros(len(ivals))
    out[:]=ivals
    #out=np.asarray(ivals,dtype='float')
    npt=len(coeffs)
    n=len(ivals)
    for i in range(npt,n):
        for j in range(npt):
            out[i]=out[i]+coeffs[j]*out[i-npt+j]
        out[i]=np.round(out[i])
    return out
@nb.njit
def reconstruct_channel_nofill(ivals,coeffs):
    out=np.zeros(len(ivals))
    out[:]=ivals
    #out=np.asarray(ivals,dtype='float')
    npt=len(coeffs)
    n=len(ivals)
    for i in range(npt,n):
        out[i]=0
        for j in range(npt):
            out[i]=out[i]+coeffs[j]*ivals[i-npt+j]
        out[i]=np.round(out[i])
    return out

@nb.njit
def predict_channel(vals,coeffs):
    out=np.zeros(len(vals))
    #out[:]=vals
    #out=np.asarray(ivals,dtype='float')
    npt=len(coeffs)
    n=len(vals)
    for i in range(npt,n):
        for j in range(npt):
            out[i]=out[i]+coeffs[j]*vals[i-npt+j]
    return out

@nb.njit
def _get_corr(chan,npt):
    n=len(chan)
    mycorr=np.zeros(npt+1)
    for i in range(n-npt):
        mycorr[:]=mycorr[:]+chan[i]*chan[i:i+npt+1]
    mycorr=mycorr/(n-npt) # we don't actually need to normalize, but why not
    return mycorr

def compress_channel(chan,nbit,npt):
    mycorr=np.zeros(npt+1)
    n=len(chan)
    #for i in range(n-npt):
    #    mycorr[:]=mycorr[:]+chan[i]*chan[i:i+npt+1]
    #mycorr=mycorr/(n-npt) # we don't actually need to normalize, but why not
    mycorr=_get_corr(chan,npt)
    if mycorr[0]==0:
        return np.zeros(len(chan),dtype='int'),np.zeros(npt),0.0

    mat=linalg.toeplitz(mycorr)
    matinv=np.linalg.inv(mat)
    coeffs=-matinv[:-1,-1]/matinv[-1,-1]
    pred=predict_channel(chan,coeffs)
    delt=chan-pred
    mystd=np.median(np.abs(delt[npt:-npt]))
    #print('mystd is ',mystd)
    fac=2**nbit/mystd
    ichan=np.asarray(np.round(chan*fac))
    #print('ichan is ',ichan[:10])
    ipred=reconstruct_channel_nofill(ichan,coeffs)
    idelt=ichan-ipred
    idelt[:npt]=ichan[:npt]
    #return idelt,coeffs,fac
    return idelt,coeffs,fac


def read_file(fname):
    f=open(fname,'r')
    info=np.fromfile(f,'int32',4)
    #print('info is ',info)
    ndat=info[0]
    nchan=info[1]
    npt=info[2]
    nbit=info[3]
    scats=np.fromfile(f,'float',nchan)
    coeffs=np.fromfile(f,'float',npt*nchan)
    coeffs=np.reshape(coeffs,[npt,nchan])
    idelt=np.fromfile(f,'int32',ndat*nchan)
    idelt=np.reshape(idelt,[ndat,nchan])
    f.close()
    out=np.zeros(idelt.shape)
    for i in range(nchan):
        #out[:,i]=reconstruct_channel(idelt[:,i],coeffs[:,i])*scats[i]/2**nbit
        if scats[i]>0:
            out[:,i]=reconstruct_channel(idelt[:,i],coeffs[:,i])/scats[i]
    out=out[:,:nchan//2]+1J*out[:,nchan//2:]
    return out

def compress_vis(vis,fname,npt=4,nbit=5):
    vv=np.hstack([np.real(vis),np.imag(vis)])
    ivv=np.zeros(vv.shape,dtype='int32')
    nchan=vv.shape[1]
    coeffs=np.zeros([npt,nchan])
    scats=np.zeros(nchan)
    for i in range(nchan):
        ivv[:,i],coeffs[:,i],scats[i]=compress_channel(vv[:,i],nbit,npt)
    outf=open(fname,'w')
    info=[vv.shape[0],vv.shape[1],npt,nbit]
    info=np.asarray(info,dtype='int32')
    info.tofile(outf)
    scats.tofile(outf)
    coeffs.tofile(outf)
    ivv.tofile(outf)
    outf.close()
