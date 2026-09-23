"""Independent NumPy diagnostics of saved fields (not solver internals)."""
import numpy as np
import json
from pathlib import Path

def frames(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def grid(values):
    a=np.asarray(values,dtype=float); n=int(np.sqrt(a.size))
    if n*n != a.size or not np.isfinite(a).all(): raise ValueError("expected finite square field")
    return a.reshape(n,n)

def waves(n, derivative=False):
    k=np.fft.fftfreq(n)*n
    if derivative: k[n//2]=0
    return np.meshgrid(k,k)

def diagnostics(frame):
    u=grid(frame["u"]); v=grid(frame["v"]); kx,ky=waves(len(u),True)
    uh=np.fft.fft2(u); vh=np.fft.fft2(v)
    omega=np.fft.ifft2(1j*kx*vh-1j*ky*uh).real
    divergence=np.fft.ifft2(1j*kx*uh+1j*ky*vh).real
    scale=max(float(np.max(np.abs(omega))),1e-30)
    return {"E":float(np.mean(u*u+v*v)/2),"Z":float(np.mean(omega*omega)/2),
            "divergence":float(np.max(np.abs(divergence))/scale),
            "consistency":float(np.max(np.abs(grid(frame["omega"])-omega))/scale),"omega":omega}

def budget(times, energies, enstrophies, nu):
    t=np.asarray(times); e=np.asarray(energies); z=np.asarray(enstrophies)
    integral=np.r_[0.,np.cumsum(np.diff(t)*(z[1:]+z[:-1])/2)]
    predicted=e[0]-2*nu*integral
    return predicted,float(abs(e[-1]-predicted[-1])/max(abs(e[-1]-e[0]),1e-30))

def transfer(initial, final, nu, t):
    kx,ky=waves(len(initial))
    w0=np.fft.fft2(initial); w1=np.fft.fft2(final)
    return float(np.linalg.norm(w1-w0*np.exp(-nu*(kx*kx+ky*ky)*t))/np.linalg.norm(w0))

def histogram(x, y, bins=16):
    if len(x)!=len(y) or len(x)==0: raise ValueError("nonempty paired positions required")
    counts,_,_=np.histogram2d(np.asarray(y)%(2*np.pi),np.asarray(x)%(2*np.pi),bins=bins,range=[[0,2*np.pi],[0,2*np.pi]])
    expected=len(x)/bins**2
    return counts,float(np.sum((counts-expected)**2/expected)/(bins**2-1))

def relative_error(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/np.linalg.norm(b))

def reduced(frame):
    return {"t":frame["t"],"step":frame["step"],"omega":grid(frame["omega"])[::2,::2].ravel().tolist()}
