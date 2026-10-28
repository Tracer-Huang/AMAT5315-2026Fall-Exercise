"""Fit the published Chinchilla scaling runs and compute optimal budgets."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
N0, D0, DELTA = 1e9, 1e10, .001
PRINTED = (1.69, 406.4, 410.7, .34, .28)


def optimal_split(params, compute):
    _, A, B, alpha, beta = params
    if any(np.any(np.asarray(value) <= 0) for value in (A, B, alpha, beta)):
        raise ValueError('positive coefficients and exponents required')
    G = (alpha*A/(beta*B))**(1/(alpha+beta))
    n = G*(np.asarray(compute)/6)**(beta/(alpha+beta))
    return n, np.asarray(compute)/(6*n)


def robust_objective(theta, n, d, observed):
    e, a, b = np.exp(theta[:3]); alpha, beta = theta[3:]
    xn = np.log(n/N0); xd = np.log(d/D0)
    x = a*np.exp(-alpha*xn); y = b*np.exp(-beta*xd)
    predicted = e+x+y
    residual = np.log(predicted)-np.log(observed)
    h = np.where(np.abs(residual) <= DELTA, .5*residual**2, DELTA*(np.abs(residual)-.5*DELTA))
    derivative = np.clip(residual,-DELTA,DELTA)/predicted/len(residual)
    grad = np.array([np.sum(derivative*e),np.sum(derivative*x),np.sum(derivative*y),
                     np.sum(derivative*(-x*xn)),np.sum(derivative*(-y*xd))])
    return float(h.mean()), grad


def unscale(theta):
    e,a,b = np.exp(theta[:3]); alpha,beta=theta[3:]
    return np.array([e,a*N0**alpha,b*D0**beta,alpha,beta])


def predict(params,n,d):
    e,a,b,alpha,beta=params
    return e+a*np.asarray(n)**(-alpha)+b*np.asarray(d)**(-beta)


def fit(n,d,y,start):
    bounds=[(-3,2),(-8,5),(-8,5),(.01,1.2),(.01,1.2)]
    return minimize(robust_objective,start,args=(n,d,y),jac=True,method='L-BFGS-B',bounds=bounds,
                    options=dict(maxiter=2000,ftol=1e-13,gtol=1e-10,maxls=50))


def main():
    source=ROOT/'data'/'chinchilla-runs.csv'
    rows=list(csv.DictReader(source.open()))
    n=np.array([float(r['Model Size']) for r in rows]); compute=np.array([float(r['Training FLOP']) for r in rows])
    y=np.array([float(r['loss']) for r in rows]); d=compute/(6*n)
    keep=np.argsort(y)[:-5]; n,d,y,compute=n[keep],d[keep],y[keep],compute[keep]
    if len(n)!=240:
        raise ValueError('expected 245 runs and 240 after excluding the five highest losses')
    starts=[np.array([np.log(e),np.log(.4),np.log(.4),a,b])
            for e,a,b in itertools.product([1.6,1.8,2.0],[.25,.35,.45],[.25,.35,.45])]
    fits=[fit(n,d,y,s) for s in starts]; best=min(fits,key=lambda r:r.fun)
    if not best.success:
        raise RuntimeError(f'best fit did not converge: {best.message}')
    params=unscale(best.x)
    print('fit',params,'objective',best.fun,flush=True)
    out=ROOT/'artifacts'/'chinchilla';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(531508); indices=rng.integers(0,len(n),size=(400,len(n)))
    draws=[]
    for i,index in enumerate(indices):
        file=out/f'bootstrap-{i:03d}.json'
        if file.exists():
            record=json.loads(file.read_text())
            if record['data_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or record['index']!=i:
                raise ValueError('bootstrap cache belongs to another dataset or draw')
            theta=np.array(record['theta'])
        else:
            result=fit(n[index],d[index],y[index],best.x)
            if not result.success:
                raise RuntimeError(f'bootstrap {i} did not converge: {result.message}')
            theta=result.x
            with file.open('x') as f:
                json.dump(dict(index=i,theta=theta.tolist(),objective=float(result.fun),
                               data_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),f)
        draws.append(unscale(theta))
        if (i+1)%50==0: print(f'bootstrap {i+1}/400',flush=True)
    draws=np.array(draws)
    budget=5.76e23
    nstar,dstar=optimal_split(params,budget)
    b_n,b_d=optimal_split(draws.T,budget)
    fit_res=np.log(predict(params,n,d))-np.log(y)
    paper_res=np.log(predict(PRINTED,n,d))-np.log(y)
    summary=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=len(rows),fitted=len(n),
                 grid_starts=len(starts),huber_delta=DELTA,bootstrap=400,bootstrap_seed=531508,
                 parameters=dict(zip(['E','A','B','alpha','beta'],map(float,params))),
                 standard_errors=dict(zip(['E','A','B','alpha','beta'],map(float,draws.std(axis=0,ddof=1)))),
                 objective=float(best.fun),mean_log_residual=float(fit_res.mean()),
                 printed_mean_log_residual=float(paper_res.mean()),
                 budget=budget,N_star=float(nstar),D_star=float(dstar),tokens_per_parameter=float(dstar/nstar),
                 tokens_per_parameter_95=np.percentile(b_d/b_n,[2.5,97.5]).tolist())
    (ROOT/'evidence'/'part6-chinchilla.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(n,d,y,compute,params,fit_res,paper_res)
    print(json.dumps(summary,indent=2))


def plot(n,d,y,c,params,fit_res,paper_res):
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    ax[0].scatter(n,y,c='0.8',s=10,label='240 runs')
    grid=np.geomspace(n.min(),n.max(),200)
    for budget,color in zip([1e19,1e20,1e21],['#2b6cb0','#dd6b20','#2f855a']):
        select=np.abs(c/budget-1)<.15
        ax[0].scatter(n[select],y[select],color=color,s=12)
        ax[0].plot(grid,predict(params,grid,budget/(6*grid)),color=color,label=f'{budget:.0e} FLOP')
    ax[0].set(xscale='log',xlabel='parameters N',ylabel='final loss');ax[0].legend(fontsize=8)
    fit=predict(params,n,d);ax[1].scatter(fit,y,c=np.log10(c),s=12,cmap='viridis')
    limits=[min(fit.min(),y.min()),max(fit.max(),y.max())];ax[1].plot(limits,limits,'k--',lw=.8)
    ax[1].set(xlabel='fitted loss',ylabel='measured loss')
    fig.savefig(ROOT/'evidence'/'part6-chinchilla-fit.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    ax[0].scatter(n,fit_res,s=10,label='refit',alpha=.7);ax[0].scatter(n,paper_res,s=10,label='printed',alpha=.7)
    ax[0].axhline(0,c='k',lw=.6);ax[0].set(xscale='log',xlabel='parameters N',ylabel='predicted − measured log loss');ax[0].legend()
    ax[1].hist(fit_res,bins=30,alpha=.6,label='refit');ax[1].hist(paper_res,bins=30,alpha=.6,label='printed');ax[1].legend()
    fig.savefig(ROOT/'evidence'/'part6-residuals.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.4),constrained_layout=True)
    budgets=np.geomspace(1e18,1e25,200)
    for p,style,lab in [(params,'-','refit'),(PRINTED,'--','printed')]:
        ns,ds=optimal_split(p,budgets);ax.plot(budgets,ns,style,label=f'N*, {lab}');ax.plot(budgets,ds,style,label=f'D*, {lab}')
    ax.scatter([5.76e23,5.76e23],[70e9,1.4e12],marker='*',s=70,color='black',label='Chinchilla')
    ax.set(xscale='log',yscale='log',xlabel='training FLOP',ylabel='parameters / tokens');ax.legend(fontsize=8)
    fig.savefig(ROOT/'evidence'/'part6-optimal.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    main()
