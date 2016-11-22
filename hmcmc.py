#!/usr/bin/python3



### 1. Data import

import sys  ## system
import numpy as np  ## Matrix Calculate
import glob  ## global variable
import random  ## random sample
from multiprocessing import Pool  ## palarellel
from math import exp, gamma, log, sqrt  ## exp,log,sqrt cal
from scipy.stats import multivariate_normal
from numpy import inf
import scipy.linalg as sl
import scipy.stats as ss
#import matplotlib.pyplot as plt

list_of_files = sorted(glob.glob('/stfs1/uhome/y90027/MCMC/carb/demand/*.csv'))
list_of_files2 = sorted(glob.glob('/stfs1/uhome/y90027/MCMC/carb/prices/*.csv'))
#list_of_files = glob.glob('D:\OneDrive\dat\carb\demand\*.csv')
#list_of_files2 = glob.glob('D:\OneDrive\dat\carb\prices\*.csv')

demand = [np.loadtxt(f, skiprows=1, delimiter=",") for f in list_of_files]
prices = [np.loadtxt(f, skiprows=1, delimiter=",") for f in list_of_files2]
# print(list_of_files)
# print(list_of_files2)

#####################  2.  Fucntions
###prior : 0 = pu0 , 1 = pv0, 2 = gu0 , 3 = gv0 , 4 = tpp , 5 = tpg
def mdcev(par, dem, pri):
    x = np.array(dem)
    p = np.array(pri)

    k = x.shape[1]
    n = x.shape[0]
    # print(x.shape,p.shape)
    psi = np.array(par["psi"])
    psi = psi.reshape((k, 1))
    expgam = np.array(np.exp(par["gam"]))
    expgam = expgam.reshape((k, 1))
    temp = np.array([psi.transpose()] * n).reshape((n, k))
    temp2 = np.array([expgam.transpose()] * n).reshape((n, k))
    v = temp - np.log(np.array(x * temp2 + 1)) - np.log(p)
    xp = x > 0
    # print("v",v)
    mv = xp.sum(axis=1)  ## 0 = col  1 = row
    # print(mv)
    temp3 = np.array([1.] * len(mv))
    for i in range(1, len(mv)):
        temp3[i] = gamma(mv[i])
    llv = (v * xp).sum(axis=1) - mv * np.log(np.exp(v).sum(axis=1)) + np.log(temp3)

    # print("llv:/n",llv)
    expgammv = np.array([expgam.transpose()] * n).reshape((n, k)) * xp
    jacv = expgammv / (expgammv * x + 1)
    # print("jacv", jacv )
    temm = np.log(jacv)
    temm[temm == -inf] = 0.

    temp = np.array(temm * np.array(xp)).sum(axis=1)
    temm = np.array(1 / jacv)
    temm[temm == inf] = 0.
    temp2 = np.log((temm * np.array(xp)).sum(axis=1))
    jac = temp + temp2
    ll = llv + jac
    sumll = sum(ll)
    # print("sumll",sumll)
    return (sumll)


# print(demand[0].shape,prices[0].shape,demand[1].shape,p   rices[1].shape)
def rwmh(par, ll, prior, dem, pri):  ## prior[0] =   priorh = {"zt":zt[i, :], "vp":vp, "zp":zp[i, :], "vg": vg,
    # "tpp":tpp[i] * np.diag(np.array([1] * kp)),
    # "tpg":tpg[i] * np.diag(np.array([1] * k)) }
    k = len(prior["gu0"])  ###
    kp = k - 1
    new_par = par
    # print(new_par)
    ite = [0, 0]  ### num of mcmc iteration
    ## sampling psi
    llp = ll + multivariate_normal.logpdf(par["psi"][0:kp], prior["pu0"], prior["pv0"])  ###### //
    acc = 0
    while acc == 0:

        ite[0] += 1
        new_psi = np.append(multivariate_normal.rvs(mean=par["psi"][0:kp], cov=prior["tpp"], size=1), [0.])
        new_par["psi"] = new_psi  ### vstack vex stack   hstack = col
        # print("newpsi",new_psi)

        new_ll = mdcev(par=new_par, dem=dem, pri=pri)
        # print("ll",ll,"newll",new_ll)

        # print(new_par[0][0:kp])
        # print(new_par["psi"][0:kp])
        # print(new_ll, (multivariate_normal.logpdf(new_par["gam"], prior["gu0"], prior["gv0"] * 0.1)))
        new_llp = new_ll + multivariate_normal.logpdf(new_par["psi"][0:kp], prior["pu0"], prior["pv0"]*0.1)
        # print("newllp",new_llp ,"llp" ,llp,"pro:",exp(new_llp-llp))
        # print("in psia"," ll",ll,"new_ll",new_ll,"llp",llp,"new_llp",new_llp)
        # print("in psia")
        if (new_llp - llp) > np.log(random.uniform(0, 1)):
            acc = 1
            par = new_par
            ll = new_ll
    ## sampling gam
    llp = ll + multivariate_normal.logpdf(par["gam"], prior["gu0"], prior["gv0"])
    acc = 0
    while acc == 0:
        # print("in gamma")
        ite[1] += 1
        # if ite[1] > 10000:
        # print(new_llp - llp)

        new_gam = multivariate_normal.rvs(mean=par["gam"], cov=prior["tpg"], size=1).transpose()
        new_par["gam"] = np.array(new_gam)
        new_ll = mdcev(new_par, dem, pri)
        # print(new_par["gam"])
        # print(new_ll,( multivariate_normal.logpdf(new_par["gam"], prior["gu0"], prior["gv0"]*0.1)))
        new_llp = new_ll + multivariate_normal.logpdf(new_par["gam"], prior["gu0"], prior["gv0"]*0.1)
        # print("newgam:",new_gam)
        if (new_llp - llp) > np.log(random.uniform(0, 1)):
            acc = 1
            par = new_par
            ll = new_ll
            # print(par["psi"])
    # out =[par,ll,ite]
    # print("par",par)
    return {"par": par, "ll": ll, "ite": ite}


# print(np.array(ss.chi2.rvs(df=50,size=100) ) )

def rwishart(nu, v):
    m = v.shape[0]
    df = int((nu + nu - m + 1) - (nu - m + 1))
    if m > 1:
        # t = np.diag( sqrt( ss.chi2.rvs(df=df, size=np.array([1] * m) )  )  )
        t = np.diag(np.sqrt(ss.chi2.rvs(df=df, size=m)))
        l = np.tril_indices(t.shape[0], -1)

        t[l] = np.random.normal(0, 1, (m * (m + 1) / 2 - m))
    else:
        t = sqrt(ss.chi2.rvs(size=1, df=df))
    if v.shape[0] == 1:
        u = np.sqrt(v)
    else:
        u = np.linalg.cholesky(v)
    c = np.dot(t.transpose(), u)
    ci = np.array(sl.solve_triangular(c, np.diag([1] * m), lower=False))
    # print(np.dot(ci.transpose(),ci))
    return {
        "W": np.dot(c.transpose(), c),
        "IW": np.dot(ci.transpose(), ci),
        "C": c,
        "CI": ci
    }

### cholesky decomposition
def chol(x):
    if (x.shape[1] == 1 and x.shape[0] == 1):
        x = np.array(sqrt(x)).reshape((1, 1))
    else:
        x = np.linalg.cholesky(x)
    return (x)

### create symmetric matrix from vector
def xpnd(x):
    dim = (-1 + sqrt(1 + 8 * len(x))) / 2
    new = np.zeros((dim, dim))
    inds = np.triu_indices_from(new)
    new[inds] = x
    new[(inds[1], inds[0])] = x
    return new

### bayesian multivariate regression
def rmultireg(y, x, bbar, a, nu, v, n):
    l1 = y.shape[0]
    m = y.shape[1]
    k = x.shape[1]
    ## first draw sigma
    # print(m)
    ra = chol(a)
    w = np.vstack((x, ra))
    # print("bbar:",bbar)
    z = np.vstack((y, np.dot(ra, bbar)))
    # print("z:",z)

    # print(np.dot(w.transpose(),z))
    # ir = 1 / chol(chol(np.dot(w.transpose(), w)), np.diag([1] * k))

    ir = 1 / chol(chol(np.dot(w.transpose(), w)))
    btilde = np.dot(np.dot(ir, ir.transpose()), np.dot(w.transpose(), z))
    # print("ir", ir)
    # print("btilde",btilde)
    temp = z - np.dot(w, btilde)
    s = np.dot(temp.transpose(), temp)
    out1 = np.array([0.] * n * m).reshape((n, m))
    if m == 1:
        out2 = np.array([0.] * n).reshape((n, 1))
    if m > 1:
        out2 = {"list": [] for i in range(n)}
    for i in range(n):
        rwout = rwishart(nu + l1, np.linalg.inv(np.dot(chol(v + s).transpose(), chol(v + s))))
        # print(np.random.standard_normal(m*k))
        out1[i, :] = btilde + np.dot(np.dot(ir, np.random.standard_normal(m * k).reshape((k, m))), rwout["CI"])
        # print(out1[i,:])
        if m == 1:
            out2[i, 0] = np.array(rwout["IW"])
        else:
            out2[i] = np.array(rwout["IW"])
    return {"B": out1, "Sigma": out2, "BStar": btilde}


k = demand[1].shape[1]
kp = k - 1
h = len(demand)

zdata = np.array([1.] * h).reshape((h, 1))
rankz = zdata.shape[1]

### 2. MCMC settings
burnin = 300
mcmc = 700
if len(sys.argv) > 2:
    if len(sys.argv) > 3:
        mcmc = sys.argv[2]
    burnin = sys.argv[3]

thin = 2
nmcmc = burnin + thin * mcmc

### 3.  Prior Setting
tu0 = np.array([0.] * rankz * kp).reshape((rankz, kp))
tv0 = np.diag(np.array([1.] * rankz)) * 0.001

pf0 = kp + 1
pg0 = np.diag(np.array([1.] * kp)) * pf0

hu0 = np.array([0.] * rankz * k).reshape((rankz, k))
hv0 = np.diag(np.array([1.] * rankz)) * 0.001

gf0 = k + 1
gg0 = np.diag(np.array([1.] * k)) * gf0

tpp = np.array([0.01] * h)
tpg = np.array([0.01] * h)

### 4. Initial Value
psia = np.array([0.] * h * k).reshape((h, k))
psig = np.array([0.] * mcmc * h * k).reshape((mcmc, h * k))

theta = np.array([0.] * rankz * kp).reshape((rankz, kp))
thetag = np.array([0.] * mcmc * rankz * kp).reshape((mcmc, rankz * kp))

vp = np.diag(np.array([1.] * kp))
# vpg = np.array([0] * mcmc * int( kp * ((kp + 1) / 2))).reshape((mcmc, int(kp * (kp + 1)/2 )))
temp = int(float(kp) * (float(kp) + 1) / 2)
vpg = np.array([0.] * mcmc * temp).reshape((mcmc, temp))

gama = np.array([0.] * h * k).reshape((h, k))
gamg = np.array([0.] * mcmc * h * k).reshape((mcmc, h * k))

phi = np.array([0.] * rankz * k).reshape((rankz, k))
phig = np.array([0.] * mcmc * rankz * k).reshape((mcmc, rankz * k))

vg = np.diag(np.array([1.] * k))
vgg = np.array([0] * mcmc * int(k * (k + 1) / 2)).reshape((mcmc, int(k * (k + 1) / 2)))
# vgg = np.array([0.] * mcmc * 900).reshape((mcmc, 900))

lla = [0.] * h
for i in range(0, h):
    parh = {"psi": psia[i, :], "gam": gama[i, :]}
    lla[i] = mdcev(parh, demand[i], prices[i])

llg = np.array([0.] * mcmc * h).reshape((mcmc, h))
itea = np.array([0] * h * 2).reshape((h, 2))

zt = np.dot(zdata, theta)
zp = np.dot(zdata, phi)
priorh = {"pu0": zt[0, :], "pv0": vp, "gu0": zp[0, :], "gv0": vg, "tpp": tpp[0] * np.diag(np.array([1] * k)),
          "tpg": tpg[0] * np.diag(np.array([1] * k))}


### parallel
def cal(i):
    global psia, gama, zt, vp, zp, vg, tpp, tpg
    # print("iteration:",i)
    parh = {"psi": psia[i, :], "gam": gama[i, :]}
    priorh = {"pu0": zt[i, :], "pv0": vp, "gu0": zp[i, :], "gv0": vg,
              "tpp": tpp[i] * np.diag(np.array([1] * kp)),
              "tpg": tpg[i] * np.diag(np.array([1] * k))}
    outh = rwmh(parh, lla[i], priorh, demand[i], prices[i])
    return outh

### MCMC
def hmcmc():
    global psia, gama, zt, vp, zp, vg, tpp, tpg, theta, phi, zdata, tu0, tv0, pf0, pg0, tpp, tpg, hu0, hv0, gf0, gg0
    for imcmc in range(nmcmc):
        print(imcmc)
        # print("psia",psia[1,:])
        zt = np.dot(zdata, theta)
        zp = np.dot(zdata, phi)
        result = p.map(cal, argList)
        for i in range(h):
            psia[i, :] = result[i]["par"]["psi"]
            gama[i, :] = result[i]["par"]["gam"]
            lla[i] = result[i]["ll"]
            itea[i, :] = result[i]["ite"]
        out1 = rmultireg(y=psia[:, 0:kp], x=zdata, bbar=tu0, a=tv0, nu=pf0, v=pg0, n=1)
        theta = np.array(out1["B"])
        vp = out1["Sigma"][0]
        # print(np.squeeze(np.asarray(psia)))
        out2 = rmultireg(y=gama, x=zdata, bbar=hu0, a=hv0, nu=gf0, v=gg0, n=1)
        # print("out1[B]",out1["B"])
        # print("out2[B]",out2["B"])
        # print(psia)
        phi = out2["B"]
        vg = out2["Sigma"][0]

        if imcmc > burnin:
            jmcmc = (imcmc - burnin) / thin
            psig[jmcmc, :] = psia.reshape((607 * 30))
            gamg[jmcmc, :] = gama.reshape((607 * 30))
            thetag[jmcmc, :] = theta.reshape((29))
            # print(vp[np.tril_indices(vp.shape[0], -1)])
            vpg[jmcmc,] = vp[np.tril_indices(vp.shape[0], 0)]
            phig[jmcmc, :] = phi.reshape((30))
            vgg[jmcmc,] = vg[np.tril_indices(vg.shape[0], 0)]
            llg[jmcmc, :] = lla
    output = {"psig": psig, "gamg": gamg, "thetag": thetag,
              "phig": phig, "llg": llg, "vpg": vpg, "vgg": vgg}

    return (output)


if __name__ == '__main__':
    coren = 24
    if (len(sys.argv) > 1):
        coren = sys.argv[1]
    argList = range(h)
    p = Pool()

    ## MCMC
    outp = hmcmc()

    ## Posterior Mean
    ## Posterior s.d.
    psim = np.mean(outp["psig"], axis=0).reshape((h, k))
    psis = np.std(outp["psig"], axis=0).reshape((h, k))

    thetam = np.mean(outp["thetag"], axis=0).reshape((rankz, kp))
    thetas = np.std(outp["thetag"], axis=0).reshape((rankz, kp))

    vpm = xpnd(np.mean(outp["vpg"], axis=0))
    vps = xpnd(np.std(outp["vpg"], axis=0))

    gamm = np.mean(outp["gamg"], axis=0).reshape((h, k))
    gams = np.std(outp["gamg"], axis=0).reshape((h, k))

    phim = np.mean(outp["phig"], axis=0).reshape((rankz, k))
    phis = np.std(outp["phig"], axis=0).reshape((rankz, k))

    vgm = xpnd(np.mean(outp["vgg"], axis=0))
    vgs = xpnd(np.std(outp["vgg"], axis=0))

    accept_rate = mcmc / itea

    ## dic
    llm = np.zeros(h)
    for i in range(h):
        parm = {"psi": psim[i, :], "gam": gamm[i, :]}
        llm[i] = mdcev(parm, demand[i], prices[i])
    dic = -4 * np.sum(np.mean(llg, axis=1)) + 2 * np.sum(llm)
    ### print
    print("llm:", llm)
    print("psim:", psim)
    print("psis", psis)
    print(dic)
    np.savez("/stfs1/uhome/y90027/MCMC/result.npz",psig=outp["psig"],thetag=outp["thetag"],vpg=outp["thetag"],
             gamg=outp["gamg"],phig=outp["phig"],vgg=outp["vgg"],
             dic=dic,psim=psim,psis=psis,thetam=thetam,
             vpm=vpm, vps=vps, gamm=gamm, gams=gams, phim=phim, phis=phis, vgm=vgm, vgs=vgs,accept_rate=accept_rate)
    # for i in range(90):
    #     plt.plot(outp["psig"][:, i])
    #     title = "psia" + str(i)
    #     plt.title(title)
    #     plt.savefig("/stfs1/uhome/y90027/MCMC/PIC/psia/" + title + ".png")
    #     plt.clf()
    # for i in range(90):
    #     plt.plot(outp["gamg"][:, i])
    #     title = "gama" + str(i)
    #     plt.title(title)
    #     plt.savefig("/stfs1/uhome/y90027/MCMC/PIC/gama/" + title + ".png")
    #     plt.clf()















