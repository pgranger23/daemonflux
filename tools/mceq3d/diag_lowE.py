"""Low-energy MC/engine level diagnostic (INTEGRATION.md sec. 12.z.1): r = <G_MC>_az / <F_engine>_az per species,
E bin (0.1-0.4 GeV) and cos Z band, GSF and GH-native engine. Run from tools/mceq3d with the diag_v1 environment."""
import numpy as np, os, diag_v1 as V, diag_integ as D, warnings
warnings.filterwarnings("ignore")
W=D.W
fine=np.load(os.path.join(V.OUT,"engine_F_fine.npz")); gd=dict(np.load(V.G_FILE_B,allow_pickle=True))
g,gerr=gd["G"],gd["G_err_total"]
ed=V.engine_down12(fine)
SP=D.MC_SPECIES
def ratio(G,F,rows):
    return G[:,:,rows].mean((-1,-2))/F[:,:,rows].mean((-1,-2))
def sig(Gerr,F,rows):
    n=len(range(*rows.indices(10)))*12
    return np.sqrt((Gerr[:,:,rows]**2).sum((-1,-2)))/n/F[:,:,rows].mean((-1,-2))
# m3b cells (rows 0.05, 0.15, 0.25), E-averaged on the table bins
def m3b(nm,mb):
    m=np.load(W+f"/runs/rigfix/out/m3b_engine_{nm}_mb{mb}.npz"); out=np.zeros((4,10,3,12))
    for k,sp in enumerate(D.SPECIES):
        s=sp
        f=m[s]/fine[f"base_hor_{sp}"][:,None,:]
        out[k]=V.fine_avg(fine,f,fine[f"base_hor_{sp}"])
    return out
Fgsf,Fghn=m3b("gsf",0),m3b("ghnative",0)
Ea=("0.10-0.16","0.16-0.25","0.25-0.40")
print("(a) r=<G_MC>/<F_engine GSF>, engine_F_fine; sigma_r MC only")
for ie in range(3):
    for k,sp in enumerate(SP):
        row=[]
        for ic in range(5):
            rows=slice(2*ic,2*ic+2)
            r=ratio(g,ed,rows)[k,ie]; s=sig(gerr,ed,rows)[k,ie]; row.append(f"{r:.3f}+-{s:.3f}")
        print(Ea[ie],f"{sp:9s}"," | ".join(row))
print("\n(b) cz 0-0.2 (rows 0.05,0.15) and row 0.25 (cz 0.2-0.3): GSF m3b mb0 | GH-native m3b mb0 ; G_MC same rows; (b)-(a) in r")
for ie in range(3):
    for k,sp in enumerate(SP):
        o=[]
        for rows,mrows in ((slice(0,2),slice(0,2)),(slice(2,3),slice(2,3))):
            G=g[k,ie,rows].mean(); a=Fgsf[k,ie,mrows].mean(); b=Fghn[k,ie,mrows].mean()
            o.append(f"GSF {G/a:.3f} GHn {G/b:.3f} (F_ghn/F_gsf {b/a:.3f})")
        print(Ea[ie],f"{sp:9s}"," | ".join(o))
rd=np.load(V.R_FILE,allow_pickle=True)["R"]  # (4,7,5,6)
print("\n<R-1>_az measured (species x E bin 0-3, cz 0.2 bands): max|.|, median|.| over E<0.4 bins and all cz:",np.abs(rd.mean(-1)-1)[:,:3].max().round(4),np.median(np.abs(rd.mean(-1)-1)[:,:3]).round(4)," all E<2.5:",np.abs(rd.mean(-1)-1).max().round(4),np.median(np.abs(rd.mean(-1)-1)).round(4))
print("per cz band, E<0.4: ",[ (np.abs(rd.mean(-1)-1)[:,:3,ic].max()).round(3) for ic in range(5)])
