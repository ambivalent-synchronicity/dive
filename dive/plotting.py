## Plotting 

# # Import modules
import matplotlib.pyplot as plt
import numpy as np
import random
import arviz as az
from IPython.display import display
import deerlab as dl
import copy

from .utils import *
from .models import *

def summary(df, model, Vexp, t, r, nDraws = 100, Pref = None):

    # Figure out what Vars are present -----------------------------------------
    PossibleVars = ["r0","w","a","k","lamb","V0","sigma","delta",'lg_alpha']
    PresentVars = df.varnames

    Vars = []
    for Var in PossibleVars:
        if Var in PresentVars:
            Vars.append(Var)
    nVars = len(Vars)

    # Print summary for RVs ----------------------------------------------------
    with model:
        summary = az.summary(df,var_names=Vars)
    # replace the labels with their unicode characters before displaying
    summary.index = betterLabels(summary.index.values)
    display(summary)
    
    # Plot marginalized posteriors ---------------------------------------------
    plotsperrow = 4

    # figure out layout of plots and creat figure
    nrows = int(np.ceil(nVars/plotsperrow))
    if nVars > plotsperrow:
        fig, axs = plt.subplots(nrows, plotsperrow)
        axs = np.reshape(axs,(nrows*plotsperrow,))
        width = 11
    else:
        fig, axs = plt.subplots(1, nVars)
        width = 3*nVars
    height = nrows*3.5
   
    # set figure size
    fig.set_figheight(height)
    fig.set_figwidth(width)
    
    # KDE of chain samples and plot them
    for i in range(nVars):
        az.plot_kde(df[Vars[i]],ax = axs[i])
        axs[i].set_xlabel(betterLabels(Vars[i]))
        axs[i].yaxis.set_ticks([])

    # Clean up figure
    fig.tight_layout()
    plt.show()

    # Pairwise correlation plots ----------------------------------------------
    # determine figure size
    if nVars < 3:
        corwidth = 7
        corheight = 7
    else:
        corwidth = 11
        corheight = 11

    # use arviz librarby to plot them
    with model:
        axs = az.plot_pair(df,var_names=Vars,kind='kde',figsize=(corwidth,corheight))

    # replace labels with the nicer unicode character versions
    if len(Vars) > 2:
        # reshape axes so that we can loop through them
        axs = np.reshape(axs,np.shape(axs)[0]*np.shape(axs)[1])

        for ax in axs:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel:
                ax.set_xlabel(betterLabels(xlabel))
            if ylabel:
                ax.set_ylabel(betterLabels(ylabel))
    else:
        xlabel = axs.get_xlabel()
        ylabel = axs.get_ylabel()
        axs.set_xlabel(betterLabels(xlabel))
        axs.set_ylabel(betterLabels(ylabel))

    # show plot
    plt.show()

    # Posterior sample plot -----------------------------------------------------
    # Draw samples
    Ps, Vs, _, _ = drawPosteriorSamples(df,r,t,nDraws)
    # Plot them
    plotMCMC(Ps, Vs, Vexp, t, r, Pref)


def betterLabels(x):
    # replace strings with their corresponding (greek) symbols
    if type(x) == str:
        x = LabelLookup(x)
    else:
        for i in range(len(x)):
            x[i] = LabelLookup(x[i])

    return x

def LabelLookup(input):
    # look up table that contains the strings and their symbols
    if input == "lamb":
        return "λ"
    elif input == "sigma":
        return "σ"
    elif input == "delta":
        return "δ"
    elif input == "tau":
        return "τ" 
    elif input == "V0":
        return "V₀"
    elif input == "r0":
        return "r₀"
    elif input == "alpha":
        return "α"
    elif input == "lg_alpha":
        return "lg(α)"
    else:
        return input

def drawPosteriorSamples(df, r = np.linspace(2, 8,num = 200), t = np.linspace(0,3,num = 200), nDraws = 100):
    VarNames = df.varnames

    # Determine if a Gaussian model was used and how many iterations were run -------
    if 'r0' in VarNames:
        if df['r0'].ndim == 1:
            nGaussians = 1
        else:
            nGaussians = df['r0'].shape[1]

        nChainSamples = df['r0'].shape[0]

    else:
        nChainSamples = df['P'].shape[0]

    # Generate random indeces from chain samples ------------------------------------
    idxSamples = random.sample(range(nChainSamples),nDraws)

    # Draw P's -------------------------------------------------------------------
    Ps = []

    if 'r0' in VarNames:
        r0_vecs = df['r0'][idxSamples]
        w_vecs = df['w'][idxSamples]
        if nGaussians == 1:
            a_vecs = np.ones_like(idxSamples)
        else:
            a_vecs = df['a'][idxSamples]

        for iDraw in range(nDraws):
            P = dd_gauss(r,r0_vecs[iDraw],w_vecs[iDraw],a_vecs[iDraw])
            Ps.append(P)
    else:
        for iDraw in range(nDraws):
            P = df['P'][idxSamples[iDraw]]
            Ps.append(P)

    # Draw corresponding time domain parameters ---------------------------------
    if 'V0' in VarNames:
        V0_vecs = df['V0'][idxSamples]

    if 'k' in VarNames:
        k_vecs = df['k'][idxSamples]

    if 'lamb' in VarNames:   
        lamb_vecs = df['lamb'][idxSamples]

    # Generate V's from P's and other parameters --------------------------------
    Vs = []
    K0 = dl.dipolarkernel(t,r,integralop=False)
    dr = r[1] - r[0]

    for iDraw in range(nDraws):
        K_ = copy.copy(K0)

        # The below construction of the kernel only takes into account RVs that were actually sampled
        # During development the model was sometimes run with fixed values for λ, k, or V₀
        if 'lamb' in VarNames: 
            K_ = (1-lamb_vecs[iDraw]) + lamb_vecs[iDraw]*K_

        if 'k' in VarNames:
            B = bg_exp(t,k_vecs[iDraw])
            K_ = K_*B[:, np.newaxis]

        if 'V0' in VarNames:
            K_ = V0_vecs[iDraw]*K_

        K_ = K_*dr
        Vs.append(K_@Ps[iDraw])

    return Ps, Vs, t, r


def plotMCMC(Ps,Vs,Vdata,t,r, Pref = None):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(11)

    if min(Vs[0])<0.2:
        residuals_offset = -max(Vs[0])/3
    else:
        residuals_offset = 0

    for V,P in zip(Vs,Ps):
        ax1.plot(t, V, color = '#3F60AE', alpha=0.2)
        ax1.plot(t, Vdata-V+residuals_offset, color = '#3F60AE', alpha=0.2)
        ax2.plot(r, P, color = '#3F60AE', alpha=0.2)
    ax1.plot(t, Vdata , color = 'black')
    ax1.hlines(residuals_offset,min(t),max(t), color = 'black')

    ax1.set_xlabel('$t$ (µs)')
    ax1.set_ylabel('$V$ (arb.u)')
    ax1.set_xlim((min(t), max(t)))
    ax1.set_title('time domain and residuals')

    ax2.set_xlabel('$r$ (nm)')
    ax2.set_ylabel('$P$ (nm$^{-1}$)')
    ax2.set_xlim((min(r), max(r)))
    ax2.set_title('distance domain')

    if Pref is not None:
        ax2.plot(r, Pref , color = 'black')

    plt.grid()

    plt.show()