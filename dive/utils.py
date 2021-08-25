import numpy as np
import math as m
import sys
from scipy.special import fresnel
import pymc3 as pm
from datetime import date
import os   

from .constants import *
from .deerload import *
from .samplers import *


def addnoise(V,sig):
    """
    Add Gaussian noise with standard deviation sig to signal
    """
    noise = np.random.normal(0, sig, np.size(V))
    Vnoisy = V + noise
    return Vnoisy


def FWHM2sigma(FWHM):
    """
    Convert the full width at half maximum, FWHM, of a Gaussian to the standard deviation, sigma.
    """
    sigma = FWHM/(2*m.sqrt(2*m.log(2)))

    return sigma


def sigma2FWHM(sigma):
    """
    Convert the standard deviation, sigma, of a Gaussian to the full width at half maximum, FWHM.
    """
    FWHM = sigma/(2*m.sqrt(2*m.log(2)))

    return FWHM


def dipolarkernel(t,r):
    """
    K = dipolarkernel(t,r)
    Calculate dipolar kernel matrix.
    Assumes t in microseconds and r in nanometers
    """
    omega = 1e-6 * D/(r*1e-9)**3 # rad µs^-1
    
    # Calculation using Fresnel integrals
    nr = np.size(r)
    nt = np.size(t)
    K = np.zeros((nt, nr))
    for ir in range(nr):
        ph = omega[ir]*np.abs(t)
        z = np.sqrt(6*ph/m.pi)
        S, C = fresnel(z)
        K[:,ir] = (C*np.cos(ph)+S*np.sin(ph))/z
    
    K[t==0,:] = 1   # fix div by zero
    
    # Include delta-r factor for integration
    if len(r)>1:
        dr = np.mean(np.diff(r))
        K *= dr
    
    return K


def loadTrace(FileName):
    """
    Load a DEER trace, can be a bruker or comma separated file.
    """
    if FileName.endswith('.dat') or FileName.endswith('.txt') or FileName.endswith('.csv'):
        data = np.genfromtxt(FileName,delimiter=',',skip_header=1)
        t = data[:,0]
        Vdata = data[:,1]
    elif FileName.endswith('.dat'):
        t, Vdata, Parameters = deerload(FileName)
    else:
        sys.exit('The file format is not recognized.')

    return t, Vdata


def sample(model_dic, MCMCparameters, steporder=None, NUTSorder=None, NUTSpars=None):
    """ 
    Use PyMC3 to draw samples from the posterior for the model, according to the parameters provided with MCMCparameters.
    """  
    
    # Complain about missing required keywords
    requiredKeys = ["draws", "tune", "chains"]
    for key in requiredKeys:
        if key not in MCMCparameters:
            raise KeyError(f"The required MCMC parameter '{key}' is missing.")
    
    # Supplement defaults for optional keywords
    defaults = {"cores": 2, "progressbar": True, "return_inferencedata": False}
    MCMCparameters = {**defaults, **MCMCparameters}
    
    model = model_dic['model']
    model_pars = model_dic['pars']
    method = model_pars['method']
    
    # Set stepping methods, depending on model
    if method == "gaussian":
        
        removeVars  = ["r0_rel"]
        
        with model:
            if model_pars['ngaussians']==1:
                NUTS_varlist = [model['r0_rel'], model['w'], model['sigma'], model['k'], model['V0'], model['lamb']]
            else:
                NUTS_varlist = [model['r0_rel'], model['w'], model['a'], model['sigma'], model['k'], model['V0'], model['lamb']]
            if NUTSorder is not None:
                NUTS_varlist = [NUTS_varlist[i] for i in NUTSorder] 
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
                
        step = step_NUTS
        
    elif method == "regularization":
        
        removeVars = None
        
        with model:
            NUTS_varlist = [model['k'], model['V0'], model['lamb']]
            if NUTSorder is not None:
                NUTS_varlist = [NUTS_varlist[i] for i in NUTSorder] 
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
            
            step_tau = randTau_posterior(model['tau'], model_pars['tau_prior'], model_pars['K0'], model['P'], model_dic['Vexp'], model_pars['r'], model_dic['t'], model['k'], model['lamb'], model['V0'])
            step_P = randPnorm_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['k'], model['lamb'], model['V0'])
            step_delta = randDelta_posterior(model['delta'], model_pars['delta_prior'], model_pars['L'], model['P'])
        
        step = [step_P, step_tau, step_delta, step_NUTS]
        if steporder is not None:
            step = [step[i] for i in steporder]
                
    elif method == "regularization2":
        
        removeVars = None
        
        with model:
            NUTS_varlist = [model['tau'], model['delta'], model['k'], model['V0'], model['lamb']]
            if NUTSorder is not None:
                NUTS_varlist = [NUTS_varlist[i] for i in NUTSorder] 
            if NUTSpars is None:
                step_NUTS = pm.NUTS(NUTS_varlist)
            else:
                step_NUTS = pm.NUTS(NUTS_varlist, **NUTSpars)
            
            step_P = randPnorm_posterior(model['P'], model_pars['K0'] , model_pars['LtL'], model_dic['t'], model_dic['Vexp'], model_pars['r'], model['delta'], [], model['tau'], model['k'], model['lamb'], model['V0'])
        
        step = [step_P, step_NUTS]
        if steporder is not None:
            step = [step[i] for i in steporder]
                
    else:
        
        raise KeyError(f"Unknown method '{method}'.",method)

    # Perform MCMC sampling
    trace = pm.sample(model=model, step=step, **MCMCparameters)

    # Remove undesired variables
    if removeVars is not None:
        [trace.remove_values(key) for key in removeVars if key in trace.varnames]

    return trace


def saveTrace(df, Parameters, SaveName='empty'):
    """
    Save a trace to a CSV file.
    """
    if SaveName == 'empty':
        today = date.today()
        datestring = today.strftime("%Y%m%d")
        SaveName = "./traces/{}_traces.dat".format(datestring)
    
    if not SaveName.endswith('.dat'):
        SaveName = SaveName+'.dat'

    shape = df.shape 
    cols = df.columns.tolist()

    os.makedirs(os.path.dirname(SaveName), exist_ok=True)

    f = open(SaveName, 'a+')
    f.write("# Traces from the MCMC simulations with pymc3\n")
    f.write("# The following {} parameters were investigated:\n".format(shape[1]))
    f.write("# {}\n".format(cols))
    f.write("# nParameters nChains nIterations\n")
    if Parameters['nGauss'] == 1:
        f.write("{},{},{},0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 2:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 3:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))
    elif Parameters['nGauss'] == 4:
        f.write("{},{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n".format(shape[1],Parameters['chains'],Parameters['draws']))

    df.to_csv (f, index=False, header=False)

    f.close()
