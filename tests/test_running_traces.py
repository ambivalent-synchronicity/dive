import dive
import pymc as pm
import deerlab as dl
import numpy as np

import pytest

@pytest.fixture
def check_trace(method, nGauss=0):
    # setting up data
    t = np.linspace(-0.1,5,300)
    r = np.linspace(2,7,251) 
    P = dl.dd_gauss(r,5,0.8)

    lam = 0.4                       # modulation depth
    B = dl.bg_hom3d(t,0.2,lam)         # background decay
    K = dl.dipolarkernel(t,r,mod=lam,bg=B)  # kernel matrix

    Vexp = K@P + dl.whitegaussnoise(t,0.01,seed=10)

    # setting up trace
    if method == "gaussian":
        pars = {"method": method, "r": np.linspace(2,7,50), "progressbar": False, "nGauss": nGauss}
    else:
        pars = {"method": method, "r": np.linspace(2,7,50)}
    MCMCparameters = {"draws": 100,"tune": 200, "chains": 1, "progressbar": False, "cores": 1}
    model_dictionary = dive.model(t, Vexp, pars)

    # running trace
    trace = dive.sample(model_dictionary, MCMCparameters, seed=100)
    return trace.posterior["lamb"][0][0].values

def test_running_regularization():
    assert check_trace("regularization") == 0.39950062979541884

def test_running_regularizationP():
    assert check_trace("regularizationP") == 0.40798238859993907

def test_running_gaussian_1():
    assert check_trace("gaussian", nGauss=1) == 0.4011008024924596

def test_running_gaussian_2():
    assert check_trace("gaussian", nGauss=2) == 0.39991417362040915