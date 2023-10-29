import pytest
import numpy as np
from scipy.io import loadmat

def test_load_t():
    loaded_data = np.genfromtxt(f"data/3992_bad.dat", skip_header=1, delimiter=',')
    t = loaded_data[:,0]
    assert t[17] == 0.17

def test_load_Vexp():
    loaded_data = np.genfromtxt(f"data/3992_bad.dat", skip_header=1, delimiter=',')
    Vexp = loaded_data[:,1]
    assert Vexp[17] == 0.88773

def test_load_rref():
    rref = np.squeeze(loadmat('data/edwards_testset/distributions_2LZM')['r0'])
    assert rref[17] == 0.585

def test_laod_Pref():
    P0s = loadmat('data/edwards_testset/distributions_2LZM',verify_compressed_data_integrity=False)['P0']
    Pref = P0s[3991,:]
    assert Pref[17] == 1.3106523429764491e-170