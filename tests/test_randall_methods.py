import sys
sys.path.append('../')

import pytest
import glob
import numpy as np
import src.data_utils as du

from src.randall_methods import randall_method_1, randall_method_2, randall_method_3

def test_randall_method_1():
    '''Test the first Randall method implementation'''
    fs = 12e3
    mat_files = glob.glob("CWRU-dataset/**/*209*.mat", recursive=True)
    signal, _, _, rpm = du.extract_signals(mat_files[0], normal=False)

    sq_env_f, sq_env = randall_method_1(signal, fs)
    actual = sq_env.take(5)
    expected = sq_env_f.take(5)
    # np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)

def test_randall_method_2():
    '''Test the second Randall method implementation'''
    actual = ...
    expected = ...
    # np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    #np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)

def test_randall_method_3():
    '''Test the third Randall method implementation'''
    actual = ...
    expected = ...
    # np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    #np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)

