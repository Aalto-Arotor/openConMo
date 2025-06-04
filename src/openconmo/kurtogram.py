'''
Copyright (c) 2015, Jerome Antoni
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

#------------------------------------------------------------------------------
# kurtogram.py
#
# Implement the Fast Kurtogram Algorithm in Python
# 
#
# Created: 09/21/2019 - Daniel Newman -- danielnewman09@gmail.com
#
# Modified:
#   * 09/21/2019 - DMN -- danielnewman09@gmail.com
#           - Replicated Jerome Antoni's Fast Kurtogram algorithm in Python
#             https://www.mathworks.com/matlabcentral/fileexchange/48912-fast-kurtogram

#   * 17/10/2024 -- sampolaine
#           - Fixed bugs
#
#------------------------------------------------------------------------------

from scipy.signal import firwin
from scipy.signal import lfilter
import numpy as np
import matplotlib.pyplot as plt

import scipy.io

def fast_kurtogram(x,fs,nlevel=7, verbose=False):
    """
    (Compute the fast kurtogram of a signal using wavelet packet decomposition.

    This implementation is adapted from Jerome Antoni's fast kurtogram algorithm,
    and uses a recursive quadrature mirror filter bank to estimate spectral kurtosis
    over different frequency bands.) Needs to be analyzed further

    Parameters
    ----------
    x : ndarray
        Input signal (1D array). Should be a real-valued or complex-valued signal.
    fs : float
        Sampling frequency of the signal in Hz.
    nlevel : int, optional
        Number of decomposition levels (default is 7). Must be small enough to satisfy the signal length requirement.
    verbose : bool, optional
        If True, prints additional information about the maximum kurtosis and corresponding frequency.

    Returns
    -------
    Kwav : ndarray
        2D array representing the kurtosis map over the wavelet packet decomposition levels and subbands.
    Level_w : ndarray
        1D array of decomposition levels corresponding to the rows of `Kwav`.
    freq_w : ndarray
        1D array of center frequencies corresponding to the columns of `Kwav`.
    fc : float
        Center frequency (in Hz) of the subband with the highest kurtosis.
    bandwidth : float
        Bandwidth (in Hz) of the optimal subband with maximum kurtosis.

    Raises
    ------
    ValueError
        If the number of decomposition levels exceeds the limit based on input signal length.
    """
    N = x.flatten().size
    N2 = np.log2(N) - 7
    if nlevel > N2:
        raise ValueError('Please enter a smaller number of decomposition levels')
    x -= np.mean(x)
    N = 16
    fc = 0.4
    h = firwin(N+1,fc) * np.exp(2j * np.pi * np.arange(N+1) * 0.125)
    n = np.arange(2,N+2)
    g = h[(1-n) % N] * (-1.)**(1-n)
    N = int(np.fix(3/2*N))
    h1 = firwin(N+1,2/3 * fc) * np.exp(2j * np.pi * np.arange(0,N+1) * 0.25/3)
    h2 = h1 * np.exp(2j * np.pi * np.arange(0,N+1) / 6)
    h3 = h1 * np.exp(2j * np.pi * np.arange(0,N+1) / 3)
    Kwav = _K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt2')
    Kwav = np.clip(Kwav,0,np.inf)
    Level_w = np.arange(1,nlevel+1)
    Level_w = np.vstack((Level_w,
                         Level_w + np.log2(3)-1)).flatten()
    Level_w = np.sort(np.insert(Level_w,0,0)[:2*nlevel])
    freq_w = fs*(np.arange(3*2**nlevel)/(3*2**(nlevel+1)) + 1/(3*2**(2+nlevel)))

    max_level_index = np.argmax(Kwav[np.arange(Kwav.shape[0]),np.argmax(Kwav,axis=1)])
    max_kurt = np.amax(Kwav[np.arange(Kwav.shape[0]),np.argmax(Kwav,axis=1)])
    level_max = Level_w[max_level_index]

    bandwidth = fs*2**(-(Level_w[max_level_index] + 1))

    index = np.argmax(Kwav)
    index = np.unravel_index(index,Kwav.shape)
    l1 = Level_w[index[0]]
    fi = (index[1])/3./2**(nlevel+1)
    fi += 2.**(-2-l1)
    fc = fs*fi
    
    if verbose:
        print('Max Level: {}'.format(level_max))
        print('Freq: {}'.format(fi))
        print('Fs: {}'.format(fs))
        print('Max Kurtosis: {}'.format(max_kurt))
        print(f'Center frequency: {fc/1e3}')
        print('Bandwidth: {}'.format(bandwidth))


    return Kwav, Level_w, freq_w, fc, bandwidth

def _kurt(this_x,opt):
    """Compute kurtosis of a signal using method 'kurt1' or 'kurt2'."""
    eps = 2.2204e-16

    if opt.lower() == 'kurt2':
        if np.all(this_x == 0):
            K = 0
            return K
        this_x -= np.mean(this_x)

        E = np.mean(np.abs(this_x)**2)
        if E < eps:
            K = 0
            return K
        K = np.mean(np.abs(this_x)**4) / E**2
        if np.all(np.isreal(this_x)):
            K -= 3
        else:
            K -= 2
    elif opt.lower() == 'kurt1':
        if np.all(this_x == 0):
            K = 0
            return K
        x -= np.mean(this_x)
        E = np.mean(np.abs(this_x))
        if E < eps:
            K = 0
            return K
        K = np.mean(np.abs(this_x)**2) / E**2
        if np.all(np.isreal(this_x)):
            K -= 1.57
        else:
            K -= 1.27

    return K


def _K_wpQ(x,h,g,h1,h2,h3,nlevel,opt,level=None):
    """
    Compute the kurtosis matrix K from a binary-ternary wavelet packet transform of a signal.

    Performs decomposition of the input signal x up to nlevel using lowpass
    and highpass filters (h and g). The resulting kurtosis values K are sorted
    according to the frequency decomposition.
    """

    if level == None:
        level = nlevel
    x = x.flatten()
    L = np.floor(np.log2(x.size))
    x = np.atleast_2d(x).T
    KD,KQ = _K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level)

    K = np.zeros((2 * nlevel,3 * 2**nlevel))

    K[0,:] = KD[0,:]

    for i in np.arange(1,nlevel):
        K[2*i-1,:] = KD[i,:]
        K[2*i,:] = KQ[i-1,:]

    K[2*nlevel-1,:] = KD[nlevel,:]

    return K

def _K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level):
    """
    Recursive helper for computing the kurtosis tree at each level.

    Decomposes signal using binary and ternary filter banks and collects
    kurtosis values for each node in the tree.
    """

    a,d = _DBFB(x,h,g)
    N = np.amax(a.shape)
    d = d * (-1)**(np.atleast_2d(np.arange(1,N+1)).T)

    Lh = np.amax(h.shape)
    Lg = np.amax(g.shape)
    K1 = _kurt(a[Lh-1:],opt)
    K2 = _kurt(d[Lg-1:],opt)

    if level > 2:
        # print("hello 1")
        a1,a2,a3 = _TBFB(a,h1,h2,h3)
        d1,d2,d3 = _TBFB(d,h1,h2,h3)
        Ka1 = _kurt(a1[Lh-1:],opt)
        Ka2 = _kurt(a2[Lh-1:],opt)
        Ka3 = _kurt(a3[Lh-1:],opt)
        Kd1 = _kurt(d1[Lh-1:],opt)
        Kd2 = _kurt(d2[Lh-1:],opt)
        Kd3 = _kurt(d3[Lh-1:],opt)
    else:
        # print("hello 2")
        Ka1 = 0
        Ka2 = 0
        Ka3 = 0
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0

    if level == 1:
        # print("hello 3")
        K = np.concatenate((K1 * np.ones(3),K2 * np.ones(3)))
        KQ = np.array([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3])

    if level > 1:
        # print("hello 4")
        Ka,KaQ = _K_wpQ_local(a,h,g,h1,h2,h3,nlevel,opt,level-1)
        Kd,KdQ = _K_wpQ_local(d,h,g,h1,h2,h3,nlevel,opt,level-1)
        K1 *= np.ones(np.amax(Ka.shape))
        K2 *= np.ones(np.amax(Kd.shape))
        K = np.vstack((np.concatenate([K1,K2]),
                       np.hstack((Ka,Kd))))

        Long = int(2/6 * np.amax(KaQ.shape))
        Ka1 *= np.ones(Long)
        Ka2 *= np.ones(Long)
        Ka3 *= np.ones(Long)
        Kd1 *= np.ones(Long)
        Kd2 *= np.ones(Long)
        Kd3 *= np.ones(Long)

        KQ = np.vstack((np.concatenate([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3]),
                        np.hstack((KaQ,KdQ))))


    if level == nlevel:
        # print("hello 5")
        K1 = _kurt(x,opt)
        # print(K1.shape)
        # print(K.shape)
        K = np.vstack((K1 * np.ones(np.amax(K.shape)),K))
        a1,a2,a3 = _TBFB(x,h1,h2,h3)
        Ka1 = _kurt(a1[Lh-1:],opt)
        Ka2 = _kurt(a2[Lh-1:],opt)
        Ka3 = _kurt(a3[Lh-1:],opt)
        Long = int(1/3 * np.amax(KQ.shape))
        Ka1 *= np.ones(Long)
        Ka2 *= np.ones(Long)
        Ka3 *= np.ones(Long)
        KQ = np.vstack((np.concatenate([Ka1,Ka2,Ka3]),
                        KQ[:-2,:]))

    return K,KQ


def _TBFB(x,h1,h2,h3):
    """Apply ternary filter bank to input signal and return 3 subbands."""
    N = x.flatten().size
    a1 = lfilter(h1,1,x.flatten())
    a1 = a1[2:N:3]
    a1 = np.atleast_2d(a1).T

    a2 = lfilter(h2,1,x.flatten())
    a2 = a2[2:N:3]
    a2 = np.atleast_2d(a2).T

    a3 = lfilter(h3,1,x.flatten())
    a3 = a3[2:N:3]
    a3 = np.atleast_2d(a3).T

    return a1,a2,a3


def _DBFB(x,h,g):
    """Apply binary filter bank to input signal and return low and high subbands."""
    N = x.flatten().size
    a = lfilter(h,1,x.flatten())
    a = a[1:N:2]
    a = np.atleast_2d(a).T
    d = lfilter(g,1,x.flatten())
    d = d[1:N:2]
    d = np.atleast_2d(d).T

    return a,d

def binary(i,k):
    """
    Convert an integer to its binary representation with fixed length.

    Parameters
    ----------
    x : int
        The integer to convert.
    n : int
        The number of binary digits (bits) in the output.

    Returns
    -------
    list of int
        A list of binary digits (0 or 1), with length `n`, representing `x`.
        Most significant bit is first.
    """
    k = int(k)
    if i > 2**k:
        raise ValueError('i must be such that i < 2^k')
    a = np.zeros(k)
    temp = i
    for l in np.arange(k)[::-1]:
        a[-(l+1)] = np.fix(temp / 2**l)
        temp -= a[-(l+1)] * 2 ** l

    return a

def plot_kurtogram(data, sampling_rate, nlevel=9, verbose=False):
    """
    Plots the kurtogram for a given signal data.

    Parameters
    ----------
    data : array_like
        Input signal data (should be a 1D array or a 2D array).
    
    sampling_rate : float
        Sampling rate of the signal in Hz.
    
    nlevel : int, optional
        The number of decomposition levels for the kurtogram. Default is 9.
    
    verbose : bool, optional
        If True, prints additional information. Default is False.

    Returns
    -------
    None
        This function does not return any value, but generates a plot of the kurtogram.
    """
    # Run the fast kurtogram function
    Kwav, Level_w, freq_w, fc, bandwidth = fast_kurtogram(data, sampling_rate, nlevel=nlevel, verbose=verbose)

    # Calculate the transformation for the y-axis 
    Level_w_transformed = np.array([0] + [(i + np.log2(3) - 1) for i in range(1, 2 * len(Level_w))])

    # Define the frequency in kHz for the x-axis
    freq_w_kHz = freq_w / 1000  # Convert frequency from Hz to kHz

    # Calculate the window lengths based on 2^(n+1)
    win_lengths = [2**(Level_w[i]+1) for i in range(len(Level_w))]

    # Create y-tick labels (Decomposition level and corresponding window length)
    ytick_labels = [f'{lvl:.1f} ({wl:.0f})' for lvl, wl in zip(Level_w, win_lengths)]

    # Plot the kurtogram
    plt.figure(figsize=(10, 6))

    # Use imshow to plot the kurtogram (with aspect ratio and proper extent)
    im = plt.imshow(Kwav, aspect='auto',
                    extent=(freq_w_kHz[0], freq_w_kHz[-1], 0, len(Level_w)-1),
                    interpolation='none', origin='upper')  # Origin at the top (to match MATLAB)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Spectral Kurtosis', fontsize=14)

    # Label x and y axes
    plt.xlabel('Frequency [kHz]', fontsize=12)
    plt.ylabel('Decomposition Level (Window Length)', fontsize=12)

    # Title with dynamic values
    max_level_index = np.argmax(Kwav[np.arange(Kwav.shape[0]), np.argmax(Kwav, axis=1)])
    max_kurt = np.amax(Kwav[np.arange(Kwav.shape[0]), np.argmax(Kwav, axis=1)])
    level_max = Level_w[max_level_index]
    Bw = sampling_rate * 2**-(Level_w[0] + 1)  # Bandwidth calculation

    plt.title(f'K_max={max_kurt:.4f} at level {level_max:.1f}, Optimal Window Length={2**(Level_w[max_level_index]+1):.0f}, Center Frequency={fc/1000:.4f} kHz, Bandwidth={bandwidth/1000:.5f} kHz', fontsize=10, pad=10)

    # Set custom y-ticks based on decomposition levels and window lengths
    yticks_position = np.linspace(len(Level_w)-1 - 0.5, 0.5, len(Level_w))
    plt.yticks(ticks=yticks_position, labels=ytick_labels)

    # Display the plot
    plt.tight_layout()
    plt.show()