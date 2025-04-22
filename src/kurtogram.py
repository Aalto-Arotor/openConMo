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
    '''
    Computes the kurtosis K of the complete "binary-ternary" wavelet
    packet transform w of signal x, up to nlevel, using the lowpass
    and highpass filters h and g, respectively. The values in K are
    sorted according to the frequency decomposition.
    '''
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
    N = x.flatten().size
    a = lfilter(h,1,x.flatten())
    a = a[1:N:2]
    a = np.atleast_2d(a).T
    d = lfilter(g,1,x.flatten())
    d = d[1:N:2]
    d = np.atleast_2d(d).T

    return a,d

def binary(i,k):
    k = int(k)
    if i > 2**k:
        raise ValueError('i must be such that i < 2^k')
    a = np.zeros(k)
    temp = i
    for l in np.arange(k)[::-1]:
        a[-(l+1)] = np.fix(temp / 2**l)
        temp -= a[-(l+1)] * 2 ** l

    return a


if __name__ == "__main__":


    data = scipy.io.loadmat('src/matlab_example_data/outer_fault.mat')
    data = data['xOuter']
    samplingRate = 97656

    Kwav, Level_w, freq_w, fc, bandwidth = fast_kurtogram(data, samplingRate, nlevel=9)
    print(Kwav.shape)
    # Plotting the kurtogram

    im = plt.imshow(Kwav, aspect='auto',
               extent=(freq_w[0], freq_w[-1], Level_w[0], Level_w[-1]),
               interpolation='none')
    cbar = plt.colorbar(im)
    cbar.set_label('Kurtosis', fontsize=14)

    plt.title('Kurtogram and Spectral Kurtosis for Band Selection', fontsize=10, pad = 10)

    plt.show()
    
    '''pass
    # data = np.genfromtxt('aoyu_example_data.txt',skip_header=23,delimiter='\n')
    # samplingRate = 100e3
    # speed = 1000/60
    # Kwav, Level_w, freq_w, fc, bandwidth = fast_kurtogram(data, samplingRate, nlevel=6)
    # plt.imshow(np.clip(Kwav,0,np.inf),aspect=10, interpolation="none")
    # plt.show()'''
