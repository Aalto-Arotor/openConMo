import glob

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fft
from tqdm import tqdm
from scipy.fft import fft, ifft
from kurtogram import fast_kurtogram
import utils as ut


def randall_method_1(signal, fs):
    '''
    Simply the envelope analysis (squared envelope spectrum) of the full
    bandwidth raw signal. The justification is that there are no obvious
    masking sources in this case, and in fact many bearing faults were
    found to be easily diagnosable with this method. For such cases it
    means that the signals are not very demanding in terms of diagnostic
    power, and do not provide a good test for newly proposed algorithms
    (unless for specific purposes, such as the determination of fault size).

    inputs:
        signal: Sampled signal
        fs:     Sampling frequency
    outputs:
        f:      Squared envelope spectrum frequency
        X:      Squared envelope spectrum amplitude
    '''
    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2
    f, X = ut.oneside_fft(squared_envelope, fs)

    return f, X


def DRS(signal, N, Delta):
    '''
    Step 1 of Randall's method 3
    Discrete/random separation (DRS) to remove deterministic (discrete
    frequency) components.

    Implemented as described in
    Unsupervised noise cancellation for vibration signals:
    part II—a novel frequency-domain algorithm
    '''
    # nom and denom of H
    nominator = np.zeros(N, dtype=complex)
    denominator = np.zeros(N, dtype=complex)
    for k in tqdm(range(2*N+Delta, len(signal),100), desc="Building DRS filter"):
        window_idx = np.arange(k-N, k)
        delayed_window_idx = np.arange(k-2*N-Delta, k-N-Delta)

        # N length window of signal
        x = signal[window_idx]*scipy.signal.windows.hann(N)
        # N length time delayed window of signal
        x_d = signal[delayed_window_idx]*scipy.signal.windows.hann(N)

        X_k_f = scipy.fft.fft(x)
        X_d_k_f = scipy.fft.fft(x_d)

        #  Cross-power spectrum between sequence and delayed sequence
        nominator += X_d_k_f * X_k_f.conj()
        # Power spectrum of delayed sequence
        denominator += X_d_k_f * X_d_k_f.conj()

    # Compute the filter H(f) in the frequency domain
    H_f = nominator / denominator
    # Transform the frequency-domain filter back to the time domain
    h_n = scipy.fft.ifft(H_f).real  # Impulse response of the filter

    # Transform the frequency-domain filter back to the time domain
    h_n = scipy.fft.ifft(H_f).real  # Impulse response of the filter

    # Applies time domain filter to original signal
    # The phase correction is achieved by correctly aligning the delayed part
    # of the signal with the filter response
    deterministic_part = np.zeros(len(signal))
    random_part = np.zeros(len(signal))
    for i in tqdm(range(N+Delta, len(signal), 1), desc="Computing the DRS"):
        deterministic_part[i] = signal[i-N-Delta:i-Delta]@h_n
        random_part[i] = signal[i] - deterministic_part[i]

    return random_part

def randall_method_2(signal, fs):
    print("Inside randall_method_2")

    t = np.linspace(0, len(signal)/fs, len(signal))
    X = scipy.fft.fft(signal)
    phase = np.angle(X)

    log_amplitude = np.log(np.abs(X)+ 1e-10)

    real_cepstrum = scipy.fft.ifft(log_amplitude)

    edited_cepstrum = np.zeros_like(real_cepstrum)
    edited_cepstrum[0] = real_cepstrum[0]

    edited_log_amplitude_spectrum = scipy.fft.fft(edited_cepstrum)

    edited_log_spectrum = edited_log_amplitude_spectrum + 1j * phase

    X = np.exp(edited_log_spectrum)

    # time_signal = scipy.fft.ifft(X)

    time_signal = np.real(scipy.fft.ifft(X))

    return t, time_signal

'''def randall_method_22(signal, fs):
    print("TESTING 2")
    t = np.linspace(0, len(signal)/fs, len(signal))
    x_prewithened = np.abs(ifft(np.log(np.absolute(fft(signal)))))
    return t, x_prewithened'''

def randall_method_3(signal, fs, N=16384, Delta=500, nlevel=2):
    '''
    1. Discrete/random separation (DRS) to remove deterministic
    (discrete frequency) components.
    2. Spectral kurtosis to determine the most impulsive band,
    followed by bandpass filtering.
    3. Envelope analysis (squared envelope spectrum) of the
    bandpass filtered signal.
    ''' # 1. DRS Filtering
    filtered_signal = DRS(signal, N, Delta)[N+Delta:]
    t = np.linspace((N+Delta)/fs, len(signal)/fs, len(signal)-N-Delta)
    
    """  # 2. Kurtogram band selection and bandpass filtering
    _, _, _, fc, bandwidth = fast_kurtogram(filtered_signal, fs, nlevel=nlevel)
    filtered_signal = ut.bandpass_filter(filtered_signal, fs, fc, bandwidth)"""

    # 3. Squared envelope spectrum
    analytic_signal = scipy.signal.hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2
    f, X = ut.oneside_fft(squared_envelope, fs)

    return t, filtered_signal, f, X


def plot_kurtogram(Kwav, freq_w, level_w):
    plt.imshow(Kwav, aspect='auto',
               extent=(freq_w[0], freq_w[-1], level_w[0], level_w[-1]),
               interpolation='none')
    plt.colorbar()
    plt.savefig("kurtogram.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    mat_files = glob.glob("matlab_example_data/outer_fault.mat", recursive=True)
    print("\n".join(mat_files))

    fs = 97656
    signal = scipy.io.loadmat(mat_files[0])['xOuter'].ravel()
    print(signal)
    time = np.linspace(0, len(signal)/fs, len(signal))
    plt.plot(time, signal, c="b", linewidth=0.5)
    plt.xlim(0, 0.1)
    plt.show()

    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2
    f, X = ut.oneside_fft(squared_envelope, fs)

    plt.xlim(0, 1000)
    print(f[0], f[-1])
    f, X = scipy.signal.periodogram(squared_envelope, fs, window="hann")
    X = np.sqrt(X)
    plt.plot(f, X, c="b", linewidth=0.5)
    plt.show()

