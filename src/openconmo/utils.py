import scipy
import numpy as np

def oneside_fft(x, fs):
    """
    Compute the one-sided FFT (magnitude spectrum) of a real-valued signal.

    This function returns the positive frequency components of the discrete
    Fourier transform (DFT) of the input signal, scaled appropriately for
    power analysis.

    Parameters
    ----------
    x : ndarray
        Input time-domain signal (real-valued).
    fs : float
        Sampling frequency of the signal in Hz.

    Returns
    -------
    freqs : ndarray
        Array of positive frequency values in Hz.
    spectrum : ndarray
        One-sided magnitude spectrum of the input signal.
    """
    # calculate DFT
    n = len(x)
    freqs = np.fft.fftfreq(n, d=1/fs)
    fft_values = np.fft.fft(x)

    # scale
    fft_values = fft_values[:n // 2] / n
    fft_values[1:] *= 2

    return freqs[:n // 2], np.abs(fft_values)


def bandpass_filter(signal, fs, fc, BW, order=200):
    """
    Designs and applies a bandpass FIR filter to the input signal.

    The filter is centered at the given frequency `fc` and spans the specified
    bandwidth `BW`. It uses a linear-phase FIR design with the given filter order.

    Parameters
    ----------
    signal : ndarray
        Input time-domain signal (1D array).
    fs : float
        Sampling frequency of the signal in Hz.
    fc : float
        Center frequency of the bandpass filter in Hz.
    BW : float
        Bandwidth of the bandpass filter in Hz.
    order : int, optional
        Filter order (default is 200). Determines the sharpness of the filter.
    
    Returns
    -------
    filtered_signal : ndarray
        The signal after applying the bandpass FIR filter.
    """
    # Calculate the Nyquist frequency
    nyquist = fs / 2

    # Compute normalized cutoff frequencies and clip to safe range
    # Avoid 0 and 1 to prevent firwin from throwing errors
    lowcut = np.max([0.000000001, (fc - BW / 2) / nyquist])
    highcut = np.min([0.99999999, (fc + BW / 2) / nyquist])

    # Debug prints (can be removed in production)
    print('Fs: {}'.format(fs))
    print('Fc: {}'.format(fc))
    print('Nyquist: {}'.format(nyquist))
    print('Lowcut: {}'.format(lowcut))
    print('Highcut: {}'.format(highcut))
    print('signal : {}'.format(signal))

    # Design the FIR filter using the window method
    taps = scipy.signal.firwin(order + 1, [lowcut, highcut], pass_zero=False)

    # Apply FIR filter using direct-form linear convolution
    filtered_signal = scipy.signal.lfilter(taps, 1.0, signal)

    return filtered_signal


def downsample(x, fs_now, fs_resampled):
    """
    Downsamples or upsamples a signal to a new sampling frequency.

    This function resamples the input signal `x` from its current sampling
    frequency `fs_now` to a new target sampling frequency `fs_resampled`.
    The signal is resampled by changing the number of samples based on the
    ratio of the two frequencies.

    Parameters
    ----------
    x : np.ndarray
        Input signal to be resampled, of shape (N,).
    fs_now : float
        Current sampling frequency of the input signal `x`.
    fs_resampled : float
        Target sampling frequency to which the signal should be resampled.

    Returns
    -------
    np.ndarray
        The resampled signal with a new number of samples based on the ratio 
        `fs_resampled / fs_now`.
    """
    # Get the current length of the signal
    N_now = len(x)

    # Calculate the target number of samples after resampling
    # Proportional to the ratio of new sampling rate to the original rate
    N_resampled = int(fs_resampled/fs_now * N_now)

    # Resample the signal to the target number of samples
    return scipy.signal.resample(x, N_resampled)


def vandermonde(t, w0, L):
    """
    Constructs a Vandermonde-like matrix for sinusoidal modeling.

    Parameters
    ----------
    t : np.ndarray
        Time vector of shape (N,), representing the time points at which the
        model is evaluated.
    w0 : float
        Base angular frequency (in radians per second) for generating the sine
        and cosine terms.
    L : int
        Model order, which determines the number of harmonics (terms) in the
        Vandermonde matrix. The matrix will include terms for the first `L`
        harmonics.

    Returns
    -------
    Z : np.ndarray
        The Vandermonde-like matrix of shape (N, 2*L), containing both sine and
        cosine terms for each harmonic frequency up to `L`. The first `L`
        columns are sine terms, and the last `L` columns are cosine terms.
   """
    # Compute cosine terms for harmonics 1 to L
    c = np.cos(np.outer(t*w0, np.arange(1, L+1)))

    # Compute sine terms for harmonics 1 to L
    s = np.sin(np.outer(t*w0, np.arange(1, L+1)))

    # Stack sine and cosine terms horizontally to form the final matrix
    # Shape: (N, 2L) where first L coumns are sine, last L are cosine
    Z = np.hstack((s, c))

    return Z


def nls(x, L, t, fs, f_start, f_end, fast=True, num_points=5000):
    """
    Non-linear least squares (NLS) estimation to fit a signal to a specified
    model.

    Parameters
    ----------
    x : np.ndarray
        Input signal to be fitted, of shape (N, 1).
    L : int
        Model order, which determines the number of harmonics
    t : np.ndarray
        Time vector of shape (N,), representing the time points of the signal.
    fs : float
        Sampling frequency of the signal `x`.
    fast : bool, optional
        If True, uses a faster but approximate computation of the cost function
        `J`. Default is True.
    f_start : float
        Starting frequency for the output
    f_end : float
        End frequency for the output
    num_points : int, optional
        Number of frequency points in the grid for `f0_grid`. Default is 5000.

    Returns
    -------
    f0_grid : np.ndarray
        The frequency grid used for fitting, of shape (num_points,).
    J : np.ndarray
        The computed cost function values for each frequency in `f0_grid`,
        of shape (K,).
    """
    # Ensure x is a column vector and initialize the cost arry
    x, J = x.reshape(-1, 1), np.zeros(num_points)

    # Create a frequency grid from f_start to f_end
    f0_grid = np.linspace(f_start, f_end, num_points)

    # Iterate through each frequency candidate
    for i, f0 in enumerate(f0_grid):
        # Construct the model matrix (harmonic basis) for the current frequency
        Z = vandermonde(t, f0*2*np.pi, L)

        # Solve the least squares problem to get the model coefficients
        a_hat = scipy.linalg.lstsq(Z, x, lapack_driver="gelsy")[0]

        # Compute cost function J: optionally use fast approximation
        if fast:
            J[i] = (x.T @ Z @ np.eye(Z.shape[1], Z.shape[1]) @ Z.T @ x)
        else:
            # Exact cost using estimated coefficients
            J[i] = x @ Z @ a_hat

    return f0_grid, J
