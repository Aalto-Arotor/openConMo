# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.openconmo.benchmark_methods import envelope, cepstrum_prewhitening, benchmark_method

from openconmo.benchmark_methods import envelope, cepstrum_prewhitening, benchmark_method
import numpy as np
import plotly.graph_objects as go
from scipy.signal import hilbert

def create_dummy_figure(title):
    """Create an empty placeholder Plotly figure with a title and styled layout."""
    return {
        'data': [],
        'layout': {
            'title': title,
            'height': 300,
            'border': '1px solid black',
            'plot_bgcolor': '#f0f0f0',
            'paper_bgcolor': '#f0f0f0',
        }
    }

def create_time_series_plot(signal, fs, title="Time Series", unit=""):
    """
    Create a time series plot using Plotly.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Signal data to plot
    fs : float
        Sampling frequency in Hz
    title : str, optional
        Plot title
    unit : str, optional
        Unit of measurement for y-axis
    
    Returns:
    --------
    dict
        Plotly figure dictionary
    """
    # Create time array
    time = np.arange(len(signal)) / fs
    
    # Create figure
    fig = {
        'data': [{
            'x': time,
            'y': signal,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Signal',
            'line': {'color': '#1f77b4'}
        }],
        'layout': {
            'title': title,
            'xaxis': {
                'title': {
                    'text': 'Time (s)',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'yaxis': {
                'title': {
                    'text': f'Amplitude ({unit})',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified',
            'height': 200,
        }
    }
    
    return fig

def create_frequency_domain_plot(signal, fs, title="Frequency Spectrum", unit=""):
    """
    Create a frequency domain plot using Plotly.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Signal data to plot
    fs : float
        Sampling frequency in Hz
    title : str, optional
        Plot title
    unit : str, optional
        Unit of measurement
    
    Returns:
    --------
    dict
        Plotly figure dictionary
    """
    # Compute FFT
    n = len(signal)
    signal = signal - np.mean(signal)
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(n, d=1/fs)
    
    # Compute magnitude spectrum
    magnitude = 2/n * np.abs(fft)
    
    # Create figure
    fig = {
        'data': [{
            'x': freq,
            'y': magnitude,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Spectrum',
            'line': {'color': '#2ca02c'},
            'hovertemplate': 'Frequency: %{x:.2f} Hz<br>Magnitude: %{y:.2e} ' + unit + '<extra></extra>'
        }],
        'layout': {
            'title': title,
            'xaxis': {
                'title': {
                    'text': 'Frequency (Hz)',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'yaxis': {
                'title': {
                    'text': f'Magnitude ({unit})',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
        }
    }
    
    return fig

def create_envelope_spectrum_plot(signal, fs, title="Envelope Spectrum", unit=""):
    """
    Create an envelope spectrum plot using Plotly.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Signal data to plot
    fs : float
        Sampling frequency in Hz
    title : str, optional
        Plot title
    unit : str, optional
        Unit of measurement
    
    Returns:
    --------
    dict
        Plotly figure dictionary
    """
    # Remove DC component
    signal = signal - np.mean(signal)
    
    # Compute analytic signal using Hilbert transform
    analytic_signal = hilbert(signal)
    
    # Compute envelope
    envelope = np.abs(analytic_signal)
    
    # Remove DC from envelope
    envelope = envelope - np.mean(envelope)
    
    # Compute FFT of envelope
    n = len(envelope)
    fft = np.fft.rfft(envelope)
    freq = np.fft.rfftfreq(n, d=1/fs)
    
    # Compute magnitude spectrum
    magnitude = 2/n * np.abs(fft)
    
    # Create figure
    fig = {
        'data': [{
            'x': freq,
            'y': magnitude,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Envelope Spectrum',
            'line': {'color': '#ff7f0e'},  # Different color from regular spectrum
            'hovertemplate': 'Frequency: %{x:.2f} Hz<br>Magnitude: %{y:.2e} ' + unit + '<extra></extra>'
        }],
        'layout': {
            'title': title,
            'xaxis': {
                'title': {
                    'text': 'Frequency (Hz)',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'yaxis': {
                'title': {
                    'text': f'Envelope Magnitude ({unit})',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
        }
    }
    return fig

def squared_envelope_plot(signal, fs, title="Squared Envelope Spectrum", unit=""):
    f,X = envelope(signal, fs)
    fig = {
        'data': [{
            'x': f,
            'y': X,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Envelope Spectrum',
            'line': {'color': '#1f77b4'},  # Different color from regular spectrum
            'hovertemplate': 'Frequency: %{x:.2f} Hz<br>Magnitude: %{y:.2e} ' + unit + '<extra></extra>',
            'showlegend': False
        }],
        'layout': {
            'title': title,
            'xaxis': {
                'title': {
                    'text': 'Frequency (Hz)',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': False,
                'gridcolor': 'lightgray'
            },
            'yaxis': {
                'title': {
                    'text': f'Envelope Magnitude ({unit})',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
    }}
    return fig
    
def cepstrum_prewhitening_plot(signal, fs, title="Squared Envelope Spectrum", unit=""):
    _, time_signal = cepstrum_prewhitening(signal, fs)
    s_mid = time_signal[2000:-2000]
    f,X = envelope(s_mid, fs)
    fig = {
        'data': [{
            'x': f,
            'y': X,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Envelope Spectrum',
            'line': {'color': '#1f77b4'},  # Different color from regular spectrum
            'hovertemplate': 'Frequency: %{x:.2f} Hz<br>Magnitude: %{y:.2e} ' + unit + '<extra></extra>',
            'showlegend': False
        }],
        'layout': {
            'title': title,
            'xaxis': {
                'title': {
                    'text': 'Frequency (Hz)',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'yaxis': {
                'title': {
                    'text': f'Envelope Magnitude ({unit})',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
    }}
    return fig

def benchmark_plot(signal, fs, title="Squared Envelope Spectrum", unit=""):
    t, s, f, X = benchmark_method(signal, fs, N=8192, Delta=500)
    fig = {
        'data': [{
            'x': f,
            'y': X,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Envelope Spectrum',
            'line': {'color': '#1f77b4'},  # Different color from regular spectrum
            'hovertemplate': 'Frequency: %{x:.2f} Hz<br>Magnitude: %{y:.2e} ' + unit + '<extra></extra>',
            'showlegend': False
        }],
        'layout': {
            'title': title,
            'xaxis': {
                'title': {
                    'text': 'Frequency (Hz)',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'yaxis': {
                'title': {
                    'text': f'Envelope Magnitude ({unit})',
                    'font': {'size': 14, 'family': 'Arial, sans-serif'},
                    'standoff': 10
                },
                'type': 'log',
                'showgrid': True,
                'gridcolor': 'lightgray'
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
    }}
    return fig