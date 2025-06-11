import pandas as pd
import base64
import io
import pandas as pd
import numpy as np
from datetime import datetime

def read_from_parquet(contents):
    """
    Read measurement data from a parquet file uploaded through Dash.
    
    Parameters:
    -----------
    contents : str
        Base64 encoded string from Dash Upload component
    
    Returns:
    --------
    tuple
        (signal, fs, name, measurement_location, unit, fault_frequencies)
    """
    # Decode the base64 string
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Read parquet from bytes
    df = pd.read_parquet(io.BytesIO(decoded))
    
    # Extract signal and time data
    signal = df['signal'].values
    
    # Extract metadata (taking first value since these are broadcasted)
    fs = df['sampling_frequency'].iloc[0]
    name = df['name'].iloc[0]
    measurement_location = df['measurement_location'].iloc[0]
    unit = df['unit'].iloc[0]
    meas_id = df['meas_id'].iloc[0]
    fault = df['fault'].iloc[0]
    rotating_freq_hz = df['rotating_freq_hz'].iloc[0]
    
    # Reconstruct fault frequencies dictionary
    fault_frequencies = {}
    fault_freq_columns = [col for col in df.columns if col.startswith('fault_freq_')]
    
    if fault_freq_columns:
        for col in fault_freq_columns:
            _, location, fault_type = col.split('_', 2)
            if location not in fault_frequencies:
                fault_frequencies[location] = {}
            fault_frequencies[location][fault_type] = df[col].iloc[0]
    else:
        fault_frequencies = None
    
    return signal, fs, name, measurement_location, unit, meas_id, fault, fault_frequencies, rotating_freq_hz