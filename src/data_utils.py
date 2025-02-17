import glob
import os

import numpy as np
import pandas as pd
import scipy

import utils

cwru_fault_frequencies = {"DE": {"BPFI": 5.415,
                                  "BPFO": 3.585,
                                  "BSF": 2.357,
                                  "FTF": 0.3983},
                          "FE": {"BPFI": 4.947,
                                "BPFO": 3.053,
                                 "BSF": 1.994,
                                 "FTF": 0.3816}}
def save_to_parquet(signal, fs, name, measurement_location, unit, meas_id, fault, fault_frequencies, rotating_freq_hz):
    """
    Save measurement data to a parquet file.
    """
    
    # Create DataFrame with both data and metadata
    df = pd.DataFrame({
        'signal': signal,
        'unit': unit,
        # Store metadata as columns with single values
        'sampling_frequency': fs,
        'name': name,
        'fault': fault,
        'measurement_location': measurement_location,
        'rotating_freq_hz': rotating_freq_hz,
        'meas_id': meas_id
    })
    
    # Store fault frequencies as additional columns if provided
    if fault_frequencies is not None:
        for location, freqs in fault_frequencies.items():
            for fault_type, value in freqs.items():
                # Broadcast the value to match the length of the DataFrame
                df[f'fault_freq_{location}_{fault_type}'] = value
    
    filename = f"measurements/{name}_{meas_id}_{measurement_location}.parquet"
    # Save to parquet
    df.to_parquet(filename)
    
    return filename

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

def get_bearing_frequencies(end="DE"):
    '''
    Input:
         end: "DE" drive end "FE" fan end bearing.
    '''
    # Position on rig Model number Fault frequencies (multiple of shaft speed)
    #                             BPFI  BPFO  FTF    BSF
    # Drive end SKF 6205-2RS JEMa 5.415 3.585 0.3983 2.357
    # Fan end SKF 6203-2RS JEM    4.947 3.053 0.3816 1.994

    if end == "DE":
        return 5.415, 3.585, 0.3983, 2.357

    elif end == "FE":
        return 4.947, 3.053, 0.3816, 1.994

    else:
        return None


def load_data(fault_class=0, printall=False):
    fault_classes = ["12k_Drive_End_Bearing_Fault_Data",
                     "12k_Fan_End_Bearing_Fault_Data",
                     "48k_Drive_End_Bearing_Fault_Data",
                     "Normal"]
    root_dir = fault_classes[fault_class]
    mat_files = glob.glob(f"CWRU-dataset/{root_dir}/**/*.mat", recursive=True)

    if printall:
        for idx, file in enumerate(mat_files):
            print(f"{idx}: {file}")

    return mat_files


def get_timeseries_data(data, normal=True):
    keys = data.keys()
    DE_key = next((key for key in keys if "DE" in key), None)
    FE_key = next((key for key in keys if "FE" in key), None)
    BA_key = next((key for key in keys if "BA" in key), None)
    rpm_key = next((key for key in keys if "rpm" in key), None)

    acc_DE = data[DE_key] if DE_key else None
    acc_FE = data[FE_key] if FE_key else None
    acc_BA = data[BA_key] if BA_key else None
    rpm = data[BA_key] if rpm_key else None

    return acc_DE, acc_FE, acc_BA, rpm


def extract_signals(filename, normal=False):
    num_samples = -1
    data = scipy.io.loadmat(filename)
    acc_DE, acc_FE, acc_BA, rpm = get_timeseries_data(data, normal)

    # Flatten and slice only if the data is not None
    acc_DE = acc_DE[:num_samples].flatten() if acc_DE is not None else None
    acc_FE = acc_FE[:num_samples].flatten() if acc_FE is not None else None
    acc_BA = acc_BA[:num_samples].flatten() if acc_BA is not None else None

    return acc_DE, acc_FE, acc_BA, rpm


def load_allu():
    mat_files = glob.glob("pu-dataset/*05*.mat", recursive=True)
    return mat_files


def extract_allu_signals(filename):
    print(filename)
    data = scipy.io.loadmat(filename)
    data = data[filename.split(".")[0].split("/")[-1]][0][0][2][0][6][2][0]
    # print(data.shape)
    return data


def get_CWRU_data():
    """Load the CWRU dataset as a Pandas DataFrame.

    Returns:
        cwru_data: pd.DataFrame - Columns
            'measurement location': str - 'DE' -> Drive End, 'FE' -> Fan End
            'fault location': str - 'DE' -> Drive End, 'FE' -> Fan End
            'fault type': str - 'ir' -> Inner Ring, 'or' -> Outer Ring, 'b' -> ball
            'fault diameter': int - in mils, e.g. 7 = 0.007
            'fault orientation': str - '-' -> N/A, 'c' -> center/12 o'clock, 'or' -> orthogonal, 'op' -> opposite
            'sampling rate': int - in kHz
            'motor load': int - in HP
            'tags': list[str]
    """
    return pd.read_feather(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "CWRU_openCoMo.feather"))
    )


def get_CWRU_measurement(
    measurement_location: str,
    fault_location: str,
    fault_type: str,
    fault_diameter: int,
    fault_orientation: str,
    sampling_rate: int,
    motor_load: int,
    CWRU_data: pd.DataFrame = None,
    downsample: bool = False,
):
    """
    Get the specified CWRU measurement. Raises an exception if the measurement
    does not exist. Use '-' when a specification is not applicable, e.g. 'fault
    location' for healthy measurements.

    Args:
        ...
        CWRU_data: Can optionally be passed a preloaded CWRU dataframe.
    Recommended if getting multiple measurements in succession.
        downsample: If True, 48 kHz measurements are downsampled to 12 kHz

    Returns:
        measurement: numpy array
        rpm: int - measurement rpm
        tags: list[str] - possible problems in measurement as identified by Smith & Randall
    """
    # Motor load (HP) to RPM
    rpm_map = {
        0: 1797,
        1: 1772,
        2: 1750,
        3: 1730,
    }

    # Ensure CWRU data is loaded
    if CWRU_data is None:
        CWRU_data = get_CWRU_data()

    # Get correct measurement
    row = CWRU_data[
        (CWRU_data["measurement location"] == measurement_location)
        & (CWRU_data["fault location"] == fault_location)
        & (CWRU_data["fault type"] == fault_type)
        & (CWRU_data["fault diameter"] == fault_diameter)
        & (CWRU_data["fault orientation"] == fault_orientation)
        & (CWRU_data["sampling rate"] == sampling_rate)
        & (CWRU_data["motor load"] == motor_load)
    ]

    # Check that measurement exists
    if len(row) == 0:
        raise Exception(
            f"No such measurement with specifications `{sampling_rate}kHz, {motor_load}HP, {measurement_location}, {fault_location}, {fault_type}, {fault_diameter} mil, {fault_orientation}`!"
        )

    rpm = rpm_map[row["motor load"].iloc[0]]
    tags = list(row["tags"].iloc[0])
    measurement = row["measurement"].iloc[0]

    # Downsample measurement from 48kHz to 12kHz if specified and necessary
    if downsample and sampling_rate == 48:
        measurement = utils.downsample(measurement, 48000, 12000)

    return measurement, rpm, tags


# PU bearing codes and faults for convenience
##

PU_bearing_map = {
    # Healthy
    "K001": "Normal",
    "K002": "Normal",
    "K003": "Normal",
    "K004": "Normal",
    "K005": "Normal",
    "K006": "Normal",
    # Artificial
    "KA01": "or",
    "KA03": "or",
    "KA05": "or",
    "KA06": "or",
    "KA07": "or",
    "KA08": "or",
    "KA09": "or",
    "KI01": "ir",
    "KI03": "ir",
    "KI05": "ir",
    "KI07": "ir",
    "KI08": "ir",
    # Real
    "KA04": "or",
    "KA15": "or",
    "KA16": "or",
    "KA22": "or",
    "KA30": "or",
    "KB23": "ir+or",  # ir dominant
    "KB24": "ir+or",  # ir dominant
    "KB27": "ir+or",
    "KI04": "ir",
    "KI14": "ir",
    "KI16": "ir",
    "KI17": "ir",
    "KI18": "ir",
    "KI21": "ir",
}


def get_PU_data():
    """Load the Paderborn University Bearing Fault dataset accelerometer measurements as a Pandas DataFrame. Measurements values are stored as float32 and the sampling rate is 64 kHz. Dataset publication @ papers.phmsociety.org/index.php/phme/article/download/1577/542

    Returns:
        pu_data: pd.DataFrame - Columns
            'bearing': str - Bearing code used in publication
            'sample num': int - Sample number. Each bearing was measured 20 times for 4 s. Numbers start from 1.
            'fault type': str - Bearing fault type. 'ir' -> Inner Ring, 'or' -> Outer Ring, 'ir+or' -> Inner & Outer Ring faults present
            'rpm': int - Motor (and bearing) RPM
            'motor load': int - Load torque in Nm x 10, i.e. 7 -> 0.7 Nm
            'radial force': int - Radial force in N
    """
    return pd.read_feather(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "PU_openCoMo.feather"))
    )


def get_PU_measurement(
    bearing: str,
    sample_num: str,
    rpm: int,
    motor_load: int,
    radial_force: int,
    PU_data: pd.DataFrame = None,
):
    """
    Get the specified PU measurement. Raises an exception if the measurement
    does not exist.

    Args:
        ...
        PU_data: Can optionally be passed a preloaded CWRU dataframe.
    Recommended if getting multiple measurements in succession.

    Returns:
        measurement: numpy array
        fault_type: str
    """
    # Ensure PU data is loaded
    if PU_data is None:
        PU_data = get_PU_data()

    # Get correct measurement
    row = PU_data[
        (PU_data["bearing"] == bearing)
        & (PU_data["sample num"] == sample_num)
        & (PU_data["rpm"] == rpm)
        & (PU_data["motor load"] == motor_load)
        & (PU_data["radial force"] == radial_force)
    ]

    # Check that measurement exists
    if len(row) == 0:
        raise Exception(
            f"No such measurement with specifications `{bearing}, #{sample_num}, {rpm} RPM, {motor_load / 10} Nm, {radial_force} N`!"
        )

    measurement = row["measurement"].iloc[0]
    fault_type = PU_bearing_map[bearing]

    return measurement, fault_type
