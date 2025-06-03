from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd

import os


def download(url: str, filepath: str, chunk_size=1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)


# FILE CHECK / DOWNLOAD
#######################

# cur_dir = Path(__file__).parent.parent # IF from .py file
try:
    # Works in .py script
    cur_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback for notebooks where __file__ is undefined
    cur_dir = Path(os.getcwd())

download_dir = cur_dir / "data" / "AGFD" / "RAW"
download_dir.mkdir(parents=True, exist_ok=True)
# TOGGLE TRUE FOR 32 float conversion (True if data used for DL, false if 64 floats required)
conversion = True

if (download_dir / "3012Hz").is_dir():
    # All ok
    pass
elif (download_dir / "AGFD.rar").is_file():
    print(
        "Please extract the AGFD.rar file contents to the data/AGFD/RAW directory and run the script again."
    )
else:
    print("Downloading AGFD dataset...")
    print()

    # Mendeley data download url
    # For manual download, visit: https://data.mendeley.com/datasets/fywnj597d8/2
    url = "https://data.mendeley.com/public-files/datasets/fywnj597d8/files/1888dd1e-4ceb-401b-800a-fb49443a1cbc/file_downloaded"

    download(url, download_dir / "AGFD.rar")

    print()
    print("Download complete.")
    print(
        "Please extract the AGFD.rar file contents to the data/AGFD/RAW directory and run the script again."
    )

# CLEANING
##########

# HELPERS


def fix_time(X):
    increment = 1 / 3012  # For 3012 Hz sampling rate
    loc = np.where(X[1:] < X[:-1])[0] + 1  # + 1 needed to get the right index

    if len(loc) == 0:
        return X

    for l in loc:
        wrong = X[l - 1] - X[l]
        print(f"Wrong time at {l}")
        diff = np.concatenate(
            [
                np.zeros(l),
                np.ones(len(X) - l) * (increment + wrong),
            ]
        )
        X = X + diff

    return X


# PROCESSING

torque_map = {
    "0_12Nm": 1,
    "0_71Nm": 6,
    "1_31Nm": 11,
}

fault_map = {
    "Mild_Pitting_GP1": ("mild", "pitting"),
    "Severe_Pitting_GP6": ("severe", "pitting"),
    "Mild_Wear_GP7": ("mild", "wear"),
    "Severe_Wear_GP2": ("severe", "wear"),
    "Mild_Micropitting_GP4": ("mild", "micropitting"),
    "Severe_Micropitting_GP3": ("severe", "micropitting"),
    "Mild_TFF_GP9": ("mild", "crack"),
    "Severe_TFF_GP5": ("severe", "crack"),
}

# Specify raw data
files = download_dir.glob("**/*.csv")

# Go through files
dfs = []
for f in files:
    # SPECS
    ##

    # Get measurement specifications from file path
    p = f.parts

    if p[-2] == "Healthy":
        rpm = int(p[-4].replace("RPM", ""))
        torque = torque_map[p[-3]]
        installation = 0
        severity = "-"
        GP = int(p[-1][2])
        fault = "healthy"
    elif p[-3] == "Faulty":
        rpm = int(p[-5].replace("RPM", ""))
        torque = torque_map[p[-4]]
        installation = int(p[-2][-1])
        severity, fault = fault_map.get(
            p[-1].replace(".csv", ""), (None, None)
        )  # Fault types we don't care about not in map
        GP = 0
    else:
        raise ValueError(f"Unknown type directory")

    # Skip unknown faults
    if fault is None:
        raise ValueError(f"Unknown fault type: {p[-1]}")

    # SIGNAL
    ##

    # Read CSV
    df = pd.read_csv(f, sep=",", index_col=0, header=0)

    print(str(f))

    # Add info from file name
    df["class"] = fault
    df["rpm"] = rpm
    df["torque"] = torque
    df["installation"] = installation
    df["severity"] = severity
    df["healthy_GP"] = GP

    # Fix signal data
    #################

    # Enc4 runs the wrong way in most files
    if df["enc4_ang"].iloc[0] > df["enc4_ang"].iloc[1]:
        df["enc4_ang"] = -df["enc4_ang"]

    # Some measurements have glitches in time
    df["time"] = fix_time(df["time"].to_numpy())

    # Make DF of one file
    dfs.append(df)


# Combine files
dfs = pd.concat(dfs)

# * Conversion done because deep learning computations are done with float32 anyway
# Get float 64 columns
if conversion:

    float64_cols = list(dfs.select_dtypes(include="float64"))
    # Convert those columns float 32 pitäiskö tehdä?
    dfs[float64_cols] = dfs[float64_cols].astype("float32")

string_cols = [
    "class",
    "severity",
]
dfs[string_cols] = dfs[string_cols].astype("string")

# Reset index to counteract concatenating a bunch of separate dataframes
dfs = dfs.reset_index(drop=True)

print("Converssion to feather done.")
print("Saving to feather...")

dfs.to_feather(download_dir.parent / "AGFD_downloaded.feather")

print("Saving done.")
print()

print("Dataframe shape:", dfs.shape)
print("Data types:")
print(dfs.dtypes)

print()
print("First 3 rows of the dataframe:")
print(dfs.head(3))
