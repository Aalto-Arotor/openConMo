

from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
import subprocess
import scipy.io as io
import os
import math
import sys
import threading


# Allun download


def download(url: str, filepath: str, chunk_size=1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)


try:
    # Works in .py script
    cur_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback for notebooks where __file__ is undefined
    cur_dir = Path(os.getcwd())
download_dir = cur_dir / "data" / "Paderborn" / "RAW"
download_dir.mkdir(parents=True, exist_ok=True)
conversion = True
if (download_dir / "Paderborn").is_dir():
    # All ok
    pass
elif (download_dir / "K001.rar").is_file():  # Note only checks for one file instead of all of them
    print(
        "Please extract all of the .rar file contents to the data/Paderborn/RAW directory and run the script again."
    )

else:
    print("Downloading the Paderborn dataset...")
    print()

    # Mendeley data download url
    # For manual download, visit: https://groups.uni-paderborn.de/kat/BearingDataCenter/

    files = [
        "K001.rar",
        "K002.rar",
        "K003.rar",
        "K004.rar",
        "K005.rar",
        "K006.rar",
        "KA01.rar",
        "KA03.rar",
        "KA04.rar",
        "KA05.rar",
        "KA06.rar",
        "KA07.rar",
        "KA08.rar",
        "KA09.rar",
        "KA15.rar",
        "KA16.rar",
        "KA22.rar",
        "KA30.rar",
        "KB23.rar",
        "KB24.rar",
        "KB27.rar",
        "KI01.rar",
        "KI03.rar",
        "KI04.rar",
        "KI05.rar",
        "KI07.rar",
        "KI08.rar",
        "KI14.rar",
        "KI16.rar",
        "KI17.rar",
        "KI18.rar",
        "KI21.rar"
    ]
    total = len(files)
    i = 0
    base_url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
    for file in files:
        file_url = base_url + file
        download(file_url, download_dir / file)
        print("Downloading file {} / {}".format(i+1, total))
        i += 1

    print()
    print("Download complete.")
    print(

        "Please extract all of the .rar file contents to the data/Paderborn/RAW directory and run the script again."
    )
dfs = []


files = download_dir.glob("**/*.mat")


dfs = []
for f in files:
    # SPECS
    def is_v73_mat(filepath):
        """Check if the .mat file is v7.3 (HDF5) format"""
        with open(filepath, 'rb') as f:
            header = f.read(128).decode(errors='ignore')
            return 'MATLAB 7.3' in header

    p = f.parts

    try:

        data = io.loadmat(f)
    except TypeError:
        print("This .mat file is weird: {}, Skipping".format(f))
        continue

    data_mes = data[p[-1].replace(".mat", "")]

    entry = data_mes[0, 0]
    signal_list = entry['Y'].squeeze()

    max_len = max(signal[2][0].shape[0] for signal in signal_list)

    # Initialize dict for DataFrame
    df_dict = {}

    for signal in signal_list:
        name = signal[0][0]
        values = signal[2][0]
        if values.ndim == 1:
            padded = np.pad(values, (0, max_len - len(values)),
                            constant_values=np.nan)
        else:
            # Handles cases where values are stored as (1, N) arrays
            values = values.flatten()
            padded = np.pad(values, (0, max_len - len(values)),
                            constant_values=np.nan)
        df_dict[name] = padded

    # Create DataFrame
    df = pd.DataFrame(df_dict)

    # Try to add time column from X (also padded if needed)
    x_signals = entry['X'].squeeze()
    for x_sig in x_signals:
        try:
            label = x_sig[4][0]
            if 'time' in label.lower() or 'host' in label.lower():
                time_values = x_sig[2][0]
                if len(time_values) < max_len:
                    time_values = np.pad(
                        time_values, (0, max_len - len(time_values)), constant_values=np.nan)
                df.insert(0, 'time', time_values)
                break
        except Exception:
            continue
    # Data has to be split between the bearing codes
    df["brg_code"] = p[-2]
    df["setting"] = p[-1].replace(".mat", "")
    dfs.append(df)
    # Final DataFrame is ready

print("Starting to concatenate...")


def save_dfs_to_parquet(dataframes):
    n = len(dfs)
    split_size = math.ceil(n / 4)

    for i in range(4):
        chunk = dfs[i * split_size: (i + 1) * split_size]
        print(f"Processing chunk {i+1}/4 with {len(chunk)} DataFrames...")
        df = pd.concat(chunk, ignore_index=True)
        print(df["setting"].unique())
        print(Path.cwd())
        df.to_parquet(
            Path.cwd() / "data" / "Paderborn" / "Paderborn_chunk_{}.parquet".format(i+1),
            index=False,

        )
        print("Chunk saved.")

    print("Saving done.")
    print()

    print("Dataframe shape:", df.shape)
    print("Data types:")
    print(df.dtypes)

    print()
    print("First 3 rows of the dataframe:")
    print(df.head(3))


save_dfs_to_parquet(dfs)
