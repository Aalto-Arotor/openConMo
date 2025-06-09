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


def chunks(lst, n):
    lst = list(lst)
    k = len(lst) // n
    for i in range(n):
        yield lst[i * k: (i + 1) * k] if i < n - 1 else lst[i * k:]


files = list(download_dir.glob("**/*.csv"))
file_chunks = list(chunks(files, 3))

for idx, chunk in enumerate(file_chunks):
    print(f"\nProcessing chunk {idx+1} of {len(file_chunks)}...")
    dfs = []

    for f in chunk:
        print(f"Processing file: {f}")
        p = f.parts

        # Parse measurement metadata from file path
        if p[-2] == "Healthy":
            rpm = int(p[-4].replace("RPM", ""))
            torque = torque_map.get(p[-3], "unknown")
            installation = 0
            severity = "-"
            GP = int(p[-1][2])
            fault = "healthy"
        elif p[-3] == "Faulty":
            print(p[-3])
            rpm = int(p[-5].replace("RPM", ""))
            torque = torque_map.get(p[-4], "unknown")
            installation = int(p[-2][-1])
            # Fault types we don't care about not in map
            severity, fault = fault_map.get(
                p[-1].replace(".csv", ""), (None, None))
            GP = 0
        else:
            print(f"Skipping unknown file structure: {f}")
            continue

        if fault is None:
            print(f"Skipping unknown fault type: {p[-1]}")
            continue

        try:
            df = pd.read_csv(f, sep=",", index_col=0, header=0)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

        # Preprocess signals
        if df["enc4_ang"].iloc[0] > df["enc4_ang"].iloc[1]:
            df["enc4_ang"] = -df["enc4_ang"]

        df["time"] = fix_time(df["time"].to_numpy())

        # Add labels
        df["class"] = fault
        df["rpm"] = rpm
        df["torque"] = torque
        df["installation"] = installation
        df["severity"] = severity
        df["healthy_GP"] = GP

        # Convert float64 to float32
        if conversion:
            float64_cols = list(df.select_dtypes(include="float64"))
            df[float64_cols] = df[float64_cols].astype("float32")

        dfs.append(df)

    if not dfs:
        print(f"No valid data found in chunk {idx+1}, skipping.")
        continue

    # Combine all dataframes for this chunk
    dfs = pd.concat(dfs, ignore_index=True)

    # Clean up object-type columns
    string_cols = dfs.select_dtypes(include="object").columns
    dfs[string_cols] = dfs[string_cols].apply(lambda x: x.str.strip())

    dfs = dfs.reset_index(drop=True)

    # Save to parquet
    out_path = download_dir.parent / f"AGFD_chunk_{idx+1}.parquet"
    dfs.to_parquet(out_path, index=False)
    print(f"Saved chunk {idx+1} to: {out_path}")


print("Saving done.")
print()

print("Dataframe shape:", dfs.shape)
print("Data types:")
print(dfs.dtypes)

print()
print("First 3 rows of the dataframe:")
print(dfs.head(3))
