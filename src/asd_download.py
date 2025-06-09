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

download_dir = cur_dir / "data" / "ASD" / "RAW"
download_dir.mkdir(parents=True, exist_ok=True)
# TOGGLE TRUE FOR 32 float conversion (True if data used for DL, false if 64 floats required)
conversion = True

if (download_dir / "ASD").is_dir():
    # All ok
    pass
elif (download_dir / "ASD.rar").is_file():
    print(
        "Please extract the ASD.rar file contents to the data/ASD/RAW directory and run the script again."
    )

else:
    print("Downloading ASD dataset...")
    print()

    # Mendeley data download url
    # For manual download, visit: https://data.mendeley.com/datasets/fsjhhrw2y8/1
    url = "https://data.mendeley.com/public-files/datasets/fsjhhrw2y8/files/82cee859-8e9f-4cb0-9337-850679ea0e86/file_downloaded"

    download(url, download_dir / "ASD.rar")

    print()
    print("Download complete.")
    print(
        # This needs to be changed, can rar files be extracted with python?
        "Please extract the ASD.rar file contents to the data/ASD/RAW directory and run the script again."
    )
    # info for .rar extraction on mac: https://discussions.apple.com/thread/255141368?sortBy=rank


# ASD dataset fault map
    # (number, thickness , class number)
fault_map = {
    "Healthy": ("0", "0", "0"),
    "Failure1": ("1", "0.01", "1"),
    "Failure2": ("2", "0.01", "2"),
    "Failure3": ("3", "0.01", "3"),
    "Failure4": ("1", "0.03", "4"),
    "Failure5": ("2", "0.03", "5"),
    "Failure6": ("3", "0.03", "6"),
    "Failure7": ("1", "0.05", "7"),
    "Failure8": ("2", "0.05", "8"),
    "Failure9": ("3", "0.05", "9")
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
    print(p)

    rpm = int(p[-2].replace("RPM", ""))
    number, thickness, label = fault_map.get(            # number = how many shims, thickness = the thickness of the shim
        p[-1].replace(".csv", ""), (None, None)
    )
    fault = "healthy"
    fault = "shim"

    # Skip unknown faults
    if fault is None:
        raise ValueError(f"Unknown fault type: {p[-1]}")

    # SIGNAL
    ##

    # Read CSV
    df = pd.read_csv(f, sep=",", index_col=0, header=0)

    print(str(f))

    # Add info from file name

    df["class"] = label
    df["rpm"] = rpm

    # Make DF of one file
    dfs.append(df)
    # print(len(dfs))

# dfs listassa kaikki järkevässä muodossa
# Combine files <--- Toimii!
dfs = pd.concat(dfs)

# * Conversion done because deep learning computations are done with float32 anyway <--- kaikki ei tee välttämättä DL
# Get float 64 columns
if conversion:

    float64_cols = list(dfs.select_dtypes(include="float64"))
    # Convert those columns float 32 pitäiskö tehdä?
    dfs[float64_cols] = dfs[float64_cols].astype("float32")

string_cols = [
    "class",
    "rpm",
]
dfs[string_cols] = dfs[string_cols].astype("string")

# Reset index to counteract concatenating a bunch of separate dataframes
dfs = dfs.reset_index(drop=True)

print("Converssion to parquet done.")
print("Saving to parquet...")

dfs.to_parquet(
    download_dir.parent / "ASD_downloaded.parquet",
    index=False,
)

print("Saving done.")
print()

print("Dataframe shape:", dfs.shape)
print("Data types:")
print(dfs.dtypes)

print()
print("First 3 rows of the dataframe:")
print(dfs.head(3))
