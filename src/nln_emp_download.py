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


def download_nln(cur_dir):
    download_dir = cur_dir / "data" / "NLN-EMP" / "RAW"
    print(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    conversion = True  # True if float32 needed (e.g. for deep learning)

    if (download_dir / "Dataset").is_dir():
        pass
    elif list(download_dir.glob("*.7z")):
        print("Please extract the .7z file contents to data/NLN-EMP/RAW and rerun.")

    else:
        print("Downloading NLN-EMP dataset...\n")
        url = "https://data.4tu.nl/file/2b61183e-c14f-4131-829b-cc4822c369d0/8d84b13f-98f7-4baf-a7ea-60dee1e8876f"
        download(url, download_dir /
                 "Current and Vibration Monitoring Dataset for various Faults in an E-motor-driven Centrifugal Pump.7z")
        print("\nDownload complete.\nPlease extract all contents and rerun.")
        return

    def fix_time(X):
        increment = 1 / 3012
        loc = np.where(X[1:] < X[:-1])[0] + 1
        if len(loc) == 0:
            return X
        for l in loc:
            wrong = X[l - 1] - X[l]
            diff = np.concatenate(
                [np.zeros(l), np.ones(len(X) - l) * (increment + wrong)])
            X = X + diff
        return X

    motor_map = {"Motor-2": 2, "Motor-4": 4}

    def chunks(lst, n):
        lst = list(lst)
        k = len(lst) // n
        for i in range(n):
            yield lst[i*k: (i+1)*k] if i < n - 1 else lst[i*k:]

    files = list(download_dir.glob("**/*.csv"))
    file_chunks = list(chunks(files, 3))

    for idx, chunk in enumerate(file_chunks):
        print(f"Processing chunk {idx+1} of {len(file_chunks)}...")
        print(idx)
        dfs = []
        i = 0
        for f in chunk:
            print(f)
            p = f.parts

            speed = int(p[-3])
            channel = int(p[-1].split("-")
                          [-1].replace(".csv", "").replace("ch", ""))
            try:
                severity = int(p[-2].split("-")[-1][-1])
            except ValueError:
                severity = None
            fault = str(p[-2].split("-")[-1].replace(str(severity), ""))
            n_poles = motor_map[p[-4]]
            m_type = p[-5]

            df = pd.read_csv(f, sep=",", header=0)
            df["class"] = fault
            df["speed"] = speed
            df["channel"] = channel
            df["severity"] = severity
            df["n_poles"] = n_poles
            df["m_type"] = m_type

            float64_cols = list(df.select_dtypes(include="float64"))
            df[float64_cols] = df[float64_cols].astype("float32")

            dfs.append(df)

        if not dfs:
            print(f"No valid files in chunk {idx+1}, skipping...")
            continue

        dfs = pd.concat(dfs)
        print(dfs)

        if conversion:
            float64_cols = list(dfs.select_dtypes(include="float64"))
            dfs[float64_cols] = dfs[float64_cols].astype("float32")

        string_cols = df.select_dtypes(include="object").columns
        df[string_cols] = df[string_cols].apply(lambda x: x.str.strip())
        dfs = dfs.reset_index(drop=True)

        dfs.to_parquet(
            Path.cwd() / "data" / "NLN-EMP" / "NLN_EMP_chunk_{}.parquet".format(idx+1),
            index=False,
        )
        print("Chunk saved.")
        i += 1

    print("All chunks processed and saved.")
    print("Dataframe shape:", dfs.shape)
    print("Data types:")
    print(dfs.dtypes)

    print()
    print("First 3 rows of the dataframe:")
    print(dfs.head(3))


try:
    # Works in .py script
    cur_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback for notebooks where __file__ is undefined
    cur_dir = Path(os.getcwd())
dfs = download_nln(cur_dir)
