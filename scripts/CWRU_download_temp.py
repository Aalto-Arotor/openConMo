'''
This is a temporary script to download the CWRU dataset from the Bearing Data Center.
It downloads the files, processes them, and saves them in a single Feather file.
'''


from urllib.request import urlretrieve
import requests

# import os
import scipy
from pathlib import Path
import numpy as np
import pandas as pd

###
# FILE SPECIFICATION
###

files = [
    # Format: (file_id, fault_type, fault_code, rpm, sensor_location, load_condition, index)

    # Normal baseline data: https://engineering.case.edu/bearingdatacenter/normal-baseline-data

    (97,  "normal", "-", "12k", "Drive_End", None, 0),
    (98,  "normal", "-", "12k", "Drive_End", None, 1),
    (99,  "normal", "-", "12k", "Drive_End", None, 2),
    (100, "normal", "-", "12k", "Drive_End", None, 3),

    ##
    # 12k Drive End Bearing Fault Data: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
    ##

    # IR 007
    (105, "IR", "007", "12k", "Drive_End", None, 0),
    (106, "IR", "007", "12k", "Drive_End", None, 1),
    (107, "IR", "007", "12k", "Drive_End", None, 2),
    (108, "IR", "007", "12k", "Drive_End", None, 3),

    # IR 014
    (169, "IR", "014", "12k", "Drive_End", None, 0),
    (170, "IR", "014", "12k", "Drive_End", None, 1),
    (171, "IR", "014", "12k", "Drive_End", None, 2),
    (172, "IR", "014", "12k", "Drive_End", None, 3),

    # IR 021
    (209, "IR", "021", "12k", "Drive_End", None, 0),
    (210, "IR", "021", "12k", "Drive_End", None, 1),
    (211, "IR", "021", "12k", "Drive_End", None, 2),
    (212, "IR", "021", "12k", "Drive_End", None, 3),

    # IR 028
    (3001, "IR", "028", "12k", "Drive_End", None, 0),
    (3002, "IR", "028", "12k", "Drive_End", None, 1),
    (3003, "IR", "028", "12k", "Drive_End", None, 2),
    (3004, "IR", "028", "12k", "Drive_End", None, 3),

    # B 007
    (118, "B", "007", "12k", "Drive_End", None, 0),
    (119, "B", "007", "12k", "Drive_End", None, 1),
    (120, "B", "007", "12k", "Drive_End", None, 2),
    (121, "B", "007", "12k", "Drive_End", None, 3),

    # B 014
    (185, "B", "014", "12k", "Drive_End", None, 0),
    (186, "B", "014", "12k", "Drive_End", None, 1),
    (187, "B", "014", "12k", "Drive_End", None, 2),
    (188, "B", "014", "12k", "Drive_End", None, 3),

    # B 021
    (222, "B", "021", "12k", "Drive_End", None, 0),
    (223, "B", "021", "12k", "Drive_End", None, 1),
    (224, "B", "021", "12k", "Drive_End", None, 2),
    (225, "B", "021", "12k", "Drive_End", None, 3),

    # B 028
    (3005, "B", "028", "12k", "Drive_End", None, 0),
    (3006, "B", "028", "12k", "Drive_End", None, 1),
    (3007, "B", "028", "12k", "Drive_End", None, 2),
    (3008, "B", "028", "12k", "Drive_End", None, 3),

    # OR 007 - Centered Load (-> @6)
    (130, "OR", "007", "12k", "Drive_End", "@6", 0),
    (131, "OR", "007", "12k", "Drive_End", "@6", 1),
    (132, "OR", "007", "12k", "Drive_End", "@6", 2),
    (133, "OR", "007", "12k", "Drive_End", "@6", 3),

    # OR 007 - Orthogonal load (-> @3)
    (144, "OR", "007", "12k", "Drive_End", "@3", 0),
    (145, "OR", "007", "12k", "Drive_End", "@3", 1),
    (146, "OR", "007", "12k", "Drive_End", "@3", 2),
    (147, "OR", "007", "12k", "Drive_End", "@3", 3),

    # OR 007 - Opposite load (-> @12)
    (156, "OR", "007", "12k", "Drive_End", "@12", 0),
    (158, "OR", "007", "12k", "Drive_End", "@12", 1),
    (159, "OR", "007", "12k", "Drive_End", "@12", 2),
    (160, "OR", "007", "12k", "Drive_End", "@12", 3),

    # OR 014 - Centered load (-> @6)
    (197, "OR", "014", "12k", "Drive_End", "@6", 0),
    (198, "OR", "014", "12k", "Drive_End", "@6", 1),
    (199, "OR", "014", "12k", "Drive_End", "@6", 2),
    (200, "OR", "014", "12k", "Drive_End", "@6", 3),

    # OR 021 - Centered load (-> @6)
    (234, "OR", "021", "12k", "Drive_End", "@6", 0),
    (235, "OR", "021", "12k", "Drive_End", "@6", 1),
    (236, "OR", "021", "12k", "Drive_End", "@6", 2),
    (237, "OR", "021", "12k", "Drive_End", "@6", 3),

    # OR 021 - Orthogonal load (-> @3)
    (246, "OR", "021", "12k", "Drive_End", "@3", 0),
    (247, "OR", "021", "12k", "Drive_End", "@3", 1),
    (248, "OR", "021", "12k", "Drive_End", "@3", 2),
    (249, "OR", "021", "12k", "Drive_End", "@3", 3),

    # OR 021 - Opposite load (-> @12)
    (258, "OR", "021", "12k", "Drive_End", "@12", 0),
    (259, "OR", "021", "12k", "Drive_End", "@12", 1),
    (260, "OR", "021", "12k", "Drive_End", "@12", 2),
    (261, "OR", "021", "12k", "Drive_End", "@12", 3),

    ##
    # 48k Drive End Bearing Fault Data: https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data
    ##

    # IR 007
    (109, "IR", "007", "48k", "Drive_End", None, 0),
    (110, "IR", "007", "48k", "Drive_End", None, 1),
    (111, "IR", "007", "48k", "Drive_End", None, 2),
    (112, "IR", "007", "48k", "Drive_End", None, 3),  # electric_noise note if needed

    # IR 014
    (174, "IR", "014", "48k", "Drive_End", None, 0),
    (175, "IR", "014", "48k", "Drive_End", None, 1),
    (176, "IR", "014", "48k", "Drive_End", None, 2),
    (177, "IR", "014", "48k", "Drive_End", None, 3),

    # IR 021
    (213, "IR", "021", "48k", "Drive_End", None, 0),  # identical_DE_and_FE
    (214, "IR", "021", "48k", "Drive_End", None, 1),  # clipped
    (215, "IR", "021", "48k", "Drive_End", None, 2),  # clipped
    (217, "IR", "021", "48k", "Drive_End", None, 3),

    # B 007
    (122, "B", "007", "48k", "Drive_End", None, 0),
    (123, "B", "007", "48k", "Drive_End", None, 1),
    (124, "B", "007", "48k", "Drive_End", None, 2),
    (125, "B", "007", "48k", "Drive_End", None, 3),

    # B 014
    (189, "B", "014", "48k", "Drive_End", None, 0),  # identical_DE_and_FE
    (190, "B", "014", "48k", "Drive_End", None, 1),
    (191, "B", "014", "48k", "Drive_End", None, 2),  # clipped
    (192, "B", "014", "48k", "Drive_End", None, 3),

    # B 021
    (226, "B", "021", "48k", "Drive_End", None, 0),  # identical_DE_and_FE
    (227, "B", "021", "48k", "Drive_End", None, 1),
    (228, "B", "021", "48k", "Drive_End", None, 2),  # clipped
    (229, "B", "021", "48k", "Drive_End", None, 3),  # clipped

    # OR 007 – Centered load (→ @6)
    (135, "OR", "007", "48k", "Drive_End", "@6", 0),
    (136, "OR", "007", "48k", "Drive_End", "@6", 1),
    (137, "OR", "007", "48k", "Drive_End", "@6", 2),
    (138, "OR", "007", "48k", "Drive_End", "@6", 3),

    # OR 007 – Orthogonal load (→ @3)
    (148, "OR", "007", "48k", "Drive_End", "@3", 0),
    (149, "OR", "007", "48k", "Drive_End", "@3", 1),
    (150, "OR", "007", "48k", "Drive_End", "@3", 2),
    (151, "OR", "007", "48k", "Drive_End", "@3", 3),

    # OR 007 – Opposite load (→ @12)
    (161, "OR", "007", "48k", "Drive_End", "@12", 0),
    (162, "OR", "007", "48k", "Drive_End", "@12", 1),
    (163, "OR", "007", "48k", "Drive_End", "@12", 2),
    (164, "OR", "007", "48k", "Drive_End", "@12", 3),

    # OR 014 – Centered load (→ @6)
    (201, "OR", "014", "48k", "Drive_End", "@6", 0),  # identical_DE_and_FE
    (202, "OR", "014", "48k", "Drive_End", "@6", 1),
    (203, "OR", "014", "48k", "Drive_End", "@6", 2),
    (204, "OR", "014", "48k", "Drive_End", "@6", 3),

    # OR 021 – Centered load (→ @6)
    (238, "OR", "021", "48k", "Drive_End", "@6", 0),  # identical_DE_and_FE
    (239, "OR", "021", "48k", "Drive_End", "@6", 1),
    (240, "OR", "021", "48k", "Drive_End", "@6", 2),  # clipped
    (241, "OR", "021", "48k", "Drive_End", "@6", 3),  # clipped

    # OR 021 – Orthogonal load (→ @3)
    (250, "OR", "021", "48k", "Drive_End", "@3", 0),
    (251, "OR", "021", "48k", "Drive_End", "@3", 1),
    (252, "OR", "021", "48k", "Drive_End", "@3", 2),
    (253, "OR", "021", "48k", "Drive_End", "@3", 3),

    # OR 021 – Opposite load (→ @12)
    (262, "OR", "021", "48k", "Drive_End", "@12", 0),
    (263, "OR", "021", "48k", "Drive_End", "@12", 1),
    (264, "OR", "021", "48k", "Drive_End", "@12", 2),
    (265, "OR", "021", "48k", "Drive_End", "@12", 3),

    ##
    # 12k Fan End Bearing Fault Data: https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data
    ##

    # IR 007
    (278, "IR", "007", "12k", "Fan_End", None, 0),
    (279, "IR", "007", "12k", "Fan_End", None, 1),
    (280, "IR", "007", "12k", "Fan_End", None, 2),
    (281, "IR", "007", "12k", "Fan_End", None, 3),

    # IR 014
    (274, "IR", "014", "12k", "Fan_End", None, 0),
    (275, "IR", "014", "12k", "Fan_End", None, 1),
    (276, "IR", "014", "12k", "Fan_End", None, 2),
    (277, "IR", "014", "12k", "Fan_End", None, 3),

    # IR 021
    (270, "IR", "021", "12k", "Fan_End", None, 0),
    (271, "IR", "021", "12k", "Fan_End", None, 1),
    (272, "IR", "021", "12k", "Fan_End", None, 2),
    (273, "IR", "021", "12k", "Fan_End", None, 3),


    # B 007
    (282, "B", "007", "12k", "Fan_End", None, 0),
    (283, "B", "007", "12k", "Fan_End", None, 1),  # electric_noise
    (284, "B", "007", "12k", "Fan_End", None, 2),
    (285, "B", "007", "12k", "Fan_End", None, 3),

    # B 014
    (286, "B", "014", "12k", "Fan_End", None, 0),
    (287, "B", "014", "12k", "Fan_End", None, 1),
    (288, "B", "014", "12k", "Fan_End", None, 2),
    (289, "B", "014", "12k", "Fan_End", None, 3),

    # B 021
    (290, "B", "021", "12k", "Fan_End", None, 0),
    (291, "B", "021", "12k", "Fan_End", None, 1),
    (292, "B", "021", "12k", "Fan_End", None, 2),
    (293, "B", "021", "12k", "Fan_End", None, 3),

    # OR 007 – Centered load (→ @6)
    (294, "OR", "007", "12k", "Fan_End", "@6", 0),
    (295, "OR", "007", "12k", "Fan_End", "@6", 1),
    (296, "OR", "007", "12k", "Fan_End", "@6", 2),
    (297, "OR", "007", "12k", "Fan_End", "@6", 3),

    # OR 007 – Orthogonal load (→ @3)
    (298, "OR", "007", "12k", "Fan_End", "@3", 0),
    (299, "OR", "007", "12k", "Fan_End", "@3", 1),
    (300, "OR", "007", "12k", "Fan_End", "@3", 2),
    (301, "OR", "007", "12k", "Fan_End", "@3", 3),

    # OR 007 – Opposite load (→ @12)
    (302, "OR", "007", "12k", "Fan_End", "@12", 0),
    (305, "OR", "007", "12k", "Fan_End", "@12", 1),  # jump: 303 & 304 skipped
    (306, "OR", "007", "12k", "Fan_End", "@12", 2),
    (307, "OR", "007", "12k", "Fan_End", "@12", 3),

    # OR 014 – Centered load (→ @6)
    (313, "OR", "014", "12k", "Fan_End", "@6", 0),

    # OR 014 – Orthogonal load (→ @3) — Note: flipped order on website
    (310, "OR", "014", "12k", "Fan_End", "@3", 0),
    (309, "OR", "014", "12k", "Fan_End", "@3", 1),
    (311, "OR", "014", "12k", "Fan_End", "@3", 2),
    (312, "OR", "014", "12k", "Fan_End", "@3", 3),

    # OR 021 – Centered load (→ @6)
    (315, "OR", "021", "12k", "Fan_End", "@6", 0),

    # OR 021 – Orthogonal load (→ @3) — starts from index 1 (index 0 missing)
    (316, "OR", "021", "12k", "Fan_End", "@3", 1),
    (317, "OR", "021", "12k", "Fan_End", "@3", 2),
    (318, "OR", "021", "12k", "Fan_End", "@3", 3),
    ]

###
# DOWNLOADING
###

# URL base for .mat files
url_base = "https://engineering.case.edu/sites/default/files/{}.mat"

# Resolve current directory for notebooks or scripts
try:
    cur_dir = Path(__file__).parent
except NameError:
    cur_dir = Path().absolute()

# Base folder to store all data
download_base = cur_dir.parent / "examples" / "data" / "CWRU-dataset"
download_base.mkdir(parents=True, exist_ok=True)

failed_downloads = []

# Download loop
print(f"Downloading file 0/{len(files)}", end="\r")
for i, (file_id, fault_type, fault_code, rpm, sensor_location, load, index) in enumerate(files):
    print(f"Downloading file {i + 1}/{len(files)}", end="\r")

    # Special case for normal (healthy) data
    if fault_type == "normal":
        folder = download_base / "Normal"
    else:
        folder = download_base / f"{rpm}_{sensor_location}_Bearing_Fault_Data" / fault_type / fault_code
        if fault_type == "OR" and load is not None:
            folder = folder / load

    folder.mkdir(parents=True, exist_ok=True)

    # Save file using original file_id as filename
    f = folder / f"{file_id}.mat"

    # Skip if already downloaded
    if f.exists():
        continue

    download_tries = 5  # Max number of retry attempts for a file
    while download_tries > 0:
        try:
            # Construct the URL for the .mat file
            url = url_base.format(file_id)
            print(f"Downloading: {url}")

            # Attempt to download the file
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()  # Raise HTTPError if response is not 200 OK

            # Detect incorrect downloads (e.g., HTML error pages instead of .mat content)
            if r.content.startswith(b"<html"):
                raise ValueError("Received HTML instead of .mat file")

            # Save the binary content to the target .mat file
            with open(f, "wb") as out:
                out.write(r.content)

        except Exception as e:
            print(f"Error downloading {file_id}.mat: {e}")
            download_tries -= 1
            if f.exists():
                f.unlink()
            continue
        else:
            break
    else:
        print(f"Download failed after 5 attempts: {f}")
        failed_downloads.append((file_id, fault_type, fault_code, rpm, sensor_location, load, index))

# Summary
print("\n\nDownload complete\n")

if failed_downloads:
    print(f"Failed to download {len(failed_downloads)} files:")
    for f in failed_downloads:
        print("  " + str(f))
    print()

# ###
# # PROCESSING INTO A SINGLE FILE
# ###

dfs = []

for path_object in download_base.rglob("*.mat"):
    print("Found .mat file:", path_object)
    if not path_object.is_file() or path_object.suffix != ".mat":
        continue

    file_id = int(path_object.stem)  # e.g., "209" from "209.mat"

    # Find the matching metadata from files list
    try:
        meta = next(f for f in files if f[0] == file_id)
    except StopIteration:
        print(f"Metadata not found for: {file_id}")
        continue

    _, fault_type, fault_code, rpm, sensor_location, load, index = meta

    # Parse metadata
    sampling_rate = int(rpm.replace("k", "")) * 1000
    fault_location = sensor_location.split("_")[0]  # "Drive" or "Fan"
    fault_orientation = load if fault_type == "OR" else "-"
    fault_depth = int(fault_code) if fault_code != "-" else 0
    torque = index  # Reuse index as torque placeholder

    # Load .mat content
    print(f"Processing {path_object.name}")
    data = scipy.io.loadmat(path_object)

    # Get keys for each sensor
    DE_key = next((k for k in data if "DE" in k), None)
    FE_key = next((k for k in data if "FE" in k), None)
    BA_key = next((k for k in data if "BA" in k), None)

    signals = [
        (DE_key, "DE"),
        (FE_key, "FE"),
        (BA_key, "BA")
    ]

    for key, location in signals:
        if key is None or data[key].size == 0:
            continue

        measurement = data[key].reshape(-1)

        tmp_df = pd.DataFrame(
            data=[[
                location,
                fault_location,
                fault_type,
                fault_depth,
                fault_orientation,
                sampling_rate,
                torque,
                file_id,
                measurement
            ]],
            columns=[
                "measurement location",
                "fault location",
                "fault type",
                "fault depth",
                "fault orientation",
                "sampling rate",
                "torque",
                "record number",
                "measurement"
            ]
        )

        dfs.append(tmp_df)

# Combine all records
dfs = pd.concat(dfs)

# Make strings explicitly typed for Feather
string_cols = [
    "measurement location",
    "fault location",
    "fault type",
    "fault orientation"
]
dfs[string_cols] = dfs[string_cols].astype("string")

dfs = dfs.reset_index(drop=True)

# Save feather
feather_path = download_base / "CWRU_downloaded.feather"
dfs.to_feather(feather_path)

print("\nFeather file created at:", feather_path)
print("------ DTYPES ------")
print(dfs.dtypes)
print("\n------ HEAD ------")
print(dfs.head())
