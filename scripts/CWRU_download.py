from urllib.request import urlretrieve

# import os
import scipy
from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd

###
# FILE SPECIFICATION
###

files = [
    # Format:
    # 0: file number on website
    # 1: Filename to be saved with
    # 2: DE sensor tags
    # 3: FE sensor tags
    # 4: BA sensor tags
    # Normal baseline data: https://engineering.case.edu/bearingdatacenter/normal-baseline-data
    ##
    (97, "normal_0.mat", [], [], []),
    (98, "normal_1.mat", [], [], []),
    (99, "normal_2.mat", [], [], []),
    (100, "normal_3.mat", [], [], []),
    ##
    # 12k Drive End Bearing Fault Data: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
    ##
    # IR 007
    (105, "12k_DE_IR_007_0.mat", [], [], []),
    (106, "12k_DE_IR_007_1.mat", [], [], []),
    (107, "12k_DE_IR_007_2.mat", [], [], []),
    (108, "12k_DE_IR_007_3.mat", [], [], []),
    # IR 014
    (169, "12k_DE_IR_014_0.mat", [], [], []),
    (170, "12k_DE_IR_014_1.mat", [], [], []),
    (171, "12k_DE_IR_014_2.mat", [], [], []),
    (172, "12k_DE_IR_014_3.mat", [], [], []),
    # IR 021
    (209, "12k_DE_IR_021_0.mat", [], [], []),
    (210, "12k_DE_IR_021_1.mat", [], [], []),
    (211, "12k_DE_IR_021_2.mat", [], [], []),
    (212, "12k_DE_IR_021_3.mat", [], [], []),
    # IR 028
    (3001, "12k_DE_IR_028_0.mat", [], [], []),
    (3002, "12k_DE_IR_028_1.mat", [], [], []),
    (3003, "12k_DE_IR_028_2.mat", [], [], []),
    (3004, "12k_DE_IR_028_3.mat", [], [], []),
    # B 007
    (118, "12k_DE_B_007_0.mat", [], [], []),
    (119, "12k_DE_B_007_1.mat", [], [], []),
    (120, "12k_DE_B_007_2.mat", [], [], []),
    (121, "12k_DE_B_007_3.mat", [], [], []),
    # B 014
    (185, "12k_DE_B_014_0.mat", [], [], []),
    (186, "12k_DE_B_014_1.mat", [], [], []),
    (187, "12k_DE_B_014_2.mat", [], [], []),
    (188, "12k_DE_B_014_3.mat", [], [], []),
    # B 021
    (222, "12k_DE_B_021_0.mat", [], [], []),
    (223, "12k_DE_B_021_1.mat", [], [], []),
    (224, "12k_DE_B_021_2.mat", [], [], []),
    (225, "12k_DE_B_021_3.mat", [], [], []),
    # B 028
    (3005, "12k_DE_B_028_0.mat", [], [], []),
    (3006, "12k_DE_B_028_1.mat", [], [], []),
    (3007, "12k_DE_B_028_2.mat", [], [], []),
    (3008, "12k_DE_B_028_3.mat", [], [], []),
    # OR 007 centered
    (130, "12k_DE_OR-C_007_0.mat", [], [], []),
    (131, "12k_DE_OR-C_007_1.mat", [], [], []),
    (132, "12k_DE_OR-C_007_2.mat", [], [], []),
    (133, "12k_DE_OR-C_007_3.mat", [], [], []),
    # OR 007 orthogonal
    (144, "12k_DE_OR-OR_007_0.mat", [], [], []),
    (145, "12k_DE_OR-OR_007_1.mat", [], [], []),
    (146, "12k_DE_OR-OR_007_2.mat", [], [], []),
    (147, "12k_DE_OR-OR_007_3.mat", [], [], []),
    # OR 007 opposite
    (156, "12k_DE_OR-OP_007_0.mat", [], [], []),  # ! 157 skipped on website
    (158, "12k_DE_OR-OP_007_1.mat", [], [], []),
    (159, "12k_DE_OR-OP_007_2.mat", [], [], []),
    (160, "12k_DE_OR-OP_007_3.mat", [], [], []),
    # OR 014 centered
    (197, "12k_DE_OR-C_014_0.mat", [], [], []),
    (198, "12k_DE_OR-C_014_1.mat", [], [], []),
    (199, "12k_DE_OR-C_014_2.mat", [], [], []),
    (200, "12k_DE_OR-C_014_3.mat", [], [], []),
    # OR 028 centered
    (234, "12k_DE_OR-C_028_0.mat", [], [], []),
    (235, "12k_DE_OR-C_028_1.mat", [], [], []),
    (236, "12k_DE_OR-C_028_2.mat", [], [], []),
    (237, "12k_DE_OR-C_028_3.mat", [], [], []),
    # OR 028 orthogonal
    (246, "12k_DE_OR-OR_028_0.mat", [], [], []),
    (247, "12k_DE_OR-OR_028_1.mat", [], [], []),
    (248, "12k_DE_OR-OR_028_2.mat", [], [], []),
    (249, "12k_DE_OR-OR_028_3.mat", [], [], []),
    # OR 028 oppposite
    (258, "12k_DE_OR-OP_028_0.mat", [], [], []),
    (259, "12k_DE_OR-OP_028_1.mat", [], [], []),
    (260, "12k_DE_OR-OP_028_2.mat", [], [], []),
    (261, "12k_DE_OR-OP_028_3.mat", [], [], []),
    ##
    # 48k Drive End Bearing Fault Data: https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data
    ##
    # IR 007
    (109, "48k_DE_IR_007_0.mat", [], [], []),
    (110, "48k_DE_IR_007_1.mat", [], [], []),
    (111, "48k_DE_IR_007_2.mat", [], [], []),
    (112, "48k_DE_IR_007_3.mat", ["electric_noise"], ["electric_noise"], []),
    # IR 014
    (174, "48k_DE_IR_014_0.mat", [], [], []),
    (175, "48k_DE_IR_014_1.mat", [], [], []),
    (176, "48k_DE_IR_014_2.mat", [], [], []),
    (177, "48k_DE_IR_014_3.mat", [], [], []),
    # IR 021
    (213, "48k_DE_IR_021_0.mat", [
     "identical_DE_and_FE"], ["identical_DE_and_FE"], []),
    (214, "48k_DE_IR_021_1.mat", ["clipped"], [], []),
    (215, "48k_DE_IR_021_2.mat", ["clipped"], [], []),
    (217, "48k_DE_IR_021_3.mat", [], [], []),
    # B 007
    (122, "48k_DE_B_007_0.mat", [], [], []),
    (123, "48k_DE_B_007_1.mat", [], [], []),
    (124, "48k_DE_B_007_2.mat", [], [], []),
    (125, "48k_DE_B_007_3.mat", [], [], []),
    # B 014
    (189, "48k_DE_B_014_0.mat", [
     "identical_DE_and_FE"], ["identical_DE_and_FE"], []),
    (190, "48k_DE_B_014_1.mat", [], [], []),
    (191, "48k_DE_B_014_2.mat", ["clipped"], [], []),
    (192, "48k_DE_B_014_3.mat", [], [], []),
    # B 021
    (226, "48k_DE_B_021_0.mat", [
     "identical_DE_and_FE"], ["identical_DE_and_FE"], []),
    (227, "48k_DE_B_021_1.mat", [], [], []),
    (228, "48k_DE_B_021_2.mat", ["clipped"], [], []),
    (229, "48k_DE_B_021_3.mat", ["clipped"], [], []),
    # OR 007 Centered
    (135, "48k_DE_OR-C_007_0.mat", [], [], []),
    (136, "48k_DE_OR-C_007_1.mat", [], [], []),
    (137, "48k_DE_OR-C_007_2.mat", [], [], []),
    (138, "48k_DE_OR-C_007_3.mat", [], [], []),
    # OR 007 orthogonal
    (148, "48k_DE_OR-OR_007_0.mat", [], [], []),
    (149, "48k_DE_OR-OR_007_1.mat", [], [], []),
    (150, "48k_DE_OR-OR_007_2.mat", [], [], []),
    (151, "48k_DE_OR-OR_007_3.mat", [], [], []),
    # OR 007 opposite
    (161, "48k_DE_OR-OP_007_0.mat", [], [], []),
    (162, "48k_DE_OR-OP_007_1.mat", [], [], []),
    (163, "48k_DE_OR-OP_007_2.mat", [], [], []),
    (164, "48k_DE_OR-OP_007_3.mat", [], [], []),
    # OR 014 centered
    (201, "48k_DE_OR-C_014_0.mat",
     ["identical_DE_and_FE"], ["identical_DE_and_FE"], []),
    (202, "48k_DE_OR-C_014_1.mat", [], [], []),
    (203, "48k_DE_OR-C_014_2.mat", [], [], []),
    (204, "48k_DE_OR-C_014_3.mat", [], [], []),
    # OR 021 centered
    (238, "48k_DE_OR-C_021_0.mat",
     ["identical_DE_and_FE"], ["identical_DE_and_FE"], []),
    (239, "48k_DE_OR-C_021_1.mat", [], [], []),
    (240, "48k_DE_OR-C_021_2.mat", ["clipped"], [], []),
    (241, "48k_DE_OR-C_021_3.mat", ["clipped"], [], []),
    # OR 021 orthogonal
    (250, "48k_DE_OR-OR_021_0.mat", [], [], []),
    (251, "48k_DE_OR-OR_021_1.mat", [], [], []),
    (252, "48k_DE_OR-OR_021_2.mat", [], [], []),
    (253, "48k_DE_OR-OR_021_3.mat", [], [], []),
    # OR 021 opposite
    (262, "48k_DE_OR-OP_021_0.mat", [], [], []),
    (263, "48k_DE_OR-OP_021_1.mat", [], [], []),
    (264, "48k_DE_OR-OP_021_2.mat", [], [], []),
    (265, "48k_DE_OR-OP_021_3.mat", [], [], []),
    ##
    # 12k Fan End Bearing Fault Data: https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data
    ##
    # IR 007
    (278, "12k_FE_IR_007_0.mat", [], [], []),
    (279, "12k_FE_IR_007_1.mat", [], [], []),
    (280, "12k_FE_IR_007_2.mat", [], [], []),
    (281, "12k_FE_IR_007_3.mat", [], [], []),
    # IR 014
    (274, "12k_FE_IR_014_0.mat", [], [], []),
    (275, "12k_FE_IR_014_1.mat", [], [], []),
    (276, "12k_FE_IR_014_2.mat", [], [], []),
    (277, "12k_FE_IR_014_3.mat", [], [], []),
    # IR 021
    (270, "12k_FE_IR_021_0.mat", [], [], []),
    (271, "12k_FE_IR_021_1.mat", [], [], []),
    (272, "12k_FE_IR_021_2.mat", [], [], []),
    (273, "12k_FE_IR_021_3.mat", [], [], []),
    # B 007
    (282, "12k_FE_B_007_0.mat", [], [], []),
    (283, "12k_FE_B_007_1.mat", ["electric_noise"], [
     "electric_noise"], ["electric_noise"]),
    (284, "12k_FE_B_007_2.mat", [], [], []),
    (285, "12k_FE_B_007_3.mat", [], [], []),
    # B 014
    (286, "12k_FE_B_014_0.mat", [], [], []),
    (287, "12k_FE_B_014_1.mat", [], [], []),
    (288, "12k_FE_B_014_2.mat", [], [], []),
    (289, "12k_FE_B_014_3.mat", [], [], []),
    # B 021
    (290, "12k_FE_B_021_0.mat", [], [], []),
    (291, "12k_FE_B_021_1.mat", [], [], []),
    (292, "12k_FE_B_021_2.mat", [], [], []),
    (293, "12k_FE_B_021_3.mat", [], [], []),
    # OR 007 centered
    (294, "12k_FE_OR-C_007_0.mat", [], [], []),
    (295, "12k_FE_OR-C_007_1.mat", [], [], []),
    (296, "12k_FE_OR-C_007_2.mat", [], [], []),
    (297, "12k_FE_OR-C_007_3.mat", [], [], []),
    # OR 007 orthogonal
    (298, "12k_FE_OR-OR_007_0.mat", [], [], []),
    (299, "12k_FE_OR-OR_007_1.mat", [], [], []),
    (300, "12k_FE_OR-OR_007_2.mat", [], [], []),
    (301, "12k_FE_OR-OR_007_3.mat", [], [], []),
    # OR 007 opposite
    (302, "12k_FE_OR-OP_007_0.mat", [], [], []),
    (305, "12k_FE_OR-OP_007_1.mat", [], [], []),  # ! Jump on website
    (306, "12k_FE_OR-OP_007_2.mat", [], [], []),
    (307, "12k_FE_OR-OP_007_3.mat", [], [], []),
    # OR 014 centered
    (313, "12k_FE_OR-C_014_0.mat", [], [], []),
    # OR 014 orthogonal
    # ! These were flipped on the website
    (310, "12k_FE_OR-OR_014_0.mat", [], [], []),
    # ! These were flipped on the website
    (309, "12k_FE_OR-OR_014_1.mat", [], [], []),
    (311, "12k_FE_OR-OR_014_2.mat", [], [], []),
    (312, "12k_FE_OR-OR_014_3.mat", [], [], []),
    # OR 021 centered
    (315, "12k_FE_OR-C_021_0.mat", [], [], []),
    # OR 021 orthogonal
    # ! Notice this starts from 1 HP, 0 HP missing from website
    (316, "12k_FE_OR-OR_021_1.mat", [], [], []),
    (317, "12k_FE_OR-OR_021_2.mat", [], [], []),
    (318, "12k_FE_OR-OR_021_3.mat", [], [], []),
]

###
# DOWNLOADING
###

# File url base
url_base = "https://engineering.case.edu/sites/default/files/{}.mat"

# Download directory
cur_dir = None
# For notebooks # FIXME remove
if "__file__" not in globals():
    cur_dir = Path().absolute()
else:
    cur_dir = Path(__file__)

download_dir = cur_dir / "data" / "CWRU" / "RAW"
download_dir.mkdir(parents=True, exist_ok=True)


failed_downloads = []

# Start downloading
print(f"Downloading file 0/{len(files)}", end="\r")
for i, file in enumerate(files):
    print(f"Downloading file {i + 1}/{len(files)}", end="\r")

    # Filename for new file
    f = download_dir / file[1]

    # Skip if file already exists
    if not f.exists():
        # File download fails sometimes, retry 5 times
        download_tries = 5
        while download_tries > 0:
            try:
                # Download
                urlretrieve(url_base.format(file[0]), f)
            except:
                # print("!!!!! ", file[1])
                # Retry
                download_tries -= 1
                continue
            else:
                break
        else:
            # If 5 retries is not enough
            failed_downloads.append(file)
            # Delete file if it failed to download
            f.unlink()

print()
print()
print("Download complete")
print()

if len(failed_downloads) > 0:
    print(f"Failed to download {len(failed_downloads)} files:")
    for f in failed_downloads:
        print("  " + f)
print()

###
# PROCESSING INTO A SINGLE FILE
###

dfs = []

for path_object in download_dir.rglob("*.mat"):
    if not path_object.is_file():
        continue

    measurement_specs = path_object.stem.split("_")

    if "normal" in path_object.stem:
        sampling_rate = 48
        fault_location = "-"
        fault_type = "normal"
        fault_orientation = "-"
        fault_depth = 0
        torque = int(measurement_specs[1])
    else:
        sampling_rate = int(measurement_specs[0][:2])
        fault_location = measurement_specs[1]
        fault_type = measurement_specs[2].split("-")[0]
        fault_orientation = measurement_specs[2].split(
            "-")[1] if fault_type == "OR" else "-"
        fault_depth = int(measurement_specs[3])
        torque = int(measurement_specs[4])

    data = loadmat(path_object)
    DE_key = next((k for k in data if "DE" in k), None)
    FE_key = next((k for k in data if "FE" in k), None)
    BA_key = next((k for k in data if "BA" in k), None)

    sensor_keys = [("DE", DE_key), ("FE", FE_key), ("BA", BA_key)]

    for sensor_label, key in sensor_keys:
        if key is None or key not in data:
            continue

        signal = data[key].reshape(-1)
        measurement_id = f"{path_object.stem}_{sensor_label}"

        tmp_df = pd.DataFrame({
            "measurement_id": measurement_id,
            "sample_index": np.arange(len(signal)),
            "measurement": signal,
            "measurement location": sensor_label,
            "fault location": fault_location,
            "fault type": fault_type,
            "fault depth": fault_depth,
            "fault orientation": fault_orientation,
            "sampling rate": sampling_rate,
            "torque": torque,
        })
        dfs.append(tmp_df)

# Combine all rows
dfs = pd.concat(dfs, ignore_index=True)

# Convert to string types
string_cols = [
    "measurement location",
    "fault location",
    "fault type",
    "fault orientation",
]
dfs[string_cols] = dfs[string_cols].astype("string")

# Save exploded format
dfs.to_parquet(
    Path.cwd() / "data" / "CWRU" / "CWRU_downloaded.parquet",
    index=False,
)

print()
print("DTYPES")
print("------")
print(dfs.dtypes)
print()
print("DATAFRAME HEAD")
print("--------------")
print(dfs.head)
