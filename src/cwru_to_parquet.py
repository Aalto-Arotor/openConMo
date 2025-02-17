
import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import data_utils as du

def save_cwru_to_parquet(case_num, fs, randall_diagnoses, fault):
    mat_files = glob.glob(f"CWRU-dataset/**/*{case_num}*.mat", recursive=True)
    if len(mat_files) == 0:
        raise ValueError(f"No mat file found for case {case_num}")
    mat_file = mat_files[0]
    rpm_map = {
        0: 1797,
        1: 1772,
        2: 1750,
        3: 1730,
    }
    rpm = rpm_map[int(mat_file[-5])]
    acc_DE, acc_BA, acc_FE, _ = du.extract_signals(mat_file, normal=False)
    for signal, measurement_location, randall_diagnosis in zip([acc_DE, acc_BA, acc_FE], ["DE","BA", "FE"], randall_diagnoses):
        filename = du.save_to_parquet(
            signal=signal,
            fs=fs,
            name=f"cwru {randall_diagnosis} diagnosis",
            measurement_location=measurement_location,
            unit="m/s^2",
            fault=fault,
            rotating_freq_hz=rpm/60,
            meas_id=case_num,
            fault_frequencies=du.cwru_fault_frequencies
        )
    

def run_table_B2():
    fs = 12e3
    diagnoses_table = [
        [105, ["Y2", "Y2", "Y2"]],
        [106, ["Y2", "Y2", "Y2"]],
        [107, ["Y2", "Y2", "Y2"]],
        [108, ["Y2", "Y2", "Y2"]],
        [169, ["Y2", "Y2", "P1"]],
        [170, ["Y2", "P1", "P1"]],
        [171, ["Y2", "P1", "P1"]],
        [172, ["Y2", "P1", "P1"]],
        [209, ["Y1", "Y2", "Y2"]],
        [210, ["Y1", "Y2", "Y2"]],
        [211, ["Y1", "Y2", "Y2"]],
        [212, ["Y1", "Y2", "Y2"]],
        ]
    fault="Drive end inner race fault"

    for case_num, diagnoses in diagnoses_table:
        save_cwru_to_parquet(case_num, fs, diagnoses, fault)

    return
if __name__ == "__main__":
    run_table_B2()
