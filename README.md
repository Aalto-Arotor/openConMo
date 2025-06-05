# openConMo
This is a Python library for vibration signal based condition monitoring library developed in Aalto University, Finland. The objectives of this library are as follows:

1. Provide easy access to reproducing signal based condition monitoring papers
2. Enable comparison of AI/ML based techniques with conventional signal processing tools


## Install and load datasets

If you want to download the original CWRU dataset in `.mat` format in separate files, you can use the code in `tools/CWRU_download.py`. Just run the code and the files will be downloaded to `data/CWRU/RAW` (the directory will be automatically created if missing). This is somewhat slow (~7 min).

The same code will create a `CWRU.feather` file, which holds the data in a format more easily read with python & pandas. The dataframe is formatted as follows:

| measurement location | fault location | fault type | fault depth (mil) | fault orientation | sampling rate (kHz) | torque (hp) | tags           | measurements           |
| -------------------- | -------------- | ---------- | ----------------- | ----------------- | ------------------- | ----------- | -------------- | ---------------------- |
| `string`             | `string`       | `string`   | `int`             | `string`          | `int`               | `int`       | `list[string]` | `np.array[np.float64]` |
| DE/FE                | DE/FE          | OR/IR/B    | 0/7/14/21/28      | C/OR/OP           | 12/48               | 0/1/2/3     | see below      | measurement samples    |

mil = 0.001 inches

Shorthand explanations:
DE - drive end
FE - fan end
OR (fault type) - outer ring
IR - inner ring
B - ball / rolling element
C - center ()
OR (orientation) - orthogonal
OP - opposite

Possible tags (from [Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study](http://dx.doi.org/10.1016/j.ymssp.2015.04.021)):
`electric noise` - measurement is has patches corrupted by electric noise
`clipped` - measurement is clipped
`identical_DE_and_FE` - measurements from DE and FE sensors are identical except with a scaling factor

## Run notebooks
* Notebook 1: "smith_randall_example.ipynb", reproducing results of "[Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study](http://dx.doi.org/10.1016/j.ymssp.2015.04.021)" by Smith & Randall
* Notebook 2: "matlab_example.ipynb", reproducing results of "[Rolling Element Bearing Fault Diagnosis](https://www.mathworks.com/help/predmaint/ug/Rolling-Element-Bearing-Fault-Diagnosis.html)"

# Documentation

See [Documentation](https://aalto-arotor.github.io/openConMo/)

## Authors
This software is authored and maintained by Sampo Laine, Sampo Haikonen, Aleksanteri Hämäläinen and Elmo Laine, Mechatronics research group, Aalto University.
Please email questions to arotor.software@aalto.fi