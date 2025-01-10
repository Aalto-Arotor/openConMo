import glob

import numpy as np
import matplotlib.pyplot as plt

from randall_methods import randall_method_1, randall_method_2, randall_method_3
import data_utils as du


def plot_method_1(title, time, signal, sq_env_f, sq_env):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.plot(time, signal, c="b", linewidth=0.5)
    ax2.plot(sq_env_f, sq_env, c="b", linewidth=0.5)

    # ax1.set_ylim(-3.5, 3.5)
    # ax2.set_ylim(0, 0.53)
    ax1.set_xlim(0, 0.5)
    ax2.set_xlim(0, 550)
    ax1.set_title("a", loc="left")
    ax2.set_title("b", loc="left")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_method_3(title, time, signal, sq_env_f, sq_env):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.plot(time, signal, c="b", linewidth=0.5)
    ax2.plot(sq_env_f, sq_env, c="b", linewidth=0.5)

    ax1.set_xlim(0, 0.5)
    ax2.set_xlim(0, 550)
    ax1.set_title("a", loc="left")
    ax2.set_title("b", loc="left")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    ax1.set_xlim(0, 0.5)
    ax2.set_xlim(0, 550)


def randall_fig_5():
    '''
    Record 209DE (12k, 0.021 in. drive end inner race fault, 1797 rpm).
    (a) Raw time signal; cursors at 1/fr. (b) Envelope spectrum from
    Method 1 (raw signal); Y1 diagnosis.
    '''
    fs = 12e3
    mat_files = glob.glob("CWRU-dataset/**/*209*.mat", recursive=True)

    signal, _, _, rpm = du.extract_signals(mat_files[0], normal=False)
    num_samples = int(fs*10)
    signal = signal[:num_samples]
    time = np.linspace(0, 1/fs*num_samples, num_samples)

    sq_env_f, sq_env = randall_method_1(signal, fs)
    title = mat_files[0].replace("/", "_")
    plot_method_1(title, time, signal, sq_env_f, sq_env)


def randall_fig_6():
    '''
    Record 209DE (12k, 0.021 in. drive end inner race fault, 1797 rpm).
    (a) Raw time signal; cursors at 1/fr. (b) Envelope spectrum from
    Method 1 (raw signal); Y1 diagnosis.
    '''
    fs = 12e3
    mat_files = glob.glob("CWRU-dataset/**/*169*.mat", recursive=True)

    signal, _, _, rpm = du.extract_signals(mat_files[0], normal=False)
    num_samples = int(fs*10)
    signal = signal[:num_samples]
    time = np.linspace(0, 1/fs*num_samples, num_samples)

    sq_env_f, sq_env = randall_method_1(signal, fs)
    title = mat_files[0].replace("/", "_")
    plot_method_1(title, time, signal, sq_env_f, sq_env)


def randall_fig_7():
    '''
    Record 3007DE (12k, 0.028 in. drive end ball fault, 1750 rpm).
    Envelope spectrum from Method 3 (benchmark); cursors at: fr (red dot), BSF
    harmonics (red dash-dot), sidebands spaced at FTF around 2 × BSF and
    4 × BSF (red dot); Y1 diagnosis.
    '''
    pass


def randall_fig_8():
    '''
    Record 118DE (12k, 0.007 in. drive end ball fault, 1797 rpm). Envelope
    spectrum from Method 1 (raw signal); finely tuned cursors at FTF (shown to
    be 0.4 fr); N1 diagnosis.
    '''
    pass


def randall_fig_9():
    '''
    Record 222DE (12k, 0.021 in. drive end ball fault, 1797 rpm). Envelope
    spectrum from Method 3 (benchmark); Y2 diagnosis
    '''
    fs = 12e3
    mat_files = glob.glob("CWRU-dataset/**/222*.mat", recursive=True)

    # Exctract DE signal
    signal, _, _, rpm = du.extract_signals(mat_files[0], normal=False)

    num_samples = int(fs*10)
    signal = signal[:num_samples]
    t, s, sq_env_f, sq_env = randall_method_3(signal, fs, N=16384, Delta=500)

    fig, ax1 = plt.subplots()

    ax1.plot(sq_env_f, sq_env, c="b", linewidth=0.5)
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 8e-3)
    ax1.set_title("a", loc="left")

    fig.suptitle(mat_files[0])
    plt.tight_layout()
    plt.show()


def randall_fig_11():
    '''
    Record 133DE (12k, 0.007 in. drive end outer race fault centred, 1730 rpm).
    (a) Raw time signal; cursors at 1/fr.
    (b) Envelope spectrum from Method 1 (raw signal); Y1 diagnosis.
    '''
    fs = 12e3
    mat_files = glob.glob("CWRU-dataset/**/*133*.mat", recursive=True)
    print(mat_files)
    # Exctract DE signal
    signal, _, _, rpm = du.extract_signals(mat_files[0], normal=False)
    f_r = rpm/60

    num_samples = int(fs*10)
    signal = signal[:num_samples]
    time = np.linspace(0, 1/fs*num_samples, num_samples)

    sq_env_f, sq_env = randall_method_1(signal, fs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    _, BPFO, _, _ = du.get_bearing_frequencies("DE")

    ax2.axvline(BPFO*f_r, linestyle="--", c="r", linewidth=0.5)
    ax2.axvline(BPFO*2*f_r, linestyle="--", c="r", linewidth=0.5)

    ax1.plot(time, signal, c="b", linewidth=0.5)
    ax2.plot(sq_env_f, sq_env, c="b", linewidth=0.5)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(-4, 4)
    ax2.set_xlim(0, 550)
    ax2.set_ylim(0, 1.1)
    ax1.set_title("a", loc="left")
    ax2.set_title("b", loc="left")

    fig.suptitle(mat_files[0])
    plt.tight_layout()
    plt.show()


def randall_fig_12():
    '''
    Record 200DE (12k, 0.014 in. drive end outer race fault centred, 1730 rpm).
    (a) Time signal from Method 3 (benchmark). (b) Corresponding envelope
    spectrum; N1 diagnosis.
    '''
    fs = 12e3
    mat_files = glob.glob("CWRU-dataset/**/200*.mat", recursive=True)

    # Exctract DE signal
    signal, _, _, rpm = du.extract_signals(mat_files[0], normal=False)

    num_samples = int(fs*10)
    signal = signal[:num_samples]
    t, s, sq_env_f, sq_env = randall_method_3(signal, fs, N=16384, Delta=500, nlevel=3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    # t = np.linspace(0, len(s)/fs, len(s))
    ax1.plot(t, s, c="b", linewidth=0.5)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_title("a", loc="left")

    ax2.plot(sq_env_f, sq_env, c="b", linewidth=0.5)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, 10e-4)
    ax2.set_title("b", loc="left")

    fig.suptitle(mat_files[0])
    plt.tight_layout()
    plt.show()


def randall_fig_19():
    '''
    Record 156DE (12k, 0.007 in. drive end outer race fault opposite, 1797 rpm). (a) Time signal from Method 2 (prewhitening); cursors at 1/BPFO.
    (b) Corresponding envelope spectrum; Y1 diagnosis.
    '''
    fs = 12e3
    mat_files = glob.glob("CWRU-dataset/**/156*.mat", recursive=True)

    # Exctract DE signal
    signal, _, _, rpm = du.extract_signals(mat_files[0], normal=False)

    num_samples = int(fs*10)
    signal = signal[:num_samples]
    time = np.linspace(0,len(signal)/fs,len(signal))

    t,s = randall_method_2(signal,fs)
    sq_env_f, sq_env = randall_method_1(s, fs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    # t = np.linspace(0, len(s)/fs, len(s))
    ax1.plot(t, s, c="b", linewidth=0.5)
    ax1.set_xlim(4.5, 4.66)
    ax1.set_ylim(-0.11, 0.11)
    ax1.set_title("a", loc="left")

    ax2.plot(sq_env_f, sq_env, c="b", linewidth=0.5)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, 1.5e-4)
    ax2.set_title("b", loc="left")

    fig.suptitle(mat_files[0])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # randall_fig_5()
    # randall_fig_6()
    # randall_fig_7()
    # randall_fig_8()
    # randall_fig_9()
    # randall_fig_11()
    # randall_fig_12()
    randall_fig_19()


