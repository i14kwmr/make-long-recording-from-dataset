import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

# process wavfiles


def wavread(fn, resr=None):
    if resr is None:
        data, sr = sf.read(fn)
    else:
        data, sr = sf.read(fn)
        data = signal.resample(data, int(resr * len(data) / sr))
        sr = resr
    f = sf.info(fn)
    return data, sr, f.subtype


def wavwrite(fn, data, sr, subtype, resr=None):
    if resr is None:
        sf.write(fn, data, sr, subtype)
    else:
        data = signal.resample(data, int(resr * len(data) / sr))
        sf.write(fn, data, resr, subtype)


# リサンプリングする関数
def sinc_resample(y, orig_sr, target_sr, axis=-1):
    """
    Resample a signal based on sinc interpolation.

    Parameters
    ----------
    y : np.ndarray (..., lorig, ...)
        a time-domain signal. Number of samples is 'lorig'.

    orig_sr : positive scalar
        original sampling rate of 'y'

    target_sr : positive scalar
        target sampling rate

    axis: integer
    This parameter indicates the dimension of time.

    Returns
    -------
    x : np.ndarray (..., ltarget, ...)
        a resampled signal. Number of samples is 'ltarget'.

    """

    lorig = y.shape[axis]
    ltarget = int(np.floor(lorig * (target_sr / orig_sr)))
    m = np.arange(ltarget)

    y = np.moveaxis(y, axis, -1)
    target_shape = list(y.shape)
    target_shape[-1] = ltarget

    x = np.zeros(target_shape, dtype=y.dtype)

    if target_sr < orig_sr:
        coef = target_sr / orig_sr
    else:
        coef = 1

    for l in range(lorig):
        x += coef * y[..., l, None] * np.sinc(coef * (m * orig_sr / target_sr - l))

    return np.moveaxis(x, -1, axis)


# show spectrogram


def show_spectrogram(spec, ref, title, path=None):

    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    plt.rcParams["pdf.fonttype"] = 42
    plt.figure(figsize=(16, 10), dpi=250)  # dpi 変更可能

    # make power spectrogram
    powspec = librosa.amplitude_to_db(np.abs(spec), ref=ref)
    librosa.display.specshow(
        powspec,
        sr=16000,
        hop_length=2048,
        cmap="rainbow",
        x_axis="time",
        y_axis="hz",
        norm=Normalize(vmin=40, vmax=-5),
    )

    # plot power spectrogram
    plt.title(title, fontsize=18)
    plt.colorbar(format="%+2.0fdB")
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Frequency [Hz]", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tight_layout()

    if path == None:
        plt.show()
    else:
        plt.savefig(path, transparent=True)

    plt.close()


def show_spectrogram_test(spec, ref, title, path=None):

    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    plt.rcParams["pdf.fonttype"] = 42
    plt.figure(figsize=(16, 4), dpi=250)  # dpi 変更可能

    # make power spectrogram
    powspec = librosa.amplitude_to_db(np.abs(spec), ref=ref)
    librosa.display.specshow(
        powspec,
        sr=16000,
        hop_length=2048,
        cmap="rainbow",
        x_axis="time",
        y_axis="hz",
        norm=Normalize(vmin=100, vmax=30),
    )

    # plot power spectrogram
    plt.title(title, fontsize=18)
    plt.colorbar(format="%+2.0fdB")
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Frequency [Hz]", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tight_layout()

    if path == None:
        plt.show()
    else:
        plt.savefig(path, transparent=True)

    plt.close()


# データ書き出し関係


def write_output_snr(
    distribution,
    parameter,
    input_snr,
    output_snr,
    output_snr_sync,
    target_name,
    object_name,
    path,
):

    df = pd.DataFrame(
        [
            [
                distribution,
                parameter,
                input_snr,
                output_snr,
                target_name,
                object_name,
                "Sync. (off)",
            ],
            [
                distribution,
                parameter,
                input_snr,
                output_snr_sync,
                target_name,
                object_name,
                "Sync. (on)",
            ],
        ],
        columns=[
            "distribution",
            "parameter",
            "Input SNR [dB]",
            "Output SNR [dB]",
            "target",
            "object",
            "sync",
        ],
    )
    df.to_csv(path)


def write_cost_buffer(cost_buff, cost_buff_sync, path):

    width = 1
    # write csv
    iteration = [i for i in range(len(cost_buff))]

    df = pd.DataFrame(
        [iteration, cost_buff, cost_buff_sync],
        index=["iteration", "Sync. (off)", "Sync. (on)"],
    ).T
    df.to_csv(path + "/cost_buffer.csv")

    start = 1

    # plot cost
    # plt.title("Objective function", fontsize=18)
    iteration = np.array(iteration)
    cost_buff = np.array(cost_buff)
    plt.plot(iteration[start:], cost_buff[start:], color="C1")
    plt.plot(
        iteration[start::width],
        cost_buff[start::width],
        label="Sync. (off)",
        marker="o",
        linestyle="None",
        markersize=3,
        color="C1",
    )
    plt.plot(iteration[start:], cost_buff_sync[start:], color="C2")
    plt.plot(
        iteration[start::width],
        cost_buff_sync[start::width],
        label="Sync. (on)",
        marker="o",
        linestyle="None",
        markersize=3,
        color="C2",
    )

    plt.xlabel("Number of iteration steps", fontsize=18)
    plt.ylabel("Objective function", fontsize=18)
    plt.grid()

    plt.legend()
    plt.tight_layout()
    # plt.tick_params(labelsize=16)

    plt.savefig(
        path + "/cost_buffer.pdf",
        transparent=True,
    )
    plt.close()


def write_snr_buffer(snr_buff, snr_buff_sync, path):

    width = 1
    # write csv
    iteration = [i for i in range(len(snr_buff))]

    df = pd.DataFrame(
        [iteration, snr_buff, snr_buff_sync],
        index=["iteration", "Sync. (off)", "Sync. (on)"],
    ).T
    df.to_csv(path + "/snr_buffer.csv")

    start = 1

    # plot cost
    # plt.title("Objective function", fontsize=18)
    iteration = np.array(iteration)
    snr_buff = np.array(snr_buff)
    plt.plot(iteration[start:], snr_buff[start:], color="C1")
    plt.plot(
        iteration[start::width],
        snr_buff[start::width],
        label="Sync. (off)",
        marker="o",
        linestyle="None",
        markersize=5,
        color="C1",
    )
    plt.plot(iteration[start:], snr_buff_sync[start:], color="C2")
    plt.plot(
        iteration[start::width],
        snr_buff_sync[start::width],
        label="Sync. (on)",
        marker="o",
        linestyle="None",
        markersize=5,
        color="C2",
    )

    plt.xlabel("Number of iteration steps", fontsize=18)
    plt.ylabel("Output SNR [dB]", fontsize=18)
    plt.grid()

    plt.legend()
    plt.tight_layout()
    # plt.tick_params(labelsize=16)

    plt.savefig(
        path + "/snr_buffer.pdf",
        transparent=True,
    )
    plt.close()


# make dirs


def my_makedirs(path):
    import os

    if not os.path.isdir(path):
        os.makedirs(path)
