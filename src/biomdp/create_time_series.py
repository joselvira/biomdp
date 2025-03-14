# %% -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 16:56:08 2025
Functions to create artificial sinusoidal time series.
Mainly based on xarray.

@author: josel
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.0"
__date__ = "05/03/2025"

"""
Updates:
    05/03/2025, v0.1.0
        - Initial version with create_time_series_xr function.

"""

from typing import List
import numpy as np

# import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt
from pathlib import Path

import matplotlib.pyplot as plt

# import seaborn as sns


def create_time_series_xr(
    rnd_seed: int | None = None,
    num_subj: int = 10,
    fs: float = 100.0,
    IDini: int = 0,
    range_offset: List[float] = [-2.0, -0.5],
    range_amp: List[float] = [1.0, 2.2],
    range_freq: List[float] = [1.8, 2.4],
    range_af: List[float] = [0.0, 1.0],
    range_duration: List[float] = [5.0, 5.1],
    amplific_noise: List[float] = [0.4, 0.7],
    fc_noise: List[float] = [7.0, 12.0],
) -> xr.DataArray:
    """
    Create a dummy data sample based on sine waves with noise

    Parameters
    ----------
    rnd_seed : int, optional
        Seed for random number generation. Default is None.
    num_subj : int, optional
        Number of subjects. Default is 10.
    fs : float, optional
        Sampling frequency. Default is 100.0.
    IDini : int, optional
        Initial ID number. Default is 0.
    range_offset : list of float, optional
        Range of offset values. Default is [-2.0, -0.5].
    range_amp : list of float, optional
        Range of amplitude values. Default is [1.0, 2.2].
    range_freq : list of float, optional
        Range of frequency values. Default is [1.8, 2.4].
    range_af : list of float, optional
        Range of phase angle values in degrees. Default is [0.0, 1.0].
    range_duration : list of float, optional
        Range of duration values. Default is [5.0, 5.1].
    amplific_noise : list of float, optional
        Range of amplitude values for the noise. Default is [0.4, 0.7].
    fc_noise : list of float, optional
        Range of cut-off frequency values for the noise. Default is [7.0, 12.0].

    Returns
    -------
    xarray.DataArray
        Data array of shape (num_subj, time) with the dummy data.
    """

    if rnd_seed is not None:
        # To maintain consistency when creating the random data
        np.random.seed(rnd_seed)
    subjects = []
    for subj in range(num_subj):
        # print(subj)
        a = np.random.uniform(range_amp[0], range_amp[1])
        of = np.random.uniform(range_offset[0], range_offset[1])
        f = np.random.uniform(range_freq[0], range_freq[1])
        af = np.deg2rad(
            np.random.uniform(range_af[0], range_af[1])
        )  # transform to radians
        err = a * np.random.uniform(amplific_noise[0], amplific_noise[1])
        fc_err = np.random.uniform(fc_noise[0], fc_noise[1])
        duration = np.random.uniform(range_duration[0], range_duration[1])

        Ts = 1.0 / fs  # time interval between data in seconds
        t = np.arange(0, duration, Ts)

        senal = np.array(of + a * np.sin(2 * np.pi * f * t + af))

        # Create a controlled random noise
        pasadas = 2.0  # nº de pasadas del filtro adelante y atrás
        orden = 2
        Cf = (2 ** (1 / pasadas) - 1) ** (
            1 / (2 * orden)
        )  # correqtion factor. Para 2nd order = 0.802
        Wn = 2 * fc_err / fs / Cf
        b1, a1 = butter(orden, Wn, btype="low")
        noise = filtfilt(b1, a1, np.random.uniform(a - err, a + err, len(t)))

        #################################
        subjects.append(senal + noise)
        # subjects.append(np.expand_dims(senal + noise, axis=0))
        # sujeto.append(pd.DataFrame(senal + noise, columns=['value']).assign(**{'ID':'{0:02d}'.format(subj+IDini), 'time':np.arange(0, len(senal)/fs, 1/fs)}))

    # Pad data to last the same
    from itertools import zip_longest

    data = np.array(list(zip_longest(*subjects, fillvalue=np.nan)))

    data = xr.DataArray(
        data=data,
        coords={
            "time": np.arange(data.shape[0]) / fs,
            "ID": [
                f"{i:0>2}" for i in range(IDini, IDini + num_subj)
            ],  # pad with zeros on the left.
        },
    )
    return data


# =============================================================================
# %% TESTS
# =============================================================================

if __name__ == "__main__":

    # ----Create single sample
    n = 6  # number of subjects
    duration = 5  # duration in seconds
    freq = 200.0  # sampling frequency

    data = create_time_series_xr(
        num_subj=n,
        fs=freq,
        IDini=1,
        range_offset=[25, 29],
        range_amp=[40, 45],
        range_freq=[1.48, 1.52],
        range_af=[0, 30],
        amplific_noise=[0.4, 0.7],
        fc_noise=[3.0, 3.5],
        range_duration=[duration, duration],
    )
    data.plot.line(x="time")
    plt.show()

    # ----Create multiple samples with pre and post
    rnd_seed = np.random.seed(12340)
    n = 5
    duration = 5

    freq = 200.0
    pre_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[25, 29],
            range_amp=[40, 45],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["a"], "moment": ["pre"]})
        .transpose("ID", "moment", "n_var", "time")
    )

    post_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[22, 26],
            range_amp=[36, 40],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["a"], "moment": ["post"]})
        .transpose("ID", "moment", "n_var", "time")
    )

    var_a = xr.concat([pre_a, post_a], dim="moment")

    var_a.sel(n_var="a").plot.line(x="time", col="moment")
    plt.show()

    # ----Create multiple samples with pre and post and multiple variables
    rnd_seed = np.random.seed(12340)
    n = 5
    duration = 5

    freq = 200.0
    pre_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[25, 29],
            range_amp=[40, 45],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["a"], "moment": ["pre"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    post_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[22, 26],
            range_amp=[36, 40],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["a"], "moment": ["post"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    var_a = xr.concat([pre_a, post_a], dim="moment")

    pre_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[35, 39],
            range_amp=[50, 55],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["b"], "moment": ["pre"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    post_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[32, 36],
            range_amp=[32, 45],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["b"], "moment": ["post"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    var_b = xr.concat([pre_b, post_b], dim="moment")

    pre_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[35, 39],
            range_amp=[10, 15],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["c"], "moment": ["pre"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    post_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[32, 36],
            range_amp=[12, 16],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["c"], "moment": ["post"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    var_c = xr.concat([pre_c, post_c], dim="moment")

    # Concat all the subjects
    daTotal = xr.concat([var_a, var_b, var_c], dim="n_var")
    daTotal.name = "Angle"
    daTotal.attrs["freq"] = 1 / (daTotal.time[1].values - daTotal.time[0].values)
    daTotal.attrs["units"] = "deg"
    daTotal.time.attrs["units"] = "s"

    # Plot
    daTotal.plot.line(x="time", col="moment", hue="ID", row="n_var")
