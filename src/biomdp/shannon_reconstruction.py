# %% -*- coding: utf-8 -*-
"""
Created on Tue Oct 01 13:42:08 2024
Functions to interpolate higher frequencies based on
Shannon reconstruction.

@author: Jose L. L. Elvira
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "v0.1.1"
__date__ = "11/03/2025"

"""
Updates:
    11/03/2025, v0.1.1
        - Adapted to biomdp with translations.

    01/10/2024, v0.1.0
        - Versión inicial basada en la versión C++.

"""

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

# from biomdp.general_processing_functions import detrend_dim


# =============================================================================
# %% Function for Shannon reconstruction
# =============================================================================


def _circularity(data: np.ndarray, solapar: bool = False) -> np.array:
    """
    Create circularity in a single dimension array
    """
    if np.count_nonzero(~np.isnan(data)) == 0:
        return data

    data_recons = np.full(len(data) * 2, np.nan)

    # delete nans
    data = np.array(data[~np.isnan(data)])
    # plt.plot(data)

    if solapar:
        pass

    else:  # sin solapar data
        mitad = int(len(data) / 2)
        if True:  # len(data) % 2 == 0:  # si es par
            # Carga la 1ª mitad en orden inverso al principio
            data_recons[:mitad] = 2 * data[0] - data[mitad:0:-1]

            # Continúa cargando todos los datos seguidos
            data_recons[mitad : mitad + len(data)] = data

            # Termina cargando la 2ª mitad en orden inverso al final
            data_recons[mitad + len(data) - 1 :] = 2 * data[-1] - data[: mitad - 2 : -1]

            """
            plt.plot(data_recons)

            import pandas as pd

            pd.DataFrame([2 * data[0] - data[mitad:0:-1], data, 2 * data[-1] - data[: mitad - 2 : -1]]).T.plot()
            pd.concat(
                [
                    pd.DataFrame([data[mitad:0:-1]]).T,
                    pd.DataFrame([data]).T,
                    pd.DataFrame([data[-1:mitad:-1]]).T,
                ]
            ).plot()
            """

        else:  # si es impar. OMPROBAR SI ES NECESARIO
            par_impar = 1
            # Carga la 1ª mitad en orden inverso al principio
            data_recons[: mitad + par_impar] = 2 * data[0] - data[mitad:par_impar:-1]

            # Continúa cargando todos los datos seguidos
            data_recons[mitad : mitad + len(data)] = data
            data[-1]
            # Termina cargando la 2ª mitad en orden inverso al final
            data_recons[mitad + len(data) - 1 :] = 2 * data[-1] - data[: mitad - 2 : -1]

    """
    else
            {
                DatCircular = new double[DatLeido.Length * 2];     //array con los datos cortados en circularidad sin línea tendencia.
                DatTratado = new double[DatLeido.Length * 2];  //array con la corrección de la línea de tendencia para que coincida inicio y final. Sea par o impar, el nº final es el doble -2.

                //primero comprueba si es par o impar
                int resto;
                Math.DivRem(DatLeido.Length, 2, out resto);

                if (resto == 0) //nº de datos PAR
                {
                    //carga la 1ª mitad en orden inverso al principio                    
                    for (int i = 0; i < (int)(DatLeido.Length / 2); i++)
                    {
                        DatCircular[i] = 2 * DatLeido[0] - DatLeido[(int)(DatLeido.Length / 2) - i - 1];//DatLeido[GNumVarGrafY][0] - DatLeido[GNumVarGrafY][i] + DatLeido[GNumVarGrafY][0];
                    }
                    //carga todos los datos seguidos
                    for (int i = 0; i < (int)(DatLeido.Length) - 1; i++)
                    {
                        DatCircular[(int)(DatLeido.Length / 2) + i] = DatLeido[i];
                    }
                    //carga la 2ª mitad en orden inverso al final
                    for (int i = 0; i < (int)(DatLeido.Length / 2) ; i++)
                    {
                        DatCircular[(int)(DatCircular.Length) - i - 1] = 2 * DatLeido[DatLeido.Length - 1] - DatLeido[i + (int)(DatLeido.Length / 2)];//DatLeido[GNumVarGrafY][DatLeido[GNumVarGrafY].Length - 1] - DatLeido[GNumVarGrafY][i] + DatLeido[GNumVarGrafY][DatLeido[GNumVarGrafY].Length - 1];
                    }
                    //el último lo mete a mano
                    DatCircular[(int)(DatCircular.Length) - (int)(DatLeido.Length / 2) - 1] = DatLeido[DatLeido.Length - 1];
                }
                else //nº de datos IMPAR
                {
                    //carga la 1ª mitad en orden inverso al principio                    
                    for (int i = 0; i < (int)(DatLeido.Length / 2) + 1; i++)
                    {
                        DatCircular[i] = 2 * DatLeido[0] - DatLeido[(int)(DatLeido.Length / 2) - i];//DatLeido[GNumVarGrafY][0] - DatLeido[GNumVarGrafY][i] + DatLeido[GNumVarGrafY][0];
                    }
                    //carga todos los datos seguidos
                    for (int i = 0; i < (int)(DatLeido.Length); i++)
                    {
                        DatCircular[(int)(DatLeido.Length / 2) + i + 1] = DatLeido[i];
                    }
                    //carga la 2ª mitad en orden inverso al final
                    for (int i = 0; i < (int)(DatLeido.Length / 2) ; i++)
                    {
                        DatCircular[(int)(DatCircular.Length) - i - 1] = 2 * DatLeido[DatLeido.Length - 1] - DatLeido[i + (int)(DatLeido.Length / 2) + 1];//DatLeido[GNumVarGrafY][DatLeido[GNumVarGrafY].Length - 1] - DatLeido[GNumVarGrafY][i] + DatLeido[GNumVarGrafY][DatLeido[GNumVarGrafY].Length - 1];
                    }
                }
                
            }

    """
    return data_recons


def create_circularity_xr(
    daData: xr.DataArray, freq: float = None, overlap: bool = False
) -> xr.DataArray:
    """
    Create circularity in signal
    """

    """
    data = daData[0,0,0].data
    ini = daWindow[0,0,0].sel(event='ini').data
    fin = daWindow[0,0,0].sel(event='fin').data
    """
    # daRecortado = recorta_ventana_analisis(daData, daWindow)
    daCirc = xr.apply_ufunc(
        _circularity,
        daData,
        overlap,
        input_core_dims=[["time"], []],
        output_core_dims=[["newtime"]],
        vectorize=True,
    ).rename({"newtime": "time"})

    freq = daData.freq if freq is None else freq
    daCirc = daCirc.assign_coords(time=np.arange(0, len(daCirc.time)) / freq)

    return daCirc


def _detrend(data: np.array) -> np.array:
    """
    Detrend the signal along a single dimension
    """

    trend_line = np.linspace(data[0], data[-1], len(data))
    # plt.plot(trend_line)
    # plt.plot(data)
    # plt.plot(data - trend_line)
    return data - trend_line


def detrend_xr(
    daData: xr.DataArray, freq: float | None = None, overlap: bool | None = False
) -> xr.DataArray:
    """
    Create circularity in signal
    """

    """
    data = daData[0,0,0].data
    ini = daWindow[0,0,0].sel(event='ini').data
    fin = daWindow[0,0,0].sel(event='fin').data
    """
    # daRecortado = recorta_ventana_analisis(daData, daWindow)
    data_tratado = xr.apply_ufunc(
        _detrend,
        daData,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )

    return data_tratado


def _shannon_reconstruction(
    data: np.array,
    old_freq: int | float | None = None,
    new_freq: int | float | None = None,
) -> np.array:
    """
    Reconstruct signal with Shannon reconstruction
    """
    if old_freq is None or new_freq is None:
        raise ValueError("You have to specify both old_freq and new_freq.")

    delta = 1 / old_freq
    new_delta = 1 / new_freq

    tiempo_muestra = (
        len(data) - 1
    ) * delta  # data length is doubled due to circularity
    # num_puntos_nuevo = int(tiempo_muestra / new_delta) + 1
    num_puntos_nuevo = int(len(data) * new_freq / old_freq)

    fc = 1 / (2 * delta)  # Nyquist frequency

    """
    tpo=time.perf_counter()
    for i in range(50):
        # for loop version
        data_tratado = np.zeros(num_puntos_nuevo)
        for i in range(num_puntos_nuevo):
            t = i * new_delta
            for n in range(len(data)):
                if t - n * delta != 0:
                    m = np.sin(2 * np.pi * fc * (t - n * delta)) / (np.pi * (t - n * delta))
                else:
                    m = 1 / delta
                data_tratado[i] += data[n] * m
            data_tratado[i] *= delta
    print(time.perf_counter()-tpo)
    """

    # Vectorized version (~ x150 faster)
    # Create a grid of time points for the new signal
    t = np.arange(num_puntos_nuevo) * new_delta

    # Create a matrix of time differences (t - n*delta)
    n = np.arange(len(data))
    time_diff_matrix = t[:, np.newaxis] - n * delta

    # Calculate the sinc function values, handling the case where t - n*delta = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.sin(2 * np.pi * fc * time_diff_matrix) / (np.pi * time_diff_matrix)
    m[np.isnan(m)] = 1 / delta  # Replace NaN values with 1/delta

    # Perform the reconstruction using matrix multiplication
    data_tratado = delta * np.dot(m, data)

    """
    {
        double delta = 1 / (double)frecMuestreo;            //tiempo entre datos en la frecuencia de muestreo original
        double newDelta = 1 / (double)nuevaFrecMuestreo;    //tiempo entre datos en la frecuencia de muestreo nueva 
        

        double tiempoMuestra = (double)(DatLeido.Length-1) * delta;  //el datleido que llega está duplicado de tiempo por la circularidad
        int NumPuntosNuevo = (int)((DatLeido.Length/*-1???*/) * nuevaFrecMuestreo / frecMuestreo);// (int)(tiempoMuestra / newDelta)+2;//estaba con +1
        double fc = 1 / (2 * delta);    //Nyquist frecuency
        //double NyquistFrec = (double)(frecMuestreo/2);
        //double NyquistFrec2 = 2 * NyquistFrec;

        double t, m;

        DatTratado = new double[NumPuntosNuevo];

        progressBar2.Maximum = NumPuntosNuevo + 1;
        progressBar2.Value = 0;


        for (int i = 0; i < NumPuntosNuevo; i++)
        {
            t = (double)i * newDelta;
            for (int n = 0; n < DatLeido.Length; n++)
            {
                if (t - (double)n * delta != 0)
                    m = Math.Sin(2 * Math.PI * fc * (t - (double)n * delta)) / (Math.PI * (t - (double)n * delta));
                else
                    m = 1 / delta;

                DatTratado[i] = DatTratado[i] + DatLeido[n] * m;

                //progressBar2.Value++;
            }
            DatTratado[i] = DatTratado[i] * delta;
            progressBar2.Value++;
        }
    }
    """

    return data_tratado


def shannon_reconstruction_xr(
    daData: xr.DataArray,
    old_freq: int | float | None = None,
    new_freq: int | float | None = None,
) -> xr.DataArray:
    """
    Reconstruct with Shannon
    """

    """
    data = daData[0,0,0].data
    ini = daWindow[0,0,0].sel(event='ini').data
    fin = daWindow[0,0,0].sel(event='fin').data
    """
    # daRecortado = recorta_ventana_analisis(daData, daWindow)

    old_freq = daData.freq if old_freq is None else old_freq

    data_tratado = xr.apply_ufunc(
        _shannon_reconstruction,
        daData,
        old_freq,
        new_freq,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
    )

    return data_tratado


# =============================================================================
# %% TESTS
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # %%---- Create a sample
    # =============================================================================

    import numpy as np
    import time

    # import pandas as pd
    import xarray as xr
    from scipy.signal import butter, filtfilt
    from pathlib import Path

    import matplotlib.pyplot as plt

    # import seaborn as sns
    from biomdp.create_time_series import create_time_series_xr

    rnd_seed = np.random.seed(12340)
    n = 5
    duration = 5

    freq = 200.0
    Pre_a = (
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
    Post_a = (
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
    var_a = xr.concat([Pre_a, Post_a], dim="moment")

    Pre_b = (
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
    Post_b = (
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
    var_b = xr.concat([Pre_b, Post_b], dim="moment")

    Pre_c = (
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
    Post_c = (
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
    var_c = xr.concat([Pre_c, Post_c], dim="moment")

    # concatena todos los sujetos
    daAll = xr.concat([var_a, var_b, var_c], dim="n_var")
    daAll.name = "Angle"
    daAll.attrs["freq"] = 1 / (
        daAll.time[1].values - daAll.time[0].values
    )  # incluimos la frecuencia como atributo
    daAll.attrs["units"] = "deg"
    daAll.time.attrs["units"] = "s"

    # Graphs
    daAll.plot.line(x="time", col="moment", hue="ID", row="n_var")

    # =============================================================================
    # %% Test the functions
    # =============================================================================
    """
    data = daAll[0, 0, 0, 120:201].data
    data2 = _circularity(data)

    data2 = _detrend(data2)
    plt.plot(data2)
    data2[0]
    data2[-1]
    """
    daAllSlice = daAll.isel(time=slice(100, 201))
    daAllSlice.plot.line(x="time", col="moment", hue="ID", row="n_var")

    daAllSliceCirc = create_circularity_xr(daAllSlice)
    daAllSliceCirc.plot.line(x="time", col="moment", hue="ID", row="n_var")

    daAllSliceCircDetrend = _detrend(daAllSliceCirc[0, 0, 0, :].data)
    plt.plot(daAllSliceCircDetrend)

    daAllSliceCircDetrend = detrend_xr(daAllSliceCirc)
    daAllSliceCircDetrend.plot.line(x="time", col="moment", hue="ID", row="n_var")

    data = daAllSliceCircDetrend[0, 0, 0, :].data
    plt.plot(data)

    # Misma forma original y reconstnoise
    dataShannon = _shannon_reconstruction(data, old_freq=daAll.freq, new_freq=400.0)
    plt.plot(np.arange(len(dataShannon)) / 400, dataShannon, label="Shannon rec")
    plt.plot(np.arange(len(data)) / daAll.freq, data, label="Original", ls="--")
    plt.legend()
    plt.show()

    dataShannon = _shannon_reconstruction(data, old_freq=daAll.freq, new_freq=800.0)
    plt.plot(
        np.arange(len(dataShannon[-100:])) / 400,
        dataShannon[-100:],
        label="Shannon rec",
        marker="*",
    )
    plt.plot(
        np.arange(len(data[-25:])) / daAll.freq,
        data[-25:],
        label="Original",
        ls="--",
        marker="*",
    )
    plt.legend()
    plt.show()
