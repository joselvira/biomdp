# %% -*- coding: utf-8 -*-
"""
Created on Tue Oct 01 13:42:08 2024
Funciones para interpolar frecuencias mayores basado en la
reconstrucción de Shannon.
Basado en xarray.

@author: Jose L. L. Elvira
"""


# =============================================================================
# %% Carga librerías
# =============================================================================

from typing import Optional, Union, Any

import numpy as np
import xarray as xr

import itertools

import matplotlib.pyplot as plt

from biomdp.general_processing_functions import detrend_dim

__author__ = "Jose Luis Lopez Elvira"
__version__ = "v.1.0.0"
__date__ = "01/10/2024"


"""

Modificaciones:
    01/10/2024, v1.0.0
        - Versión inicial basada en versión C++.

"""


# =============================================================================
# %% Function for Shannon reconstruction
# =============================================================================


def _circularity(data: np.ndarray, solapar=False) -> np.array:
    """
    Create circularity in a single dimension
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
    daData: xr.DataArray, freq=None, overlap=False
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


def detrend_xr(daData: xr.DataArray, freq=None, overlap=False) -> xr.DataArray:
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
    old_freq: Union[int, float] = None,
    new_freq: Union[int, float] = None,
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
    old_freq: Union[int, float] = None,
    new_freq: Union[int, float] = None,
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
    # ---- Create a sample
    # =============================================================================

    import numpy as np
    import time

    # import pandas as pd
    import xarray as xr
    from scipy.signal import butter, filtfilt
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    def create_time_series_xr(
        rnd_seed=None,
        num_subj=10,
        Fs=100.0,
        IDini=0,
        rango_offset=[-2.0, -0.5],
        rango_amp=[1.0, 2.2],
        rango_frec=[1.8, 2.4],
        rango_af=[0.0, 1.0],
        rango_duracion=[5.0, 5.1],
        amplific_ruido=[0.4, 0.7],
        fc_ruido=[7.0, 12.0],
    ):
        if rnd_seed is not None:
            np.random.seed(
                rnd_seed
            )  # para mantener la consistencia al crear los datos aleatorios
        subjects = []
        for subj in range(num_subj):
            # print(subj)
            a = np.random.uniform(rango_amp[0], rango_amp[1])
            of = np.random.uniform(rango_offset[0], rango_offset[1])
            f = np.random.uniform(rango_frec[0], rango_frec[1])
            af = np.deg2rad(
                np.random.uniform(rango_af[0], rango_af[1])
            )  # lo pasa a radianes
            err = a * np.random.uniform(amplific_ruido[0], amplific_ruido[1])
            fc_err = np.random.uniform(fc_ruido[0], fc_ruido[1])
            duracion = np.random.uniform(rango_duracion[0], rango_duracion[1])

            Ts = 1.0 / Fs  # intervalo de tiempo entre datos en segundos
            t = np.arange(0, duracion, Ts)

            senal = np.array(of + a * np.sin(2 * np.pi * f * t + af))

            # Crea un ruido aleatorio controlado
            pasadas = 2.0  # nº de pasadas del filtro adelante y atrás
            orden = 2
            Cf = (2 ** (1 / pasadas) - 1) ** (
                1 / (2 * orden)
            )  # correction factor. Para 2nd order = 0.802
            Wn = 2 * fc_err / Fs / Cf
            b1, a1 = butter(orden, Wn, btype="low")
            ruido = filtfilt(b1, a1, np.random.uniform(a - err, a + err, len(t)))

            #################################
            subjects.append(senal + ruido)
            # subjects.append(np.expand_dims(senal + ruido, axis=0))
            # sujeto.append(pd.DataFrame(senal + ruido, columns=['value']).assign(**{'ID':'{0:02d}'.format(subj+IDini), 'time':np.arange(0, len(senal)/Fs, 1/Fs)}))

        # Pad data to last the same
        import itertools

        data = np.array(list(itertools.zip_longest(*subjects, fillvalue=np.nan)))

        data = xr.DataArray(
            data=data,
            coords={
                "time": np.arange(data.shape[0]) / Fs,
                "ID": [
                    f"{i:0>2}" for i in range(num_subj)
                ],  # rellena ceros a la izq. f'{i:0>2}' vale para int y str, f'{i:02}' vale solo para int
            },
        )
        return data

    rnd_seed = np.random.seed(
        12340
    )  # fija la aleatoriedad para asegurarse la reproducibilidad
    n = 5
    duracion = 5

    freq = 200.0
    Pre_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[25, 29],
            rango_amp=[40, 45],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["a"], "momento": ["pre"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    Post_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[22, 26],
            rango_amp=[36, 40],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["a"], "momento": ["post"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    var_a = xr.concat([Pre_a, Post_a], dim="momento")

    Pre_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[35, 39],
            rango_amp=[50, 55],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["b"], "momento": ["pre"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    Post_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[32, 36],
            rango_amp=[32, 45],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["b"], "momento": ["post"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    var_b = xr.concat([Pre_b, Post_b], dim="momento")

    Pre_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[35, 39],
            rango_amp=[10, 15],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["c"], "momento": ["pre"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    Post_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[32, 36],
            rango_amp=[12, 16],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["c"], "momento": ["post"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    var_c = xr.concat([Pre_c, Post_c], dim="momento")

    # concatena todos los sujetos
    daTodos = xr.concat([var_a, var_b, var_c], dim="n_var")
    daTodos.name = "Angle"
    daTodos.attrs["freq"] = 1 / (
        daTodos.time[1].values - daTodos.time[0].values
    )  # incluimos la frecuencia como atributo
    daTodos.attrs["units"] = "deg"
    daTodos.time.attrs["units"] = "s"

    # Gráficas
    daTodos.plot.line(x="time", col="momento", hue="ID", row="n_var")

    # =============================================================================
    # %% Test the functions
    # =============================================================================
    """
    data = daTodos[0, 0, 0, 120:201].data
    data2 = _circularity(data)

    data2 = _detrend(data2)
    plt.plot(data2)
    data2[0]
    data2[-1]
    """
    daTodosTrozo = daTodos.isel(time=slice(100, 201))
    daTodosTrozo.plot.line(x="time", col="momento", hue="ID", row="n_var")

    daTodosTrozoCirc = create_circularity_xr(daTodosTrozo)
    daTodosTrozoCirc.plot.line(x="time", col="momento", hue="ID", row="n_var")

    daTodosTrozoCircDetrend = _detrend(daTodosTrozoCirc[0, 0, 0, :].data)
    plt.plot(daTodosTrozoCircDetrend)

    daTodosTrozoCircDetrend = detrend_xr(daTodosTrozoCirc)
    daTodosTrozoCircDetrend.plot.line(x="time", col="momento", hue="ID", row="n_var")

    data = daTodosTrozoCircDetrend[0, 0, 0, :].data
    plt.plot(data)

    # Misma forma original y reconstruido
    dataShannon = _shannon_reconstruction(data, old_freq=daTodos.freq, new_freq=400.0)
    plt.plot(np.arange(len(dataShannon)) / 400, dataShannon, label="Shannon rec")
    plt.plot(np.arange(len(data)) / daTodos.freq, data, label="Original", ls="--")
    plt.legend()
    plt.show()

    dataShannon = _shannon_reconstruction(data, old_freq=daTodos.freq, new_freq=800.0)
    plt.plot(
        np.arange(len(dataShannon[-100:])) / 400,
        dataShannon[-100:],
        label="Shannon rec",
        marker="*",
    )
    plt.plot(
        np.arange(len(data[-25:])) / daTodos.freq,
        data[-25:],
        label="Original",
        ls="--",
        marker="*",
    )
    plt.legend()
    plt.show()
