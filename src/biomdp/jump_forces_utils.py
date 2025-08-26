# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:33:55 2022

@author: josel
"""
# =============================================================================
# %% LOAD MODULES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "v.1.7.2"
__date__ = "26/08/2025"


# TODO: try detecting thresholds with scipy.stats.threshold
"""
Updates:
    26/08/2025, v.1.7.2
        - Added 'verbose' parameter in load_merge_bioware_pl
    
    29/07/2025, v.1.7.1
        - In non "DJ2P" jumps, adjusted the parameter 'n_above' to
          int(0.01 * daData.freq) in detect_takeoff_landing function, to avoid
          rebounds in the takeoff event.

    08/07/2025, v.1.7.0    
        - Included parameter 'avoid_file_name' and 'discardables' in 
          load_merge_vicon_csv_logsheet and load_merge_vicon_c3d_logsheet.
        - Adjusted adjust_offsetFz to round the split_window to the nearest integer.
        - Fixed graphs_events to not plot empty datasets.
        - graphs_events adapted to 'preacsj' test.
        
    14/06/2025, v.1.6.2
        - Fixed show plot when repe not in dims in check_flat_window
        - The graphs_events function includes filling colored areas.
        - Included discardable parameter in functions load_merge_bioware_pl,
          load_merge_bioware_pd, and load_merge_bioware_c3d.

    04/06/2025, v.1.6.1
        - Fixed graphs_weight_check when trimming, the end is daData.time.size,
          not daEvents.endAnalysis.

    22/05/2025, v.1.6.0
        - Included function to load Vicon CSV files and convert them to parquet.
        - Some types adjustments.

    08/03/2025, v.1.5.5
        - Adapted to biomdp with translations.
        - Derived from trata_fuerzas_saltos.py

    13/01/2025, v.1.5.4
        - Arreglado crea calcula_results y results_a_tabla.
    
    05/01/2025, v.1.5.3
        - Corregido graficas_chequeo_peso, no funcinaba con dataarray con 'repe'.
            COMPROBAR QUE FUNCIONA SIN 'repe'
        - En detect_ini_mov, avisa si algún evento es nan.
        - Actualizada función polars melt a unpivot.

    27/12/2024, v.1.5.2
        - AñadidaS variableS RSI y landing_stiffness.
    
    16/12/2024, v.1.5.1
        - Corregido que en ajusta_offsetFz no mantenía los atributos.
    
    15/12/2024, v.1.5.0
        - En función load_merge_vicon_c3d_selectivo incluido parámetro read_c3d_function
          para elegir la función de lectura (read_vicon_c3d o read_vicon_ezc3d_xr)

    07/12/2024, v.1.4.0
        - Calcula más variables relacionadas con la caída (FZMaxCaida, PMinCaida, etc.).
            
    19/11/2024, v.1.3.7
        - Introducida función results_a_tabla.
        - En calcula_results añadido tDespegue y tAterrizaje.
        - Corregidos valores FZMin y tFzMin para buscarlos desde iniMov a tFzMax
        - Corregido tSMax para buscar entre iniMov y aterrizaje.
        - Actualizado cumtrapz por cumulative_trapezoid en _integra
                
    17/11/2024, v.1.3.6
        - Corregido integrar y RFD al calcular resultados.
        - TODO: hacerlo con funciones de general_processing_functions o biomdp
    
    12/05/2024, v.1.3.5
        - Introducido typo return en funciones (sin probar).
    
    2/02/2024, v.1.3.4
        - Cambiada dimensión 'evento' por 'event'.
    
    21/02/2024, v.1.3.3
        - En la detección del evento despegue reducida la ventana de tener
          que estar por debajo del umbral, de 0.15 a 0.01 s.
    
    11/02/2024, v.1.3.2
        - Cambio nombre chequea_end_plano a chequea_tramo_plano.
          Mejorado: devuelve datos 'continuo' o 'discreto', con tipo_analisis 'std' o 'delta'.
          Si se piden gráficas las hace con graficas_events de la ventana final.
    
    04/02/2024, v.1.3.1
        - En afina_peso, incluido kind 'peso_media_salto'. Calcula
          el peso haciendo la media de Fz desde iniAnalisis hasta finAnalisis.
    
    22/01/2024, v.1.3.0
        - Añadidas variables de resultado tiempo de los eventos.
                  
    04/10/2023, v.1.1.4
        - Separada función de pasar de polars a dataarray. Cuando carga
          con polars devuelve xarray con todas las variables que carga.
          Después se separa en dimensión axis y plat en otra función.
              
    22/09/2023, v.1.1.3
        - Corregido que no podía dibujar eventos sin nombre convencional.
    
    18/07/2023, v.1.1.2
        - Corregido cálculo ajusta_offsetFz, con kind='vuelo' y
          jump_test distinto de DJ2P, no devolvía los datos corregidos.
    
    17/06/2023, v.1.1.1
        - En recorta ventana se asegura de que no vengan valores menores
          que cero ni mayores que len(data).
        - Para detectar iniMov en CMJ y SJ, busca superar el umbral por 
          encima o por debajo y coge el que esté antes.
        - Creación de gráficas eventos admite parámetro sharey.
        - Al calcular resultados, cambiado nombre de variables var por n_var,
          para evitar confusiones con el método .var (varianza).
        - Al detectar ini/fin imp posit, se queda con la detección última,
          la más cercana al despegue.
        - Al calcular resultados de impulsos, restaba el peso, pero la
          fuerza está calculada en BW, por lo que ahora resta 1 como peso
          y devuelve el impulso normalizado.
        - En resultados EMG calcula activación en periodo de preactivación,
          un tiempo antes del iniMov. Tiene sentido en los SJ, en CMJ sería
          activación residual antes de empezar a moverse.
    
    20/05/2023, v.1.1.0
        - Introducia función para optimizar el ajuste del peso. Probado con
          método iterativo y con curve fitting. Con polinomio de orden 8 
          funciona bien.
        - Al afinar el peso añade el valor de los residuos a la dimensión
          stat.
    
    12/05/2023, v.1.0.7
        - Corregido el hacer gráficas al calcular el peso.
        - Reparado detect_minFz para DJ y DJ2P.
        - ajusta_offsetFz_flight y reset_Fz_flight ahora se hace directamente
          con xarray, haciendo la media por debajo del umbral pedido.
     
    06/05/2023, v.1.0.6
        - Los n_above se calculan en segundos teniendo en cuenta la
          frecuencia.
        - Cuando se pide que ajuste inifin no dibuja eventos de búsqueda
          del peso.
        - Probada nueva versión de detect_end_mov basada en la velocidad,
          cuando vuelve a ser cero después del aterrizaje. Corregida
          versión basada en fuerza, cuando después del aterrizaje baja del
          peso y vuelve a subir. Parece que funciona mejor.

    16/04/2023, v.1.0.5
        - Corregido en detect_takeoff_landing que se quede siempre
          con el último que encuentra. Si el último está al final del
          registro, coge el anterior.
    
    15/03/2023, v.1.0.4
        - Al buscar despegue y aterrizaje, si no los encuentra da una
          segunda oportunidad ajustando el umbral al valor mínimo del
          registro.
        - Corrección en show gráfica en reset_Fz_flight cuando no hay repes.
        - En función gráficas, introducido parámetro show_in_console para
          controlar si se quiere que dibuje las gráficas en la consola o
          sólo en pdf.
            
    '26/02/2023', v.1.0.3
        - Incluye función para encontrar todos los eventos de una vez.
        - Incluidas funciones para detectar saltos que empiezan o terminan
          en cero. Puede ser diferente la ventana inicial y final.
        - Incluida función que valora si es equiparable una ventana inicial
          a la final.
    
    '09/01/2023', v.1.0.2
        - En la función calcula_variables devuelve la fuerza normalizada
          como otra variable del dataset y mantiene el dtype original.
    
    '22/12/2022', v.1.0.1
        - Corrección, devuelve float al buscar inicio movimiento con DJ.
    
    '10/09/2022', v.1.0.0
        - Versión inicial con varias funciones.
"""


# import sys
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.integrate as integrate
import xarray as xr
from detecta import detect_onset
from matplotlib.backends.backend_pdf import PdfPages  # to save graphs in pdf

import biomdp.biomec_xarray_accessor  # accessor biomxr

# =============================================================================
# ---- VARIABLES
# =============================================================================
g = 9.81  # m/s2

BASIC_EVENTS = [
    "iniAnalisis",
    "preactiv",
    "iniPeso",
    "finPeso",
    "iniMov",
    "maxFz",
    "minFz",
    "maxV",
    "iniImpPos",
    "maxFlex",
    "finImpPos",
    "despegue",
    "aterrizaje",
    "finMov",
    "finAnalisis",
]


# =============================================================================
# ---- SUPPORT FUNCTIONS
# =============================================================================


def create_standard_jump_events(daData: xr.DataArray) -> xr.DataArray:
    return (
        xr.full_like(daData.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": BASIC_EVENTS},
            axis=-1,
        )
    ).copy()


def assign_subcategories_xr(
    da: xr.DataArray, n_project: str | None = None
) -> xr.DataArray:
    """
    da= da.set_index(ID=['estudio', 'particip', 'tipo', 'subtipo'])
    da.sel(tipo='SJ')
    da.ID
    """
    if len(da.ID.to_series().iloc[0].split("_")) == 5:
        da = da.assign_coords(
            estudio=(
                n_project
            ),  # "ID", da.ID.to_series().str.split("_").str[0].to_list()),
            particip=("ID", da.ID.to_series().str.split("_").str[1].to_list()),
            tipo=("ID", da.ID.to_series().str.split("_").str[2].to_list()),
            subtipo=("ID", da.ID.to_series().str.split("_").str[3].to_list()),
            repe=("ID", da.ID.to_series().str.split("_").str[4].to_list()),
        )

    elif len(da.ID.to_series().iloc[0].split("_")) == 4:
        if n_project is None:
            n_project = "X"
        da = da.assign_coords(
            estudio=("ID", [n_project] * len(da.ID)),
            particip=("ID", da.ID.to_series().str.split("_").str[0].to_list()),
            tipo=("ID", da.ID.to_series().str.split("_").str[1].to_list()),
            subtipo=("ID", da.ID.to_series().str.split("_").str[2].to_list()),
            repe=("ID", da.ID.to_series().str.split("_").str[3].to_list()),
        )

    """
    #versión basada en df polars
    da = da.assign_coords(estudio=('ID', df.filter(pl.col('time')==0.000).get_column('estudio').to_list()),
                                              particip=('ID', df.filter(pl.col('time')==0.000).get_column('particip').to_list()),
                                              tipo=('ID', df.filter(pl.col('time')==0.000).get_column('tipo').to_list()),
                                              subtipo=('ID', df.filter(pl.col('time')==0.000).get_column('subtipo').to_list()),
                                              repe=('ID', df.filter(pl.col('time')==0.000).get_column('repe').to_list()),
                                              )
    """
    """
    #Versión antigua si no se sabe si hay dimensión repe o no.
    #En la versión actual no hay dimensión repe, se guarda en el ID
    est=[] #un poco complejo, pero si no todos tienen repe=1 no funcionaba bien
    tip=[]
    subtip=[]
    partic=[]
    repe=[]
    #da.ID.str.split(dim='splt', sep='_')
    for n in da.ID:
        #if len(n.str.split('_'))
        partes = n.str.split(dim='splt', sep='_')
        if len(partes)==3:
            est.append(n_project)
            partic.append(partes.data[0])
            tip.append(partes.data[1])
            subtip.append(subtipo)
            repe.append(partes.data[-1])
        elif len(partes)==4:
            est.append(n_project)
            partic.append(partes.data[0])
            tip.append(partes.data[1])
            subtip.append(partes.data[2])
            repe.append(partes.data[-1])
        elif len(partes)==5:
            est.append(partes.data[0])
            partic.append(partes.data[1])
            tip.append(partes.data[2])
            subtip.append(partes.data[3])
            repe.append(partes.data[-1])
    
    da = da.assign_coords(estudio=('ID', est), particip=('ID', partic), tipo=('ID', tip), subtipo=('ID', subtip), repe=('ID', repe))
    """
    # if 'repe' in da.dims: #solo lo añade si no tiene ya la dimensión repe
    #     da = da.assign_coords(estudio=('ID', est), particip=('ID', partic), tipo=('ID', tip), subtipo=('ID', subtip))
    # else:
    #     da = da.assign_coords(estudio=('ID', est), particip=('ID', partic), tipo=('ID', tip), subtipo=('ID', subtip), repe=('ID', repe))

    return da


# TODO: REPLACE WITH general_processing_functions.integrate_window??
def integrate_full(daData: xr.DataArray, daEvents: xr.DataArray) -> xr.DataArray:
    def _integra(data, t, ini, fin, ID):
        if np.isnan(ini) or np.isnan(fin):
            return np.nan
        ini = int(ini)
        fin = int(fin)
        # print(ID)
        # plt.plot(data[ini:fin])
        try:
            dat = integrate.cumulative_trapezoid(data[ini:fin], t[ini:fin], initial=0)[
                -1
            ]
        except:
            # print(f'Error integrating {ID}')
            dat = np.nan
        return dat

    """
    data = daData[0].data
    time = daData.time.data
    ini = daEvents[0].isel(event=0).data
    fin = daEvents[0].isel(event=1).data
    """
    daInt = xr.apply_ufunc(
        _integra,
        daData,
        daData.time,
        daEvents.isel(event=0),
        daEvents.isel(event=1),
        daData.ID,
        input_core_dims=[["time"], ["time"], [], [], []],
        # output_core_dims=[['time']],
        exclude_dims=set(("time",)),
        vectorize=True,
        # join='exact',
    )
    return daInt


# def RMS(daData, window):
#     """
#     Calcula RMS en dataarray indicando window
#     """

#     def rms(data):
#         if np.count_nonzero(~np.isnan(data)) == 0:
#             return np.array(np.nan)
#         data = data[~np.isnan(data)]
#         return np.linalg.norm(data[~np.isnan(data)]) / np.sqrt(len(data))

#     """
#     data = daRecortado[0,0,0,0].data
#     """
#     daRecortado = trim_analysis_window(daData, window)
#     daRMS = xr.apply_ufunc(
#         rms,
#         daRecortado,
#         input_core_dims=[["time"]],
#         vectorize=True,
#     )
#     return daRMS


def df_to_da(
    dfData: pl.DataFrame | pd.DataFrame,
    merge_2_plats: int = 1,
    n_project: str | None = None,
    show: bool = False,
) -> xr.DataArray:
    if isinstance(dfData, pl.DataFrame):
        # Transforma df polars a dataarray con todas las variables cargadas
        vars_leidas = dfData.select(
            pl.exclude(
                ["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
            ),
        ).columns

        dfpd = dfData.unpivot(
            index=["ID", "time"], on=vars_leidas, variable_name="n_var"
        ).to_pandas()

    else:  # viene con pandas
        vars_leidas = dfData.drop(
            columns=["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
        ).columns
        dfpd = dfData

    daAll = (
        dfpd
        # .drop(columns=["estudio", "tipo", "subtipo", "particip", "repe"])
        # .melt(id_vars=["ID", "time"], value_vars='n_var')#, value_name='value2')
        # pd.melt(dfData.to_pandas().drop(columns=['estudio','tipo','subtipo']), id_vars=['ID', 'repe', 'time'], var_name='axis')
        .set_index(["ID", "n_var", "time"])
        .to_xarray()
        .to_array()
        .squeeze("variable")
        .drop_vars("variable")
    )

    daAll = split_dim_plats(da=daAll, merge_2_plats=merge_2_plats)
    # Renombra columnas
    # dfData = dfData.rename({'abs time (s)':'time', 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    # Asigna coordenadas extra
    daAll = assign_subcategories_xr(daAll, n_project=n_project)
    # daAll = daAll.assign_coords(estudio=('ID', dfData.filter(pl.col('time')==0.000).get_column('estudio').to_list()),
    #                                          particip=('ID', dfData.filter(pl.col('time')==0.000).get_column('particip').to_list()),
    #                                          tipo=('ID', dfData.filter(pl.col('time')==0.000).get_column('tipo').to_list()),
    #                                          subtipo=('ID', dfData.filter(pl.col('time')==0.000).get_column('subtipo').to_list()),
    #                                          repe=('ID', dfData.filter(pl.col('time')==0.000).get_column('repe').to_list()),
    #                                          )
    # Ajusta tipo de coordenada tiempo, necesario??
    ###########daTodosArchivos = daTodosArchivos.assign_coords(time=('time', daTodosArchivos.time.astype('float32').values))

    # daTodosArchivos.sel(ID='PCF_SCT05', axis='z').plot.line(x='time', col='repe')
    # daTodosArchivos.assign_coords(time=daTodosArchivos.time.astype('float32'))
    daAll.attrs = {
        "freq": (np.round(1 / (daAll.time[1] - daAll.time[0]), 1)).data,
        "units": "N",
    }
    daAll.time.attrs["units"] = "s"
    daAll.name = "Forces"

    return daAll


# ----Separa dimensión repe
def split_dim_repe(daData: xr.DataArray) -> xr.DataArray:
    # Traduce desde ID a dimensión repe
    # Asume que hay 3 repeticiones, seguir probando si funciona con otras numeraciones
    # ¿Devuelve la repe con int o con str?

    # rep0 = daData.sel(ID=daData.ID.str.endswith('1'))
    rep0 = daData.where(daData.repe == daData.repe[0], drop=True)
    rep0 = rep0.assign_coords(
        ID=["_".join(s.split("_")[:-1]) for s in rep0.ID.data.tolist()]
    ).drop_vars(  # rep0.ID.str.rstrip(f'_{rep0.repe[0].data}'))
        "repe"
    )

    rep1 = daData.where(daData.repe == daData.repe[1], drop=True)
    rep1 = rep1.assign_coords(
        ID=["_".join(s.split("_")[:-1]) for s in rep1.ID.data.tolist()]
    ).drop_vars(  # rep1.ID.str.rstrip(f'_{rep1.repe[1].data}'))
        "repe"
    )

    rep2 = daData.where(daData.repe == daData.repe[2], drop=True)
    rep2 = rep2.assign_coords(
        ID=["_".join(s.split("_")[:-1]) for s in rep2.ID.data.tolist()]
    ).drop_vars(  # rep2.ID.str.rstrip(f'_{rep2.repe[2].data}'))
        "repe"
    )

    daData = xr.concat([rep0, rep1, rep2], dim="repe").assign_coords(
        repe=[1, 2, 3]
    )  # .transpose('ID', 'n_var', 'repe', 'time')

    print("CAUTION: check that the repetition numbers are correct.")

    return daData


# Versión con nombres repe originales
def split_dim_repe_logsheet(
    daData: xr.DataArray, log_sheet: pd.DataFrame | None = None
) -> xr.DataArray:
    # Traduce desde ID a dimensión repe respetando el número de la repe original de la hoja de registro
    if log_sheet is None:
        raise ValueError("You must specify the Dataframe with the log sheet")

    h_r = log_sheet.iloc[:, 1:].dropna(how="all")

    rep0 = []
    rep1 = []
    rep2 = []
    jump_tests = sorted(
        list(set(["_".join(col.split("_")[:-1]) for col in h_r.columns]))
    )
    for S in h_r.index:
        for t in jump_tests:
            for r in h_r.filter(regex=t).loc[S]:
                # print(daData.sel(ID=daData.ID.str.contains(f'{S}_{t}_{r}')).ID.data)
                da = daData.sel(ID=daData.ID.str.contains(f"{S}_{t}_{r}"))
                reps = h_r.filter(regex=t).loc[S]
                if da.size != 0:
                    reID = "_".join(
                        da.ID.data[0].split("_")[0:-1]
                    )  # ['_'.join(n.split('_')[0:-1]) for n in list(da.ID.data)]
                    print(r, reID)
                    if da.ID.str.endswith(reps.iloc[0]):  # len(da.ID) >0:
                        rep0.append(da.assign_coords(ID=[reID]))
                    elif da.ID.str.endswith(reps.iloc[1]):  # len(da.ID) >1:
                        rep1.append(da.assign_coords(ID=[reID]))
                    elif da.ID.str.endswith(reps.iloc[2]):  # len(da.ID) >2:
                        rep2.append(da.assign_coords(ID=[reID]))

    """
    rep0=[]
    rep1=[]
    rep2=[]
    for S in h_r.index:
        for t in ['CMJ_2', 'SJ_0L', 'SJ_100L', 'SJ_100S']:            
            #for r in h_r.filter(regex=t).loc[S]:
                #print(daData.sel(ID=daData.ID.str.contains(f'{S}_{t}_')).ID.data)
            da = daData.sel(ID=daData.ID.str.contains(f'{S}_{t}_'))
            reps = h_r.filter(regex=t).loc[S]
            if da.size!=0:
                reID = '_'.join(da.ID.data[0].split('_')[0:-1]) #['_'.join(n.split('_')[0:-1]) for n in list(da.ID.data)]
                if da.ID.str.endswith(reps[0]): # len(da.ID) >0:
                    rep0.append(da.isel(ID=0).assign_coords(ID=reID))
                if da.ID.str.endswith(reps[1]): # len(da.ID) >1:
                    rep1.append(da.isel(ID=1).assign_coords(ID=reID))
                if da.ID.str.endswith(reps[2]): # len(da.ID) >2:
                    rep2.append(da.isel(ID=2).assign_coords(ID=reID))
    """
    rep0 = xr.concat(rep0, dim="ID").drop_vars("repe")
    rep1 = xr.concat(rep1, dim="ID").drop_vars("repe")
    rep2 = xr.concat(rep2, dim="ID").drop_vars("repe")

    df = pd.DataFrame(rep2.ID.data)
    df.duplicated().sum()
    df[df.duplicated(keep=False)]

    daData = xr.concat([rep0, rep1, rep2], dim="repe").assign_coords(repe=[0, 1, 2])

    try:
        if (
            np.char.find(daData.n_var.astype(str).data, "EMG", start=0, end=None).mean()
            != -1
        ):  # si es EMG
            daData = daData.transpose("ID", "n_var", "repe", "time")
        else:
            daData = daData.transpose(
                "ID", "n_var", "repe", "axis", "time"
            )  # si no es EMG
    except Exception as e:
        print(f"Dimensions could not be sorted. {e}")

    return daData


def split_dim_plats(da: xr.DataArray, merge_2_plats: int) -> xr.DataArray:
    if "Fz.1" in da.n_var.data:  # pandas
        n_vars_plat2 = ["Fx.1", "Fy.1", "Fz.1"]
    else:  # polars
        n_vars_plat2 = ["Fx_duplicated_0", "Fy_duplicated_0", "Fz_duplicated_0"]

    # Separa el dataarray en plataformas creando una dimensión
    # Si hay 2 plataformas las agrupa en una
    if merge_2_plats == 0:
        plat1 = da.sel(n_var=["Fx", "Fy", "Fz"]).assign_coords(n_var=["x", "y", "z"])
        plat2 = da.sel(n_var=n_vars_plat2).assign_coords(n_var=["x", "y", "z"])
        da = (
            xr.concat([plat1, plat2], dim="plat").assign_coords(plat=[1, 2])
            # .transpose("ID", "plat", "repe", "axis", "time")
        )

    elif merge_2_plats == 1:
        da = da.sel(n_var=["Fx", "Fy", "Fz"]).assign_coords(n_var=["x", "y", "z"])

    elif merge_2_plats == 2:
        da = da.sel(n_var=n_vars_plat2).assign_coords(n_var=["x", "y", "z"])

    elif merge_2_plats == 3:
        plat1 = da.sel(n_var=["Fx", "Fy", "Fz"]).assign_coords(n_var=["x", "y", "z"])
        plat2 = da.sel(n_var=n_vars_plat2).assign_coords(n_var=["x", "y", "z"])
        da = plat1 + plat2

    da = da.rename({"n_var": "axis"})
    return da


def compute_forces_axes(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate the axis forces from the raw forces of each sensor.
    Use after read_kistler_c3d_xr()
    """

    def _split_plataforms(da):
        plat1 = da.sel(n_var=da.n_var.str.startswith("F1"))
        plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip("F1"))

        plat2 = da.sel(n_var=da.n_var.str.startswith("F2"))
        plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip("F2"))

        da = xr.concat([plat1, plat2], dim="plat").assign_coords(
            plat=["plat1", "plat2"]
        )

        return da

    if "plat" not in da.coords:
        da = _split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains("x")).sum(dim="n_var")
    Fy = da.sel(n_var=da.n_var.str.contains("y")).sum(dim="n_var")
    Fz = da.sel(n_var=da.n_var.str.contains("z")).sum(dim="n_var")

    daReturn = xr.concat([Fx, Fy, Fz], dim="axis").assign_coords(axis=["x", "y", "z"])
    # daReturn.plot.line(x='time', col='plat')

    return daReturn


def split_dim_axis(da) -> xr.DataArray:
    # Separa el xarray en ejes creando dimensión axis

    x = da.sel(n_var=da.n_var.str.contains("x")).rename({"n_var": "axis"})
    y = da.sel(n_var=da.n_var.str.contains("y")).rename({"n_var": "axis"})
    z = da.sel(n_var=da.n_var.str.contains("z")).rename({"n_var": "axis"})
    da = (
        xr.concat([x, y, z], dim="axis")
        # .assign_coords(n_var="plat1")
        # .assign_coords(axis=["x", "y", "z"])
        .expand_dims({"n_var": 1})
    )

    return da


def load_weight_bioware_csv(file: str | Path) -> float:
    weight = 0.0
    with open(file, mode="rt") as f:
        num_lin = 0

        # Scrolls through the entire file to find the start and end of the section and the number of lines
        for linea in f:
            num_lin += 1
            # Search for section start label
            if "Normalized force (N):" in linea:
                weight = float(linea.split("\t")[1])
                break
            num_lin += 1
    if weight == 0.0:
        print("Weight label not found in the file")
    return weight


# Load a Bioware file as Polars dataframe
def load_bioware_pl(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
):  # -> pl.DataFrame | xr.DataArray:
    print("Deprecated. Redirecting to biomdp.io.read_kistler_txt function")
    from biomdp.io.read_kistler_txt import read_kistler_txt_pl

    """
    df = (
        pl.read_csv(
            file,
            has_header=True,
            skip_rows=lin_header,
            skip_rows_after_header=1,
            columns=n_vars_load,
            separator="\t",
        )  # , columns=nom_vars_cargar)
        # .slice(1, None) #quita la fila de unidades (N) #no hace falta con skip_rows_after_header=1
        # .select(pl.col(n_vars_load))
        # .rename({'abs time (s)':'time'}) #'Fx':'x', 'Fy':'y', 'Fz':'z',
        #          #'Fx_duplicated_0':'x_duplicated_0', 'Fy_duplicated_0':'y_duplicated_0', 'Fz_duplicated_0':'z'
        #          })
    ).with_columns(pl.all().cast(pl.Float64()))

    # ----Transform polars to xarray
    if to_dataarray:
        x = df.select(pl.col("^*Fx$")).to_numpy()
        y = df.select(pl.col("^*Fy$")).to_numpy()
        z = df.select(pl.col("^*Fz$")).to_numpy()
        data = np.stack([x, y, z])
        freq = 1 / (df[1, "abs time (s)"] - df[0, "abs time (s)"])
        ending = -3
        coords = {
            "axis": ["x", "y", "z"],
            "time": np.arange(data.shape[1]) / freq,
            "n_var": ["Force"],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        }
        da = (
            xr.DataArray(
                data=data,
                dims=coords.keys(),
                coords=coords,
            )
            .astype(float)
            .transpose("n_var", "axis", "time")
        )
        da.name = "Forces"
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        da.attrs["units"] = "N"

        return da

    return df
    """
    return read_kistler_txt_pl(file, lin_header, n_vars_load, to_dataarray)


# Carga un archivo Bioware C3D a xarray
def read_kistler_c3d(
    file: str | Path, n_vars_load: List[str] | None = None, engine: str = "ezc3d"
) -> xr.DataArray:
    print("Deprecated. Redirecting to biomdp.io.read_kistler_c3d function")
    from biomdp.io.read_kistler_c3d import read_kistler_c3d

    # from read_vicon_c3d import read_vicon_c3d_xr, read_vicon_c3d_xr_global

    da = read_kistler_c3d(file, n_vars_load, engine=engine)

    return da


def from_vicon_csv_to_parquet(
    path: str | Path,
    sections: List[str] | None = None,
    n_vars_load: List[str] | None = None,
) -> None:
    from biomdp.io.read_vicon_csv import read_vicon_csv

    if isinstance(path, str):
        path = Path(path)

    if sections is None:
        sections = ["Trajectories", "Model Outputs", "Forces", "EMG"]

    file_list = sorted(list(path.glob("*.csv")))  # "**/*.csv"
    file_list = [x for x in file_list if "error" not in x.name]
    # file_list.sort()

    print("Loading files...")
    timer_carga = time.perf_counter()

    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    num_processed_files = 0
    for n, file in enumerate(file_list):
        print(f"Loading file: {n}/{len(file_list)} {file.name}")
        try:
            dfFull = []
            for sec in sections:
                try:
                    dfFull.append(read_vicon_csv(file, section=sec, raw=True))
                except:
                    pass
            dfProvis = pl.concat(dfFull, how="horizontal")
            dfProvis.write_parquet(file.with_suffix(".parquet"))

            # daProvis.isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
            num_processed_files += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATTENTION. Unable to process " + file.name, err, "\n")
            error_files.append(f"{file.name}  {str(err)}")
            continue

    print(
        f"Loaded {num_processed_files} files from {len(file_list)} in {time.perf_counter() - timer_carga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION: unable to load")
        for x in range(len(error_files)):
            print(error_files[x])


def load_merge_vicon_csv(
    path: str | Path,
    section: str = "Model Outputs",
    n_vars_load: List[str] | None = None,
    n_project: str | None = None,
    data_type: str | None = None,
    assign_subcat: bool = True,
    show=False,
) -> xr.DataArray:
    from biomdp.io.read_vicon_csv import read_vicon_csv

    if isinstance(path, str):
        path = Path(path)

    file_list = sorted(list(path.glob("**/*.csv")))
    file_list = [x for x in file_list if "error" not in x.name]
    # file_list.sort()

    print("Loading files...")
    timer_carga = time.perf_counter()

    daAll = []  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    num_processed_files = 0
    for n, file in enumerate(file_list):
        print(f"Loading file: {n}/{len(file_list)} {file.name}")
        try:
            daProvis = read_vicon_csv(
                file, section=section, n_vars_load=n_vars_load
            ).expand_dims(
                {"ID": ["_".join(file.stem.split("_"))]}, axis=0
            )  # Añade columna ID
            daAll.append(daProvis)
            # daProvis.isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
            num_processed_files += 1

        except Exception as err:  # Si falla anota un error y continua
            print("\nATTENTION. Unable to process " + file.name, err, "\n")
            error_files.append(f"{file.name}  {str(err)}")
            continue

    print(
        f"Loaded {num_processed_files} files from {len(file_list)} in {time.perf_counter() - timer_carga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION: unable to load")
        for x in range(len(error_files)):
            print(error_files[x])

    # Group
    daAll = xr.concat(daAll, dim="ID")
    # daAll.sel(axis='z').plot.line(x='time', col='ID', col_wrap=3)

    # Llama asignar subcategorías aquí o después en parte principal?
    if assign_subcat:
        daAll = assign_subcategories_xr(da=daAll, n_project=n_project)

    return daAll


def load_merge_vicon_csv_logsheet(
    path,
    log_sheet=None,
    section: str = "Model Outputs",
    n_vars_load=None,
    avoid_file_name=None,
    discardables: list[str] | None = None,
    n_project=None,
    data_type=None,
    show=False,
) -> xr.DataArray:
    """Load file listing based on log sheet data"""

    from biomdp.io.read_vicon_csv import read_vicon_csv

    if log_sheet is None:
        raise ValueError("You must specify the Dataframe with the log sheet")

    h_r = log_sheet.iloc[:, 1:].dropna(how="all")

    file_list = sorted(
        list(path.glob("**/*.csv"))
    )  # incluye los que haya en subcarpetas

    if avoid_file_name is not None:
        file_list = [
            f
            for f in file_list
            if not any(noun in f.as_posix() for noun in avoid_file_name)
        ]
    if discardables is not None:
        # discardables como ruta completa
        file_list = [
            s for s in file_list if not any(Path(xs) == s for xs in discardables)
        ]

    print("Loading files...")
    timer_carga = time.time()
    daAll = []
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    num_processed_files = 0
    jump_tests = list(set(["_".join(col.split("_")[:-1]) for col in h_r.columns]))

    for S in h_r.index:
        for t in jump_tests:
            for r in h_r.filter(regex=t).loc[S]:
                if r is not np.nan:
                    # file = Path((path / f"{S}_{t}_{r}").with_suffix(".csv"))
                    try:
                        file = [x for x in file_list if file_name in x.stem][0]
                    except IndexError:
                        print(f"File {file_name} not found")
                        continue
                    print(f"Loading section {section}, file: {file.name}")
                    # print(f'{S}_{t}_{r}', f'{S}_{t}_{r}' in [x.stem for x in file_list])
                    try:
                        daProvis = read_vicon_csv(
                            file, section=section, n_vars_load=n_vars_load
                        ).expand_dims(
                            {"ID": ["_".join(file.stem.split("_"))]}, axis=0
                        )  # Añade dimensión ID
                        daAll.append(daProvis)
                        # daProvis.isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
                        num_processed_files += 1
                        """
                        read_vicon_csv(
                            file, section='Forces', n_vars_load=n_vars_load
                        )
                        """

                    except Exception as err:  # Si falla anota un error y continua
                        print(
                            "\nATTENTION. Unable to process " + file.name,
                            err,
                            "\n",
                        )
                        error_files.append(f"{file.name}  {str(err)}")
                        continue

    print(f"Loaded {num_processed_files} files in {time.time() - timer_carga:.3f} s \n")

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    # Agrupa
    daAll = xr.concat(daAll, dim="ID")
    # daAll.sel(axis='z').plot.line(x='time', col='ID', col_wrap=3)

    # Llama asignar subcategorías aquí o después en parte principal?
    daAll = assign_subcategories_xr(da=daAll, n_project=n_project)

    return daAll


def load_merge_vicon_c3d_logsheet(
    path,
    log_sheet=None,
    section=None,
    n_vars_load=None,
    avoid_file_name=None,
    discardables: list[str] | None = None,
    data_type=None,
    n_project=None,
    engine="ezc3d",
    show=False,
) -> xr.DataArray:
    """
    descrive function"""
    from biomdp.io.read_vicon_c3d import read_vicon_c3d

    if log_sheet is None:
        raise ValueError("You must specify the Dataframe with the log sheet")

    h_r = log_sheet.iloc[:, 1:].dropna(how="all")
    file_list = sorted(
        list(path.glob("**/*.c3d"))
    )  # incluye los que haya en subcarpetas

    if avoid_file_name is not None:
        file_list = [
            f
            for f in file_list
            if not any(noun in f.as_posix() for noun in avoid_file_name)
        ]
    if discardables is not None:
        # discardables como ruta completa
        file_list = [
            s for s in file_list if not any(Path(xs) == s for xs in discardables)
        ]

    print("Loading files...")
    timer_carga = time.time()

    daAll = []
    error_files = []
    num_processed_files = 0
    jump_tests = list(set(["_".join(col.split("_")[:-1]) for col in h_r.columns]))

    for S in h_r.index:
        for t in jump_tests:
            for r in h_r.filter(regex=t).loc[S]:
                if r is not np.nan:
                    file_name = f"{S}_{t}_{r}"
                    try:
                        file = [x for x in file_list if file_name in x.stem][0]
                    except IndexError:
                        print(f"File {file_name} not found")
                        continue

                    print(f"Loading section {section}, file: {file.name}")
                    # print(f'{S}_{t}_{r}', f'{S}_{t}_{r}' in [x.stem for x in file_list])
                    try:
                        daProvis = read_vicon_c3d(  # read_c3d_function(
                            file,
                            section=section,
                            n_vars_load=n_vars_load,
                            engine=engine,
                        ).expand_dims(
                            {"ID": ["_".join(file.stem.split("_"))]}, axis=0
                        )  # Añade dimensión ID
                        daAll.append(daProvis)
                        """
                        dsProvis['Trajectories'].isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
                        dsProvis['Modeled'].isel(ID=0, axis=-1).plot.line(x='time')
                        dsProvis['Forces'].isel(ID=0, n_var=0, axis=-1).plot.line(x='time')
                        dsProvis['EMG'].isel(ID=0, n_var=0).plot.line(x='time')
                        """
                        num_processed_files += 1

                    except Exception as err:  # Si falla anota un error y continua
                        print(
                            "\nATTENTION. Unable to process " + file.name,
                            err,
                            "\n",
                        )
                        error_files.append(f"{file.name}  {str(err)}")
                        continue

    print(
        "Loaded {0:d} files in {1:.3f} s \n".format(
            num_processed_files, time.time() - timer_carga
        )
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    # Agrupa
    daAll = xr.concat(daAll, dim="ID")
    # daAll.sel(axis='z').plot.line(x='time', col='ID', col_wrap=3)

    daAll = assign_subcategories_xr(da=daAll, n_project=n_project)

    return daAll


def load_merge_bioware_pl(
    path: Path | str,
    n_vars_load: List[str] | None = None,
    n_project: str | None = None,
    discardables: list[str] | None = None,
    data_type: str | None = None,
    lin_header: int = 17,
    n_col_time: str = "abs time",
    merge_2_plats: int = 1,
    show: bool = False,
    verbose=True,
) -> xr.DataArray:
    """
    Parameters
    ----------
    work_path : TYPE
        DESCRIPTION.
    n_vars_load : TYPE, optional
        DESCRIPTION. The default is None.
    n_project : string, optional
        DESCRIPTION. The name of the study.
    discardable : list of strings
        DESCRIPTION. List of strings ccontaining the complete paths of the files
        to be discarded.
    data_type:
        Conversión al tipo de datos indicado. Por defecto es None, que quiere
        decir que se mantiene el tipo original ('float64')
    lin_header : TYPE, optional
        DESCRIPTION. The default is 17.
    merge_2_plats : int, optional
        What to do with more than 1 platform.
        0: keep the twoo platforms apart.
        1: only platform 1.
        2: only plataform 2.
        3: merge both paltforms as one.
        The default is 1.
    show : bool, optional
        DESCRIPTION. The default is False.
    verbose : bool, optional
        DESCRIPTION. Shows or not messages about each file reading.
        The default is True.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    daTodosArchivos : xarray DataArray
        DESCRIPTION.
    dfTodosArchivos : pandas DataFrame
        DESCRIPTION.

    Note: since BioWare Version 5.6.1.0 time column changed from "abs time (s)" to "abs time"

    """
    from biomdp.io.read_kistler_txt import read_kistler_txt

    if isinstance(path, str):
        path = Path(path)

    # if data_type is None:
    #     data_type = float

    file_list = sorted(
        list(path.glob("**/*.txt"))
    )  #'**/*.txt' incluye los que haya en subcarpetas
    file_list = [x for x in file_list if "error" not in x.name]  # selecciona archivos
    if discardables is not None:
        # discardables como ruta completa
        file_list = [
            s for s in file_list if not any(Path(xs) == s for xs in discardables)
        ]

    if n_vars_load is None:  # si no vienen impuestas las columnas a cargar
        n_vars_load = [n_col_time]  # , 'Fx', 'Fy', 'Fz']
        if merge_2_plats != 2:  # in [0,1]:
            n_vars_load += ["Fx", "Fy", "Fz"]  # ['Fx.1', 'Fy.1', 'Fz.1']
            if merge_2_plats != 1:
                n_vars_load += [
                    "Fx_duplicated_0",
                    "Fy_duplicated_0",
                    "Fz_duplicated_0",
                ]  # ['Fx.1', 'Fy.1', 'Fz.1']
        else:
            n_vars_load += [
                "Fx_duplicated_0",
                "Fy_duplicated_0",
                "Fz_duplicated_0",
            ]  # ['Fx.1', 'Fy.1', 'Fz.1']

    print("\nLoading files...")
    timerCarga = time.perf_counter()

    num_loaded_files = 0
    dfData = []  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daAll = []
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(file_list):
        if verbose:
            print(f"Loading file num. {nf}/{len(file_list)}: {file.name}")
        try:
            timerSub = time.perf_counter()

            dfProvis = read_kistler_txt(file, lin_header, n_vars_load, raw=True)

            particip = "X"
            tipo = "X"
            subtipo = "X"
            if n_project is None:
                n_project = "projectX"

            if len(file.stem.split("_")) == 5:
                n_project = file.stem.split("_")[0] if n_project is None else n_project
                particip = file.stem.split("_")[-4]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 4:
                # estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 3:
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-2]
                subtipo = "X"

            repe = str(int(file.stem.split("_")[-1]))  # int(file.stem.split('.')[0][-1]
            ID = f"{particip}_{tipo}_{subtipo}_{repe}"  # f'{n_project}_{file.stem.split("_")[0]}_{tipo}_{subtipo}'

            # freq = np.round(1/(dfProvis['time'][1]-dfProvis['time'][0]),1)

            # Añade categorías
            dfProvis = dfProvis.with_columns(
                [
                    pl.lit(n_project).alias("estudio"),
                    pl.lit(tipo).alias("tipo"),
                    pl.lit(subtipo).alias("subtipo"),
                    pl.lit(ID).alias("ID"),
                    pl.lit(particip).alias("particip"),
                    pl.lit(repe).alias("repe"),
                ]
            )  # .select(['estudio', 'tipo', 'subtipo', 'ID', 'repe'] + nom_vars_cargar)

            dfData.append(dfProvis)

            if verbose:
                print(f"{dfData[-1].shape[0]} filas y {dfData[-1].shape[1]} columnas")
                print(f"Time {time.perf_counter() - timerSub:.3f} s \n")
            num_loaded_files += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATTENTION. Unable to process {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            error_files.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    dfData = pl.concat(dfData)

    print(
        f"Loaded {num_loaded_files} files in {time.perf_counter() - timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    if data_type is not None:
        cast = pl.Float32() if data_type == "float32" else pl.Float64()
        dfData = dfData.select(
            # pl.col(['n_project', 'tipo', 'subtipo', 'ID', 'repe']),
            pl.exclude(n_vars_load),
            pl.col(n_vars_load).cast(cast),
        )
    dfData = dfData.rename({n_col_time: "time"})  # , 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    daAll = df_to_da(dfData, merge_2_plats=merge_2_plats, n_project=n_project)

    # TODO: SEGUIR HACIENDO QUE SI CARGA VARIABLES DISTINTAS DE LAS CONVENCIONALES LAS PASE A DAARRAY PERO SIN EJES
    # if  dfData.columns == ['abs time (s)', 'Fx', 'Fy', 'Fz'] or n_vars_load == ['abs time (s)', 'Fx', 'Fy', 'Fz', 'Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0']:
    #     daAll = pasa_df_a_da(dfData, merge_2_plats=merge_2_plats, show=show)

    """    
    #Transforma a pandas y a dataarray
    daAll = (dfData.drop(['estudio','tipo','subtipo'])
                .melt(id_vars=['ID', 'repe', 'time'], variable_name='axis')
                .to_pandas()
                .set_index(['ID','repe', 'axis', 'time'])
                .to_xarray().to_array()
                .squeeze('variable').drop_vars('variable')
                )
     
    
    #Si hay 2 plataformas las agrupa en una
    if merge_2_plats==0:
        plat1 = (daAll.sel(axis=['Fx', 'Fy', 'Fz'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        plat2 = (daAll.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        daAll = (xr.concat([plat1, plat2], dim='plat')
                    .assign_coords(plat = [1, 2])
                    .transpose('ID', 'plat', 'repe', 'axis', 'time')
                    )
                             
    elif merge_2_plats==1:
        daAll = (daAll.sel(axis=['Fx', 'Fy', 'Fz'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
    
    elif merge_2_plats==2:
        daAll = (daAll.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
        
                    
    elif merge_2_plats==3:
       plat1 = (daAll.sel(axis=['Fx', 'Fy', 'Fz'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       plat2 = (daAll.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       daAll = plat1 + plat2
    
    daAll = assign_subcategories_xr(da=daAll, n_project=estudio, subtipo='2PAP')
    
    daAll.name = 'Forces'
    daAll.attrs['units'] = 'N'
    daAll.attrs['freq'] = freq
    daAll.time.attrs['units'] = 's'
    """
    return daAll


def load_merge_bioware_pd(
    path,
    n_vars_load=None,
    n_project=None,
    discardables: list[str] | None = None,
    data_type=None,
    lin_header=17,
    merge_2_plats=1,
    show=False,
) -> xr.DataArray:
    """
    Parameters
    ----------
    work_path : TYPE
        DESCRIPTION.
    n_vars_load : TYPE, optional
        DESCRIPTION. The default is None.
    n_project : string, optional
        DESCRIPTION. The name of the study.
    data_type:
        Conversión al tipo de datos indicado. Por defecto es None, que quiere
        decir que se mantiene el tipo original ('float64')
    lin_header : TYPE, optional
        DESCRIPTION. The default is 17.
    merge_2_plats : int, optional
        What to do with more than 1 platform.
        0: keep the twoo platforms apart.
        1: only platform 1.
        2: only plataform 2.
        3: merge both paltforms as one.
        The default is 1.
    show : bool, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    daTodosArchivos : xarray DataArray
        DESCRIPTION.
    dfTodosArchivos : pandas DataFrame
        DESCRIPTION.

    """
    from biomdp.io.read_kistler_txt import read_kistler_txt

    # if data_type is None:
    #     data_type = float

    file_list = sorted(
        list(path.glob("**/*.txt"))
    )  #'**/*.txt' incluye los que haya en subcarpetas
    file_list = [x for x in file_list if "error" not in x.name]  # selecciona archivos
    if discardables is not None:
        # discardables como ruta completa
        file_list = [
            s for s in file_list if not any(Path(xs) == s for xs in discardables)
        ]

    if n_vars_load is None:  # si no vienen impuestas las columnas a cargar
        n_vars_load = ["abs time (s)"]  # , 'Fx', 'Fy', 'Fz']
        if merge_2_plats != 2:  # in [0,1]:
            n_vars_load += ["Fx", "Fy", "Fz"]
            if merge_2_plats != 1:
                n_vars_load += [
                    "Fx.1",
                    "Fy.1",
                    "Fz.1",
                ]
        else:
            n_vars_load += [
                "Fx.1",
                "Fy.1",
                "Fz.1",
            ]

    print("\nLoading files...")
    timerCarga = time.perf_counter()

    num_loaded_files = 0
    dfData = []  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daAll = []
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(file_list):
        print(f"Loading file num. {nf}/{len(file_list)}: {file.name}")
        try:
            timerSub = time.perf_counter()

            dfProvis = read_kistler_txt(file, lin_header, n_vars_load)

            particip = "X"
            tipo = "X"
            subtipo = "X"
            if n_project is None:
                n_project = "projectX"

            if len(file.stem.split("_")) == 5:
                n_project = file.stem.split("_")[0] if n_project is None else n_project
                particip = file.stem.split("_")[-4]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 4:
                # estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-3]
                subtipo = file.stem.split("_")[-2]
            elif len(file.stem.split("_")) == 3:
                particip = file.stem.split("_")[0]
                tipo = file.stem.split("_")[-2]
                subtipo = "X"

            repe = str(int(file.stem.split("_")[-1]))  # int(file.stem.split('.')[0][-1]
            ID = f"{particip}_{tipo}_{subtipo}_{repe}"  # f'{n_project}_{file.stem.split("_")[0]}_{tipo}_{subtipo}'

            # freq = np.round(1/(dfProvis['time'][1]-dfProvis['time'][0]),1)

            # Añade categorías
            dfProvis.insert(0, "repe", repe)
            dfProvis.insert(0, "particip", particip)
            dfProvis.insert(0, "subtipo", subtipo)
            dfProvis.insert(0, "tipo", tipo)
            dfProvis.insert(0, "ID", ID)
            dfProvis.insert(0, "estudio", n_project)

            dfData.append(dfProvis)

            print(f"{dfData[-1].shape[0]} rows and {dfData[-1].shape[1]} columns")
            print("Time {0:.3f} s \n".format(time.perf_counter() - timerSub))
            num_loaded_files += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATTENTION. Unable to process {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            error_files.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    dfData = pd.concat(dfData)

    print(
        f"Loaded {num_loaded_files} files in {time.perf_counter() - timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    if data_type is not None:
        dfData = dfData.astype(data_type)

    dfData = dfData.rename(
        columns={"abs time (s)": "time"}
    )  # , 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    daAll = df_to_da(dfData, merge_2_plats=merge_2_plats, n_project=n_project)

    # TODO: SEGUIR HACIENDO QUE SI CARGA VARIABLES DISTINTAS DE LAS CONVENCIONALES LAS PASE A DAARRAY PERO SIN EJES
    # if  dfData.columns == ['abs time (s)', 'Fx', 'Fy', 'Fz'] or n_vars_load == ['abs time (s)', 'Fx', 'Fy', 'Fz', 'Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0']:
    #     daAll = pasa_df_a_da(dfData, merge_2_plats=merge_2_plats, show=show)

    """    
    #Transforma a pandas y a dataarray
    daAll = (dfData.drop(['estudio','tipo','subtipo'])
                .melt(id_vars=['ID', 'repe', 'time'], variable_name='axis')
                .to_pandas()
                .set_index(['ID','repe', 'axis', 'time'])
                .to_xarray().to_array()
                .squeeze('variable').drop_vars('variable')
                )
     
    
    #Si hay 2 plataformas las agrupa en una
    if merge_2_plats==0:
        plat1 = (daAll.sel(axis=['Fx', 'Fy', 'Fz'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        plat2 = (daAll.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                 .assign_coords(axis=['x', 'y', 'z'])
                 )
        daAll = (xr.concat([plat1, plat2], dim='plat')
                    .assign_coords(plat = [1, 2])
                    .transpose('ID', 'plat', 'repe', 'axis', 'time')
                    )
                             
    elif merge_2_plats==1:
        daAll = (daAll.sel(axis=['Fx', 'Fy', 'Fz'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
    
    elif merge_2_plats==2:
        daAll = (daAll.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                     .assign_coords(axis=['x', 'y', 'z'])
                     )
        
                    
    elif merge_2_plats==3:
       plat1 = (daAll.sel(axis=['Fx', 'Fy', 'Fz'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       plat2 = (daAll.sel(axis=['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'])
                .assign_coords(axis=['x', 'y', 'z'])
                )
       daAll = plat1 + plat2
    
    daAll = assign_subcategories_xr(da=daAll, n_project=estudio, subtipo='2PAP')
    
    daAll.name = 'Forces'
    daAll.attrs['units'] = 'N'
    daAll.attrs['freq'] = freq
    daAll.time.attrs['units'] = 's'
    """
    return daAll


def load_merge_bioware_c3d(
    path: str | Path,
    n_vars_load: List[str] | None = None,
    n_project: str | None = None,
    discardables: list[str] | None = None,
    data_type: str | None = None,
    split_plats: bool = False,
    merge_2_plats: int = 1,
    assign_subcat: bool = True,
    show: bool = False,
) -> xr.DataArray:
    # from read_kistler_c3d import read_kistler_c3d_xr
    from biomdp.io import read_kistler_c3d as rkc3d

    if isinstance(path, str):
        path = Path(path)

    # path = Path(r'F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros\2023PotenciaDJ\S01')
    # if data_type is None:
    #     data_type = 'float64'

    file_list = sorted(
        list(path.glob("*.c3d"))
    )  #'**/*.txt' incluye los que haya en subcarpetas
    file_list = [x for x in file_list if "error" not in x.name]  # selecciona archivos
    if discardables is not None:
        # discardables como ruta completa
        file_list = [
            s for s in file_list if not any(Path(xs) == s for xs in discardables)
        ]

    """if n_vars_load is None: #si no vienen impuestas las columnas a cargar
        n_vars_load = ['abs time (s)'] #, 'Fx', 'Fy', 'Fz']
        if n_vars_load !=2: #in [0,1]:
            n_vars_load += ['Fx', 'Fy', 'Fz'] #['Fx.1', 'Fy.1', 'Fz.1']
            if n_vars_load != 1:
                n_vars_load += ['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'] #['Fx.1', 'Fy.1', 'Fz.1']
        else:
            n_vars_load += ['Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0'] #['Fx.1', 'Fy.1', 'Fz.1']
    """

    print("\nLoading files...")
    timerCarga = time.perf_counter()

    num_loaded_files = 0
    daAll = []
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(file_list):
        print(f"Loading file num. {nf + 1}/{len(file_list)}: {file.name}")
        try:
            timerSub = time.perf_counter()

            """
            #Asigna etiquetas de categorías                   
            if len(file.stem.split("_")) == 5:
                estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[-4]
                tipo = file.stem.split('_')[-3]
                subtipo = file.stem.split('_')[-2]
            elif len(file.stem.split("_")) == 4:
                #estudio = file.stem.split("_")[0]
                particip = file.stem.split("_")[0]
                tipo = file.stem.split('_')[-3]
                subtipo = file.stem.split('_')[-2]
            elif len(file.stem.split("_")) == 3:
                particip = file.stem.split("_")[0]
                tipo = file.stem.split('_')[-2]
                subtipo = 'X'
            if n_project is None:
                estudio = 'EstudioX'
            
            repe = str(int(file.stem.split('_')[-1])) #int(file.stem.split('.')[0][-1]
            ID = f'{estudio}_{particip}_{tipo}_{subtipo}_{repe}' #f'{estudio}_{file.stem.split("_")[0]}_{tipo}_{subtipo}'
            """
            daProvis = rkc3d.read_kistler_c3d(file).expand_dims(
                {"ID": ["_".join(file.stem.split("_"))]}, axis=0
            )  # Añade columna ID

            daAll.append(daProvis)

            print("Time {0:.3f} s \n".format(time.perf_counter() - timerSub))
            num_loaded_files += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATTENTION. Unable to process {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            error_files.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    daAll = xr.concat(daAll, dim="ID")

    print(
        f"Loaded {num_loaded_files} files in {time.perf_counter() - timerCarga:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    if isinstance(
        data_type, str
    ):  # si se ha definido algún tipo de datos, por defecto es 'float64'
        daAll = daAll.astype(data_type)

    if merge_2_plats == 0:  # split_plats:
        daAll = rkc3d.split_plataforms(daAll)

    # Llama asignar subcategorías aquí o después en parte principal?
    if assign_subcat:
        daAll = assign_subcategories_xr(da=daAll, n_project=n_project)

    # daAll = assign_subcategories_xr(da=daAll, n_project=estudio, subtipo='2PAP')

    daAll.name = "Forces"

    return daAll


def load_preprocessed(
    work_path: str | Path, n_preprocessed_file: str, jump_test: str
) -> xr.DataArray:
    if isinstance(work_path, str):
        work_path = Path(work_path)

    if Path((work_path / (n_preprocessed_file)).with_suffix(".nc")).is_file():
        tpo = time.time()
        daData = xr.load_dataarray(
            (work_path / (n_preprocessed_file)).with_suffix(".nc")
        ).sel(tipo=jump_test)
        print(
            "\nLoading preprocessed file ",
            n_preprocessed_file + "_Vicon.nc en {0:.3f} s.".format(time.time() - tpo),
        )
    else:
        raise Exception("Preprocessed file not found")
    return daData


# =============================================================================
# ---- FUNCTIONS FOR JUMPING FORCES PROCESSING
# =============================================================================
def fix_zero_adjustment(daData: xr.DataArray) -> xr.DataArray:
    def _adjust_zero(data, ID):
        dat = data.copy()
        try:
            ind = detect_onset(
                -dat,
                threshold=max(-dat) * 0.5,
                n_above=int(0.2 * daData.freq),
                show=False,
            )
            split_window = int((ind[0, 1] - ind[0, 0]) * 10 / 100)
            ind[0, 0] += split_window
            ind[0, 1] -= split_window
            weight = dat[ind[0, 0] : ind[0, 1]].mean()
            dat -= weight
        except:
            print(f"ATTENTION: unable to correct {ID}")
            pass

        return dat

    """
    data = daData.data#[0,0,0].data
    """

    daCortado = xr.apply_ufunc(
        _adjust_zero,
        daData,
        daData.ID,
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # join='outer'
    )  # .dropna(dim='time', how='all')
    # daCortado.attrs = daData.attrs
    # daCortado.name = daData.name
    # daCortado.sel(axis='z').plot.line(x='time', row='ID', col='axis')

    return daCortado


def check_similar_ini_end(
    daData: xr.DataArray,
    margin: float | int = 20,
    window: int | float | None = None,
    returns: str | None = None,
    show: bool = False,
) -> xr.DataArray:
    """Better than with .where for shorter records, filled with nan.
    The parameter returns can be 'subtract' (numeric value), or any other.
    With 'subtract' it returns the difference ini-end in all, regardless of the threshold;
    with anything else it returns the complete record of those that meet the margin conditions
    in the subtraction of the margin conditions on the subtraction of the ini-end window.
    window: duration of the window at the end to check. It can be passed in seconds (float)
             or in no. of frames (int).
    """
    if "freq" not in daData.attrs:
        raise ValueError("daData must have 'freq' attribute defined")

    if window is None:
        _window = [int(0.5 * daData.freq), int(0.5 * daData.freq)]
    elif isinstance(window, int):  # if int assumes num. of data
        _window = [window, window]
    elif isinstance(window, float):  # if float assumes in seconds
        _window = [int(window * daData.freq), int(window * daData.freq)]

    def _resta_ini_end_aux(data, margin, ventana0, ventana1, returns, ID):
        resta = np.nan
        if np.count_nonzero(~np.isnan(data)) == 0:
            return resta
        data = data[~np.isnan(data)]
        return data[:ventana0].mean() - data[-ventana1:].mean()

    if "axis" in daData.dims:
        daDatosZ = daData.sel(axis="z")
    else:
        daDatosZ = daData

    """
    data = daDatosZ[0].data
    """
    daResta = xr.apply_ufunc(
        _resta_ini_end_aux,
        daDatosZ,
        margin,
        _window[0],
        _window[1],
        returns,
        daData.ID,
        input_core_dims=[["time"], [], [], [], [], []],
        # output_core_dims=[['time']],
        exclude_dims=set(("time",)),
        vectorize=True,
        # join='outer'
    )

    daCorrectos = xr.where(
        abs(daResta) < margin, daDatosZ.ID, np.nan, keep_attrs=True
    ).dropna("ID")

    """
    def chequea_ini_end_aux(data, margin, window, returns, ID):
        retID = None if returns=='nombre' else np.nan
        
        if np.count_nonzero(~np.isnan(data))==0:
            return retID
        
        data = data[~np.isnan(data)]
        diferencia = data[:window].mean() - data[-window:].mean()
        if returns=='nombre':
            if abs(diferencia) < margin:
                retID = ID
        else: retID = diferencia
        return retID    
    
    daCorrectos = xr.apply_ufunc(chequea_ini_end_aux, daData, margin, window, returns, daData.ID,
                   input_core_dims=[['time'], [], [], [], [] ],
                   #output_core_dims=[['time']],
                   exclude_dims=set(('time',)),
                   vectorize=True,
                   #join='outer'
                   )
    if returns=='nombre':
        daCorrectos = daCorrectos.dropna('ID')
    """
    if show:
        if returns == "resta":
            daResta.assign_coords(ID=np.arange(len(daResta.ID))).plot.line(
                x="ID", marker="o"
            )
        else:
            do_not_comply = daDatosZ.loc[dict(ID=~daDatosZ.ID.isin(daCorrectos.ID))]
            if len(do_not_comply.ID) > 0:
                do_not_comply.plot.line(x="time", alpha=0.5, add_legend=False)
                plt.title(
                    f"Gráfica con los {len(do_not_comply)} que no cumplen el criterio"
                )
                # graficas_events(do_not_comply)
            else:
                print(r"\nTodos los registros cumplen el criterio")

    return (
        daResta
        if returns == "resta"
        else daData.loc[dict(ID=daData.ID.isin(daCorrectos.ID))]
    )


def check_flat_window(
    daData: xr.DataArray,
    daEvent: xr.DataArray,
    threshold: float | int = 30,
    allowed_window: float = 0.1,
    window: float = 0.5,
    returns: str = "discrete",
    kind: str = "std",
    show: bool = False,
) -> xr.DataArray:
    """
    #TODO: NOT CHECKED?
    Checks if the already cut record ends with stable speed values.
    window: duration of the window at the end to check in seconds
    kind: 'std'- calculate SD of the window span. Ideally = 0.0 to be horizontal.
          'delta' - calculates the vertical difference between two parts of the end (average of a window margin)
    threshold: threshold of admissible force (in std or delta)
    returns: 'discrete' (default). Returns all values of the calculation (std or delta).
             'continuous'. Returns time series only of those that meet the threshold criteria.
    """
    if kind not in ["std", "delta"]:
        raise ValueError(r"kind must be 'std' or 'delta'")
    if returns not in ["discrete", "continuous"]:
        raise ValueError(r"returns must be 'discrete' or 'continuous'")

    if "axis" in daData.dims:
        daDatosZ = daData.sel(axis="z")
    else:
        daDatosZ = daData

    if isinstance(window, float):
        window = int(window * daData.freq)

    if isinstance(allowed_window, float):
        allowed_window = int(allowed_window * daData.freq)

    daEvent = daEvent.sel(ID=daData.ID)
    daEventosVentana = (
        xr.full_like(daDatosZ.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": ["iniAnalisis", "finAnalisis"]}, axis=-1
        )
    ).copy()
    if window < 0:
        daEventosVentana.loc[dict(event="finAnalisis")] = daEvent
        daEventosVentana.loc[dict(event="iniAnalisis")] = (
            daEvent + window
        )  # resta (window negativo)
    else:
        daEventosVentana.loc[dict(event="iniAnalisis")] = daEvent
        daEventosVentana.loc[dict(event="finAnalisis")] = daEvent + window

    # Recorta a una window
    daRecortes = trim_analysis_window(daDatosZ, daEventosVentana)

    if kind == "std":
        daStd = daRecortes.std(dim="time")
        daStd.name = "SDFinal"

        daCorrectos = xr.where(
            abs(daStd) < threshold, daDatosZ.ID, np.nan, keep_attrs=True
        ).dropna("ID")

        if show:
            if returns == "discrete":
                daStd.assign_coords(ID=np.arange(len(daStd.ID))).plot.line(
                    x="ID", marker="o"
                )

            elif returns == "continuous":
                do_not_comply = daDatosZ.loc[dict(ID=~daDatosZ.ID.isin(daCorrectos.ID))]
                if len(do_not_comply.ID) > 0:
                    if "repe" in do_not_comply.dims:
                        do_not_comply.plot.line(
                            x="time",
                            col="repe",
                            alpha=0.5,
                            add_legend=False,
                            sharey=False,
                        )
                    else:
                        do_not_comply.plot.line(
                            x="time",
                            alpha=0.5,
                            add_legend=False,  # sharey=False
                        )

                    plt.suptitle(
                        f"Graph with the {len(do_not_comply)} that do not meet the criterion '{kind}' < {threshold}",
                        fontsize=10,
                    )

                    graphs_events(daRecortes.sel(ID=do_not_comply.ID), sharey=False)
                else:
                    print(r"\nAll records meet the criterion")

            # daStd.isel(ID=slice(None)).plot.line(x='time', col='ID', col_wrap=4)

        if returns == "discrete":
            print(f"Returns the {len(daStd)} values {returns}")
            return daStd
        elif returns == "continuous":
            print(f"Returned {len(daCorrectos.ID)} continuous records.")
            return daData.loc[dict(ID=daData.ID.isin(daCorrectos.ID))]

    elif kind == "delta":
        """Returns the vertical difference between two parts of the end (average of a margin window)."""

        '''def fin_plano_aux(data, allowed_window, window, returns, ID):
            resta = np.nan
            if np.count_nonzero(~np.isnan(data))==0:
                return resta
            data = data[~np.isnan(data)]
            return data[-allowed_window:].mean() - data[-window+allowed_window:].mean()
            
        
        
        """
        data = daDatosZ[0].data
        """
        daDelta = xr.apply_ufunc(fin_plano_aux, daDatosZ, allowed_window, window, returns, daData.ID,
                    input_core_dims=[['time'], [], [], [], [] ],
                    #output_core_dims=[['time']],
                    exclude_dims=set(('time',)),
                    vectorize=True,
                    #join='outer'
                    )
        '''
        daDelta = daRecortes.isel(time=slice(0, allowed_window)).mean(
            dim="time"
        ) - daRecortes.isel(time=slice(-allowed_window, None)).mean(dim="time")
        daDelta.name = "DeltaFinal"

        daCorrectos = xr.where(
            abs(daDelta) < threshold, daDatosZ.ID, np.nan, keep_attrs=True
        ).dropna("ID")

        if show:
            if returns == "discrete":
                daDelta.assign_coords(ID=np.arange(len(daDelta.ID))).plot.line(
                    x="ID", marker="o"
                )
            elif returns == "continuous":
                do_not_comply = daDatosZ.loc[dict(ID=~daDatosZ.ID.isin(daCorrectos.ID))]
                if len(do_not_comply.ID) > 0:
                    do_not_comply.plot.line(x="time", alpha=0.5, add_legend=False)
                    plt.title(
                        f"Graph with the {len(do_not_comply)} that not meet the criterion '{kind}' < {threshold}",
                        fontsize=10,
                    )
                    graphs_events(daRecortes.sel(ID=do_not_comply.ID), sharey=True)
                else:
                    print("\nAll records meet the criterion")

        if returns == "discrete":
            print(f"Returned the {len(daDelta)} values {returns}")
            return daDelta
        elif returns == "continuous":
            print(f"Returned {len(daCorrectos.ID)} records continuous.")
            return daData.loc[dict(ID=daData.ID.isin(daCorrectos.ID))]

    raise ValueError(
        r"It seems that the type of data to be returned has not been specified."
    )


def guess_iniend_analysis(
    daData: xr.DataArray,
    daEvents: xr.DataArray,
    window: List[float] | float = [1.5, 1.5],
    jump_test: str = "CMJ",
    threshold: float = 20.0,
    show: bool = False,
) -> xr.DataArray:
    """
    Attempts to estimate the start and end of the analysis from the center of the flight.
    window: time in seconds before and after takeoff and landing, respectively."""
    if not isinstance(
        window, list
    ):  # si se aporta un solo valor, considera que la mitad es para el inicio y la otra para el final
        _window = np.array([window, window])
    else:
        _window = np.array(window)

    # Transforma a escala de fotogramas
    _window = _window * daData.freq

    if jump_test == "DJ2P":
        # Busca primer aterrizaje provisional
        def detect_aterr1(data):
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            try:
                # Busca primer aterrizaje
                aterr = detect_onset(
                    -data,
                    threshold=int(-threshold),
                    n_above=int(0.05 * daData.freq),
                    show=show,
                )[0, 0]

            except Exception as excp:
                print(excp)
                aterr = 0  # por si no encuentra el criterio
            return float(aterr)

        aterr1 = xr.apply_ufunc(
            detect_aterr1,
            daData,
            input_core_dims=[["time"]],
            # output_core_dims=[['weight']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

        # Busca segundo aterrizaje provisional
        aterr2 = detect_takeoff_landing(
            daData, jump_test=jump_test, threshold=threshold
        ).sel(event="aterrizaje")

        daEvents.loc[dict(event="iniAnalisis")] = xr.where(
            (aterr1 - _window[0]) >= 0, aterr1 - _window[0], 0
        )  # .astype(int)
        daEvents.loc[dict(event="finAnalisis")] = xr.where(
            (aterr2 + _window[1]) < len(daData.time),
            aterr2 + _window[1] - 1,
            len(daData.time) - 1,
        )  # .astype(int)

    else:
        # Busca despegue y aterrizaje provisionales
        d_a = detect_takeoff_landing(daData, jump_test=jump_test, threshold=threshold)
        desp = d_a.sel(event="despegue").where(d_a.sel(event="despegue") > 0, np.nan)
        aterr = d_a.sel(event="aterrizaje").where(
            d_a.sel(event="aterrizaje") < len(daData.time) - 1, np.nan
        )

        daEvents.loc[dict(event="iniAnalisis")] = xr.where(
            desp.notnull(),
            xr.where((desp - _window[0]) >= 0, desp - _window[0], 0),
            np.nan,
        )
        daEvents.loc[dict(event="finAnalisis")] = xr.where(
            desp.notnull(),
            xr.where(
                (aterr + _window[1] - 1) < len(daData.time),
                aterr + _window[1] - 1,
                len(daData.time) - 1,
            ),
            np.nan,
        )

        """
        centro_flight = detect_takeoff_landing(daData, jump_test=jump_test, threshold=threshold).mean('evento')
            
        daEvents.loc[dict(event='iniAnalisis')] = xr.where((centro_flight - window[0]) >= 0, centro_flight - window[0], 0)#.astype(int)
        daEvents.loc[dict(event='finAnalisis')] = xr.where((centro_flight + window[1]-1) < len(daData.time), centro_flight + window[1]-1, len(daData.time)-1)#.astype(int)
        """
    return daEvents


def calculate_weight(
    daData: xr.DataArray, weight_window: xr.DataArray, show: bool = False
) -> xr.DataArray:
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    # Con ventana de peso única para todos
    # daWeight = daData.isel(time=slice(window[0], window[1])).mean(dim='time')

    # Con ventanas personalizadas
    def _weight_indiv_xSD(data, vent0, vent1):
        try:
            vent0 = int(vent0)
            vent1 = int(vent1)
            weight = []
            weight.append(data[vent0:vent1].mean())
            weight.append(data[vent0:vent1].std())
        except:
            return np.asarray([np.nan, np.nan])
        # plt.plot(data[vent0:vent1])
        # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)
        return np.asarray(weight)

    """
    data = daData[0].data
    vent0 = weight_window.sel(event='iniPeso')[0].data
    vent1 = weight_window.sel(event='finPeso')[0].data
    """
    daWeight = xr.apply_ufunc(
        _weight_indiv_xSD,
        daData,
        weight_window.isel(event=0),
        weight_window.isel(
            event=1
        ),  # .sel(event='iniPeso'), weight_window.sel(event='finPeso'),
        input_core_dims=[["time"], [], []],
        output_core_dims=[["stat"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    ).assign_coords(stat=["media", "sd"])

    if show:

        def dibuja_peso(x, y, **kwargs):  # de momento no funciona
            print(x)  # kwargs['data'])
            # plt.plot()

        g = daData.plot.line(col="ID", col_wrap=3, hue="repe", alpha=0.8, sharey=False)
        # g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
        # g.map_dataarray(dibuja_peso, x='time', y=None)#, y='trial')
        col = ["C0", "C1", "C2"]
        for h, ax in enumerate(g.axs):  # extrae cada fila
            for i in range(len(ax)):  # extrae cada axis (gráfica)
                if (
                    g.name_dicts[h, i] is None
                ):  # en los cuadros finales que sobran de la cuadrícula se sale
                    break
                try:
                    idn = g.data.loc[g.name_dicts[h, i]].ID
                    # print('weight=', daWeight.sel(ID=idn).data)#idn)
                    # Rango medida peso
                    # ax[i].axvspan(g.data.time[int(window[0]*self.datos.freq)], g.data.time[int(window[1]*self.datos.freq)], alpha=0.2, color='C1')
                    for j in daData.repe:
                        ax[i].axvspan(
                            weight_window.sel(ID=idn, repe=j).isel(event=0)
                            / daData.freq,
                            weight_window.sel(ID=idn, repe=j).isel(event=1)
                            / daData.freq,
                            alpha=0.2,
                            color=col[j.data - 1],
                        )
                        ax[i].axhline(
                            daWeight.sel(ID=idn, repe=j, stat="media").data,
                            color=col[j.data - 1],
                            lw=1,
                            ls="--",
                            alpha=0.6,
                        )
                    # Líneas peso
                    # ax[i].hlines(daWeight.sel(ID=idn, stat='media').data, xmin=daData.time[0], xmax=daData.time[-1], colors=col, lw=1, ls='--', alpha=0.6)
                except Exception as e:
                    print(
                        f"Error drawing weight in {g.name_dicts[h, i]}, {h}, {i}. {e}"
                    )
    return daWeight


def finetune_end(
    daData: xr.DataArray,
    daEvents: xr.DataArray,
    daWeight: xr.DataArray,
    window: float = 0.2,
    margin: float = 0.005,
    kind: str = "opt",
    show: bool = False,
) -> xr.DataArray:
    """
    kind can be 'opt', 'iter', 'iter_gradiente' o 'iter_final'
    """
    if kind not in ["opt", "iter", "iter_gradiente", "iter_final"]:
        raise ValueError("kind must be 'opt', 'iter', 'iter_gradiente' or 'iter_final'")

    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    # TODO: USAR FUNCIÓN GENERAL DE INTEGRAR
    def _integra(data, t, weight):
        dat = np.full(len(data), np.nan)
        try:
            dat = integrate.cumulative_trapezoid(data - weight, t, initial=0)
        except:
            pass  # dat = np.full(len(data), np.nan)
        return dat

    if isinstance(window, float):
        window = window * daData.freq

    # Recorta a una ventana desde el final
    daEvents.loc[dict(event="iniAnalisis")] = daEvents.sel(event="finAnalisis") - window
    daDatosCort = trim_analysis_window(daData, daEvents)

    if show:
        daDatosCort.isel(ID=slice(None)).plot.line(x="time", col="ID", col_wrap=4)

    daDatosCort.std(dim="time")

    def _peso_iter_gradiente(data, t, weight, ini, fin, margin, ID):  # , rep):
        """
        Va ajustando el peso teniendo en cuenta la diferncia con la iteración
        anterior, hasta que la diferencia es menor que un nº mínimo.
        """
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(weight)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            pes = weight - 100
            delta_peso = 1
            tend_pes = []
            v = np.full(len(data), 20.0)
            for i in range(1000):
                v0 = _integra(data, t, pes) / (pes / 9.8)
                v1 = _integra(data, t, pes + delta_peso) / (pes + delta_peso / 9.8)
                pes += v0[-1] - v1[-1]  # *1
                tend_pes.append(pes)
                if i > 2 and pes - tend_pes[-2] < 0.00001:
                    break
            # plt.plot(tend_pes)

        except Exception as e:
            print(f"No se encontró. {e}")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(v0, lw=0.2, label="ajustado")
            v = _integra(data, t, weight) / (weight / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.title("Cálculo iterativo descenso gradiente")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, weight - pes])

    def _peso_iter_final(data, t, weight, ini, fin, margin, ID):  # , rep):
        # Ajuste para saltos con preactivación. Devuelve el peso ajustado y la diferencia entre el peso anterior y el ajustado
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            # primera pasada más gruesa
            itera = 0
            pes = 300
            v = np.arange(len(data))  # np.full(len(data), 20.0)
            # while not -3 < v[-1] < 3 and pes < weight+100:
            while (
                v[int(-0.2 * daData.freq) : int(-0.1 * daData.freq)].mean()
                - v[int(-0.5 * daData.freq) : int(-0.4 * daData.freq)].mean()
                > 0.05
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                pes += 5.0
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

            # Segunda pasada más fina
            itera = 0
            pes = pes - 4
            v = _integra(data, t, pes) / (pes / 9.8)
            while (
                v[int(-0.1 * daData.freq) : int(-0.05 * daData.freq)].mean()
                - v[int(-0.3 * daData.freq) : int(-0.25 * daData.freq)].mean()
                > margin
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                # plt.plot(len(v)-int(0.2*daData.freq), v[int(-0.2*daData.freq)], 'o')
                # plt.plot(len(v)-int(0.1*daData.freq), v[int(-0.1*daData.freq)], 'o')
                # plt.plot(len(v)-int(.6*daData.freq), v[int(-.6*daData.freq)], 'o')
                # plt.plot(len(v)-int(.5*daData.freq), v[int(-.5*daData.freq)], 'o')

                pes += 0.01
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

        except Exception as e:
            print(f"No se encontró. {e}")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(v, lw=0.2, label="ajustado")
            v = _integra(data, t, weight) / (weight / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.title("Cálculo iterativo")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, weight - pes])

    """
    #Con repe
    data = daData[1,0].data
    t = daData.time.data
    weight = daWeight.sel(stat='media')[1,0].data
    evIni = 'despegue'
    evFin = 'finAnalisis'
    ini = daEvents.sel(event=evIni)[1,0].data
    fin = daEvents.sel(event=evFin)[1,0].data

    #Sin repe
    data = daData[1].data
    t = daData.time.data
    weight = daWeight.sel(stat='media')[1].data
    evIni = 'despegue'
    evFin = 'finAnalisis'
    ini = daEvents.sel(event=evIni)[1].data
    fin = daEvents.sel(event=evFin)[1].data
    """
    if kind == "iter_gradiente":
        f_calculo = _peso_iter_gradiente
        evIni = "iniMov"
        evFin = "finMov"
    elif kind == "iter_final":
        f_calculo = _peso_iter_final
        evIni = "despegue"
        evFin = "finAnalisis"
    else:
        raise ValueError(f"Calculation method '{kind}' not implemented")

    daPesoReturn = xr.apply_ufunc(
        f_calculo,
        daData,
        daData.time,
        daWeight.sel(stat="media"),
        daEvents.sel(event=evIni),
        daEvents.sel(event=evFin),
        margin,
        daData.ID,  # daData.repe,
        input_core_dims=[["time"], ["time"], [], [], [], [], []],  # , []],
        output_core_dims=[["stat"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    ).assign_coords(stat=["media", "resid"])

    if daWeight is not None:
        daPesoReturn = xr.concat([daPesoReturn, daWeight.sel(stat="sd")], dim="stat")
        daPesoReturn.loc[dict(stat="sd")] = daWeight.sel(stat="sd")

    return daPesoReturn


def finetune_weight(
    daData: xr.DataArray,
    daEvents: xr.DataArray,
    daWeight: xr.DataArray,
    margin: float = 0.005,
    kind: str = "opt",
    show: bool = False,
) -> xr.DataArray:
    """
    kind: can be 'opt', 'iter', 'iter_gradiente', 'iter_final', 'peso_media_salto'
    """

    if kind not in [
        "opt",
        "iter",
        "iter_gradiente",
        "iter_final",
        "peso_media_salto",
        "'peso_media_salto'",
    ]:
        raise ValueError(
            "kind must be 'opt', 'iter', 'iter_gradiente', 'iter_final' or 'peso_media_salto'"
        )

    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    def _integra(data, t, weight):
        dat = np.full(len(data), np.nan)
        try:
            dat = integrate.cumulative_trapezoid(data - weight, t, initial=0)
        except:
            pass  # dat = np.full(len(data), np.nan)
        return dat

    def _plot(data, v0, t, weight, pes, title):
        plt.plot(v0, lw=1, alpha=0.9, label="ajustado")
        v = _integra(data, t, weight) / (weight / 9.8)
        plt.plot(v, lw=0.5, ls="--", label="raw")
        plt.text(
            0.02,
            0.9,
            f"delta with mean weight={weight - pes:.3f}",
            horizontalalignment="left",
            fontsize="small",
            color="r",
            transform=plt.gca().transAxes,
        )
        plt.title(title)
        plt.legend()
        plt.show()
        # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

    def _optimiza_peso(data, t, weight, ini, fin, margin, ID):  # , rep):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])
        try:
            # plt.plot(data)
            # plt.axhline(weight)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin]

            vy = []
            p = []
            for pes in range(
                200, 2000, 50
            ):  # calcula velocidad entre pesos extremos cada x datos
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v)
                vy.append(v[-1])  # v[-int(0.1*daData.freq):].mean())
                p.append(pes)

            # Con polinomio
            coefs, (resid, _, _, _) = np.polynomial.polynomial.Polynomial.fit(
                vy, p, deg=8, full=True
            )
            coefs = coefs.convert().coef  # convierte a escala real
            f = np.polynomial.Polynomial(coefs)
            pes = f(0)  # weight cuando la velocidad al final es cero
            """
            vy2 = np.arange(-10, 20)
            px = f(vy2) 
            plt.plot(p,vy,'o-')
            plt.plot(px,vy2)
            plt.show()
            """

            """
            #Con otras funciones
            from scipy.optimize import curve_fit
            def f(x,a,b,c):
                return a*np.exp(-b*x)+c
            popt, pcov = curve_fit(f, vy, p)
            weight = f(0, popt[0], popt[1], popt[2]) #weight cuando la velocidad al final es cero
            
            # def f(x, qi, exp, di):
            #     return qi*(1+exp*di*x)**(-1/exp)
            # popt, pcov = curve_fit(f, vy, p,  factor=1)
                                   
            
            vy2 = np.arange(-10, 10)
            px = f(vy2, popt[0], popt[1], popt[2]) 
            plt.plot(vy,p,'--')
            plt.plot(px,vy2)
            plt.show()
            """

        except Exception as e:
            print(f"No se encontró. {e}")
            return np.asarray([np.nan, np.nan])

        if show:
            v = _integra(data, t, pes) / (pes / 9.8)
            plt.plot(v, lw=1, alpha=0.9, label="ajustado")
            v = _integra(data, t, weight) / (weight / 9.8)
            plt.plot(v, lw=0.5, ls="--", label="raw")

            plt.text(
                0.02,
                0.9,
                f"delta with mean weight={weight - pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo optimizado polinomio ({ID})")
            plt.legend()
            plt.show()

        return np.asarray([pes, resid[0]])

    def _peso_iter(data, t, weight, ini, fin, margin, ID):  # , rep):
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(weight)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            # primera pasada más gruesa
            itera = 0
            pes = 300  # weight-100
            v = np.full(len(data), 20.0)
            # while not -3 < v[-1] < 3 and pes < weight+100:
            while v[-1] > 0.5 and pes < 1300:
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v)
                pes += 5
                itera += 1
                # print('iters=', itera, 'peso=', pes, 'v=', v[-1])

            # Segunda pasada más fina
            itera = 0
            # while not -margin < v[-1] < margin and pes < weight+5:
            while v[-1] > margin and pes < weight + 50:
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v)
                pes += 0.05
                itera += 1
                # print('iters=', itera, 'weight=', pes, 'v=', v[-1])
        except Exception as e:
            print(f"No se encontró. {e}")
            return np.asarray([np.nan, np.nan])

        if show:
            _plot(
                data,
                v,
                t,
                weight,
                pes,
                title=f"Cálculo iterativo ({ID})",
            )
            """plt.plot(v, lw=0.2, label="ajustado")
            v = _integra(data, t, weight) / (weight / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.text(
                0.02,
                0.9,
                f"delta with mean weight={weight-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo iterativo ({ID})")
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)
            """
        return np.asarray([pes, weight - pes])

    def _peso_iter_gradiente(data, t, weight, ini, fin, margin, ID):  # , rep):
        """
        Va ajustando el peso teniendo en cuenta la diferencia con la iteración
        anterior, hasta que la diferencia es menor que un nº mínimo.
        """
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(weight)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            pes = weight - 100
            delta_peso = 1
            tend_pes = []
            v = np.full(len(data), 20.0)
            v0 = v
            for i in range(1000):
                v0 = _integra(data, t, pes) / (pes / 9.8)
                v1 = _integra(data, t, pes + delta_peso) / (pes + delta_peso / 9.8)
                pes += v0[-1] - v1[-1]  # *1
                tend_pes.append(pes)
                if i > 2 and pes - tend_pes[-2] < 0.00001:
                    break
            # plt.plot(tend_pes)

        except Exception as e:
            print(f"No se encontró. {e}")
            return np.asarray([np.nan, np.nan])

        if show:
            _plot(
                data,
                v0,
                t,
                weight,
                pes,
                title=f"Cálculo iterativo descenso gradiente ({ID})",
            )
            """plt.plot(v0, lw=1, alpha=0.9, label="ajustado")
            v = _integra(data, t, weight) / (weight / 9.8)
            plt.plot(v, lw=0.5, ls="--", label="raw")
            plt.text(
                0.02,
                0.9,
                f"delta con weight media={weight-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo iterativo descenso gradiente ({ID})")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)
            """
        return np.asarray([pes, weight - pes])

    def _peso_iter_final(data, t, weight, ini, fin, margin, ID):  # , rep):
        # Ajuste para saltos con preactivación. Devuelve el peso ajustado y la diferencia entre el peso anterior y el ajustado
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])

        try:
            # plt.plot(data)
            # plt.axhline(weight)
            ini = int(ini)
            fin = int(fin)
            data = data[ini:fin]
            t = t[ini:fin] - t[ini]

            # primera pasada más gruesa
            itera = 0
            pes = 300
            v = np.arange(len(data))  # np.full(len(data), 20.0)
            # while not -3 < v[-1] < 3 and pes < weight+100:
            while (
                v[int(-0.2 * daData.freq) : int(-0.1 * daData.freq)].mean()
                - v[int(-0.5 * daData.freq) : int(-0.4 * daData.freq)].mean()
                > 0.05
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                pes += 5.0
                itera += 1
                # print('iters=', itera, 'weight=', pes, 'v=', v[-1])

            # Segunda pasada más fina
            itera = 0
            pes = pes - 4
            v = _integra(data, t, pes) / (pes / 9.8)
            while (
                v[int(-0.1 * daData.freq) : int(-0.05 * daData.freq)].mean()
                - v[int(-0.3 * daData.freq) : int(-0.25 * daData.freq)].mean()
                > margin
                and pes < 1300
            ):
                v = _integra(data, t, pes) / (pes / 9.8)
                # plt.plot(v, lw=0.2)
                # plt.plot(len(v)-int(0.2*daData.freq), v[int(-0.2*daData.freq)], 'o')
                # plt.plot(len(v)-int(0.1*daData.freq), v[int(-0.1*daData.freq)], 'o')
                # plt.plot(len(v)-int(.6*daData.freq), v[int(-.6*daData.freq)], 'o')
                # plt.plot(len(v)-int(.5*daData.freq), v[int(-.5*daData.freq)], 'o')

                pes += 0.01
                itera += 1
                # print('iters=', itera, 'weight=', pes, 'v=', v[-1])

        except Exception as e:
            print(f"No se encontró. {e}")
            return np.asarray([np.nan, np.nan])

        if show:
            _plot(
                data,
                v,
                t,
                weight,
                pes,
                title=f"Cálculo iterativo final ({ID})",
            )
            """plt.plot(v, lw=0.2, label="ajustado")
            v = _integra(data, t, weight) / (weight / 9.8)
            plt.plot(v, lw=0.2, label="raw")
            plt.text(
                0.02,
                0.9,
                f"delta con weight media={weight-pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo iterativo final ({ID})")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)
            """

        return np.asarray([pes, weight - pes])

    def _peso_media_salto(data, t, weight, ini, fin, margin, ID):  # , rep):
        # Ajuste para saltos con preactivación. Devuelve el peso ajustado y la diferencia entre el peso anterior y el ajustado
        # print(ID, rep)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.asarray([np.nan, np.nan])
        try:
            # plt.plot(data)
            # plt.axhline(peso)
            ini = int(ini)
            fin = int(fin)
            pes = data[ini:fin].mean()

        except Exception as e:
            print(f"Error calculating mean weight. {e}")
            return np.asarray([np.nan, np.nan])

        if show:
            plt.plot(data[ini:fin], lw=0.5, label="Fuerza")
            plt.axhline(pes, color="b", ls="-", lw=1, label="weight media total")
            plt.axhline(
                weight,
                color="r",
                alpha=0.7,
                ls="--",
                lw=0.5,
                label="Mean weight in section",
            )
            plt.text(
                0.02,
                0.9,
                f"delta with mean weight={weight - pes:.3f}",
                horizontalalignment="left",
                fontsize="small",
                color="r",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Cálculo media salto ({ID})")
            plt.legend()
            plt.show()
            # plt.axhline(data[vent0:vent1].mean(), ls='--', lw=0.5)

        return np.asarray([pes, weight - pes])

    """
    #Con repe
    data = daData[0,0].data
    t = daData.time.data
    weight = daWeight.sel(stat='media')[0,0].data
    evIni = 'iniMov' #'despegue'
    evFin = 'finMov' #'finAnalisis'
    ini = daEvents.sel(event=evIni)[0,0].data
    fin = daEvents.sel(event=evFin)[0,0].data

    #Sin repe
    data = daData[0].data
    t = daData.time.data
    weight = daWeight.sel(stat='media')[0].data
    evIni = 'despegue'
    evFin = 'finAnalisis'
    ini = daEvents.sel(event=evIni)[0].data
    fin = daEvents.sel(event=evFin)[0].data
    """
    if kind == "opt":
        f_calculo = _optimiza_peso
        evIni = "iniMov"
        evFin = "finMov"
    elif kind == "iter":
        f_calculo = _peso_iter
        evIni = "iniMov"
        evFin = "finMov"
    elif kind == "iter_gradiente":
        f_calculo = _peso_iter_gradiente
        evIni = "iniMov"
        evFin = "finMov"
    elif kind == "iter_final":
        f_calculo = _peso_iter_final
        evIni = "despegue"
        evFin = "finAnalisis"
    elif kind == "peso_media_salto":
        f_calculo = _peso_media_salto
        evIni = "iniAnalisis"
        evFin = "finAnalisis"
    else:
        raise Exception(f"Calculation method '{kind}' not implemented")

    daPesoReturn = xr.apply_ufunc(
        f_calculo,
        daData,
        daData.time,
        daWeight.sel(stat="media"),
        daEvents.sel(event=evIni),
        daEvents.sel(event=evFin),
        margin,
        daData.ID,  # daData.repe,
        input_core_dims=[["time"], ["time"], [], [], [], [], []],  # , []],
        output_core_dims=[["stat"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    ).assign_coords(stat=["media", "resid"])

    if daWeight is not None:
        daPesoReturn = xr.concat([daPesoReturn, daWeight.sel(stat="sd")], dim="stat")
        daPesoReturn.loc[dict(stat="sd")] = daWeight.sel(stat="sd")

    return daPesoReturn


def detect_takeoff_landing(
    daData: xr.DataArray,
    jump_test: str | None,
    threshold: float = 10.0,
    show: bool = False,
) -> xr.DataArray:
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    if jump_test == "DJ2P":

        def _detect_onset_aux(data, coords, **args_func_cortes):
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.array([np.nan, np.nan])
            # plt.plot(data)
            # plt.show()
            # print(ID, repe)
            ind = detect_onset(-data, **args_func_cortes)

            if len(ind) < 2:
                # Segunda oportunidad buscando a partir del mínimo
                args_func_cortes["threshold"] -= data.min()
                ind = detect_onset(-data, **args_func_cortes)
                print("Trheshold corrected", coords)
                if len(ind) < 1:
                    print(
                        "No two takeoffs/landings found on file",
                        coords,
                    )
                    plt.plot(data)
                    return np.array([np.nan, np.nan])
            # if jump_test == 'CMJ':
            #     ind=ind[0] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
            # elif jump_test in ['DJ', 'DJ2P']:
            #     ind=ind[1] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral

            # Chequea si ha detectado más de un vuelo
            if len(ind) >= 2:
                ind = ind[-1]  # por defecto se queda con el último

                # TODO: mejorar la comprobación cuando detecta más de un vuelo
                # if ind[-1,-1] < int(len(data) * 0.8) or :
                #     ind=ind[-1] #Independientemente del tipo de salto que sea, se queda con el último que encuentre
                # else:
                #     ind=ind[-2]

            return ind.astype(float)  # [1]

        """
        data = daData.sel(ID='S07_DJ_30', repe=1).data
        data = daData[0].data
        args_func_cortes = dict(threshold=-threshold, n_above=50, show=True)
        """

        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daData.ID,
            input_core_dims=[["time"], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(
                threshold=-threshold, n_above=int(0.1 * daData.freq), show=show
            ),
        ).assign_coords(event=["despegue", "aterrizaje"])

    else:

        def _detect_onset_aux(data, coords, **args_func_cortes):
            if np.count_nonzero(~np.isnan(data)) == 0:  # or data.sum()==0.0:
                return np.array([np.nan, np.nan])
            # plt.plot(data)
            # plt.show()
            # print(ID, repe)
            ind = detect_onset(-data, **args_func_cortes)
            if len(ind) < 1:
                # Segunda oportunidad buscando a partir del mínimo
                args_func_cortes["threshold"] -= data.min()
                ind = detect_onset(-data, **args_func_cortes)
                print("Threshold corrected", coords)
                if len(ind) < 1:
                    print("Two takeoffs/landings not found on file", coords)
                    return np.array([np.nan, np.nan])
            # if jump_test == 'CMJ':
            #     ind=ind[0] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
            # elif jump_test in ['DJ', 'DJ2P']:
            #     ind=ind[1] #coge el primer bloque que encuentra
            #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral

            # Chequea si ha detectado más de un vuelo
            if len(ind) > 1:
                ind = (
                    ind[1] if jump_test == "DJ" else ind[0]
                )  # por defecto se queda con el primero o el segundo
                # TODO: mejorar la comprobación cuando detecta más de un vuelo
                # if ind[-1,-1] < int(len(data) * 0.8) or :
                #     ind=ind[-1] #Independientemente del tipo de salto que sea, se queda con el último que encuentre
                # else:
                #     ind=ind[-2]

            return ind.astype(float)

        """
        data = daData.sel(ID='25-S10_CMJ_2_001').data
        data = daData[0,0].data
        args_func_cortes = dict(threshold=-threshold, n_above=50, show=True)
        """

        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daData.ID,
            input_core_dims=[["time"], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(
                threshold=-threshold, n_above=int(0.01 * daData.freq), show=show
            ),
        ).assign_coords(event=["despegue", "aterrizaje"])
    # Comprobaciones
    # daData.sel(axis='z').isel(time=daCorte.sel(event='despegue')-1) #despegue cuando ya ha pasado por debajo del umbral
    # daData.sel(axis='z').isel(time=daCorte.sel(event='aterrizaje')-1) #aterrizaje cuando ya ha pasado por debajo del umbral
    return daCorte


def detect_takeoff_landing_cusum(
    daData: xr.DataArray, jump_test: str, threshold: float = 10.0, show: bool = False
) -> xr.DataArray:
    """Test to detect several events at the same time, but it seems very irregular"""

    def _detect_onset_aux(data, **args_func_cortes):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.array([np.nan, np.nan])
        # plt.plot(data)
        # plt.show()
        # print(ID, repe)
        from detecta import detect_cusum

        detect_cusum(data, threshold=500, drift=1, ending=True, show=True)
        ind = []
        # ind = detect_onset(-data, **args_func_cortes)
        if len(ind) < 1:
            print("No se ha encontrado despegue/aterrizaje en archivo")
            return np.array([np.nan, np.nan])
        # if jump_test == 'CMJ':
        #     ind=ind[0] #coge el primer bloque que encuentra
        #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral
        # elif jump_test in ['DJ', 'DJ2P']:
        #     ind=ind[1] #coge el primer bloque que encuentra
        #     ind[1]+=1 #para que el aterrizaje coincida con pasado umbral

        # Independientemente del tipo de salto que sea, se queda con el último que en cuentre
        ind = ind[-1]

        return ind.astype("float")  # [1]

    """
    data = daData[0,-1].data
    args_func_cortes = dict(threshold=-threshold, n_above=50, show=True)
    """
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    daCorte = xr.apply_ufunc(
        _detect_onset_aux,
        daData,
        input_core_dims=[["time"]],
        output_core_dims=[["event"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
        kwargs=dict(threshold=-threshold, n_above=int(0.2 * daData.freq), show=show),
    ).assign_coords(event=["despegue", "aterrizaje"])
    # Comprobaciones
    # daData.sel(axis='z').isel(time=daCorte.sel(event='despegue')-1) #despegue cuando ya ha pasado por debajo del umbral
    # daData.sel(axis='z').isel(time=daCorte.sel(event='aterrizaje')-1) #aterrizaje cuando ya ha pasado por debajo del umbral
    return daCorte


def detect_ini_mov(
    daData: xr.DataArray,
    daWeight: xr.DataArray,
    daEvents: xr.DataArray,
    jump_test: str = "CMJ",
    SDx: int = 5,
    threshold: float = 10.0,
    show: bool = False,
) -> xr.DataArray:
    """
    Include better methods from:
        Movement Onset Detection Methods: A Comparison Using Force Plate Recordings
        April 2023 Journal of Applied Biomechanics 39(5):1-6
        DOI: 10.1123/jab.2022-0111
        The aim of this study was to compare the 5xSD threshold method, three
        variations of the reverse scanning method and five variations of the first
        derivative method against manually selected onsets, in the counter-movement
        jump (CMJ) and squat.
    """
    # Aquí SDx es el % del peso
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    # corrector_freq = daData.freq_ref if 'freq_ref' in daData.attrs else 200.0

    # Función común para buscar umbral por encima o por debajo
    def _detect_iniMov_peso_mayor_menor(
        data, weight, weight_threshold, iinianalisis, idespegue, win_up, win_down, ID
    ):
        # Intenta detectar qué es antes: descenso por debajo del peso o ascenso por encima (dando saltito)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan
        # ini_abajo = 0
        # ini_arriba = 0
        try:
            # print(ID)
            iinianalisis = int(iinianalisis)
            idespegue = int(idespegue)
        except Exception as e:
            print(f"No hay evento iniAnalisis o despegue. {e}")
            return np.nan

        dat = data[iinianalisis:idespegue]
        # plt.plot(data[iinianalisis:idespegue])
        # Pasada inicial para ver cuándo baja por debajo del umbral peso+XSD
        ini1 = detect_onset(
            -dat,
            threshold=-(weight - weight_threshold),
            n_above=int(win_down * daData.freq),
            threshold2=-(weight - weight_threshold) * 1.2,
            n_above2=int(win_down * daData.freq * 0.5),
            show=show,
        )
        if ini1.size != 0:
            # Pasada hacia atrás buscando ajuste fino que supera el peso
            ini2 = detect_onset(
                dat[ini1[0, 0] : 0 : -1], threshold=weight, n_above=1, show=show
            )
            if ini2.size == 0:
                ini2 = np.array([[0, 0]])  # si no encuentra, suma cero
            elif ini2[0, 0] / daData.freq > 0.2:
                ini2[0, 0] = int(
                    0.2 * daData.freq
                )  # si la detección atrás se va muy lejos, retrocede un valor arbitrario de segundos
            ini_abajo = (
                iinianalisis + ini1[0, 0] - ini2[0, 0] + 1
            )  # +1 para coger el que ya ha pasado por debajo del peso
        else:
            ini_abajo = len(data)  # por si no encuentra el criterio

        # except:
        #     ini_abajo = len(data) #por si no encuentra el criterio

        # try:
        # Pasada inicial para ver cuándo baja por debajo del threshold peso+XSD
        ini1 = detect_onset(
            dat,
            threshold=(weight + threshold),
            n_above=int(win_up * daData.freq),
            show=show,
        )
        if ini1.size != 0:
            # Pasada hacia atrás buscando ajuste fino que supera el weight
            ini2 = detect_onset(
                -dat[ini1[0, 0] : 0 : -1], threshold=-weight, n_above=1, show=show
            )
            if ini2.size == 0:
                ini2 = np.array([[0, 0]])  # si no encuentra, suma cero
            elif ini2[0, 0] / daData.freq > 0.2:
                ini2[0, 0] = int(
                    0.2 * daData.freq
                )  # si la detección atrás se va muy lejos, retrocede un valor arbitrario de segundos
            ini_arriba = (
                iinianalisis + ini1[0, 0] - ini2[0, 0] + 1
            )  # +1 para coger el que ya ha pasado por encima del peso
        else:
            ini_arriba = len(data)  # por si no encuentra el criterio
        # except:
        #     ini_arriba = len(data) #por si no encuentra el criterio

        if ini_arriba == len(data) and ini_abajo == len(data):
            idx = 0
        else:
            idx = np.min([ini_arriba, ini_abajo])

        return float(idx)

    if jump_test == "DJ":

        def _detect_onset_aux(data, **args_func_cortes):
            # plt.plot(data)
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            ini = detect_onset(-data, **args_func_cortes)[0]
            return float(
                ini[1] + 1
            )  # +1 para que se quede con el que ya ha pasado el umbral

        # data= daData[0,0,2].data
        # args_func_cortes = dict(threshold=-10.0, n_above=50, show=False)
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            input_core_dims=[["time"]],
            # output_core_dims=[['weight']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(
                threshold=-threshold, n_above=int(0.2 * daData.freq), show=show
            ),
        )
        # daData.sel(axis='z').isel(time=daCorte-1)

    elif jump_test == "CMJ":
        """
        def detect_iniMov_peso_pcto(data, weight, weight_threshold, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral weight+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(weight-weight_threshold), n_above=int(0.2*daData.freq), show=show)
                #Pasada hacia atrás buscando ajuste fino que supera el weight
                ini2 = detect_onset(data[ini1[-1,0]:0:-1], threshold=(weight-weight_threshold*0.5), n_above=int(0.02*daData.freq), show=show)

                ini = ini1[-1,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por debajo del peso
                #data[ini] #peso
            except:
                ini = 0 #por si no encuentra el criterio
            return float(ini)
        """
        """
        data = daData[3,2].data
        plt.plot(data)
        weight = daWeight[3,2].sel(stat='media').data
        sdpeso = (daWeight[3,2].sel(stat='sd')*SDx).data
        weight_threshold = (daWeight[3,2].sel(stat='sd')*SDx).data
        iinianalisis = daEvents[3,2].sel(event='iniAnalisis').data
        idespegue = daEvents[3,2].sel(event='despegue').data
        ID = daEvents[3,2].ID.data
        """
        func_detect = _detect_iniMov_peso_mayor_menor  # detect_iniMov_peso_pcto
        daCorte = xr.apply_ufunc(
            func_detect,
            daData,
            daWeight.sel(ID=daData.ID, stat="media"),
            daWeight.sel(ID=daData.ID, stat="sd") * SDx,
            daEvents.sel(ID=daData.ID).sel(event="iniAnalisis"),
            daEvents.sel(ID=daData.ID).sel(event="despegue"),
            0.1,
            0.05,
            daData.ID,
            input_core_dims=[["time"], [], [], [], [], [], [], []],
            # output_core_dims=[['weight']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

    elif jump_test == "SJ":
        """
        def detect_iniMov_peso_pcto(data, weight, weight_threshold, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan

            try:
                #Pasada inicial para ver cuándo supera el umbral weight+XSD
                ini1 = detect_onset(data[:int(idespegue)], threshold=(weight+weight_threshold), n_above=int(0.2*daData.freq), show=show)

                #Pasada hacia atrás buscando ajuste fino que quede por debajo del weight
                ini2 = detect_onset(-data[ini1[-1,0]:0:-1], threshold=-(weight+weight_threshold*0.5), n_above=int(0.01*daData.freq), show=show)

                ini = ini1[-1,0] - ini2[0,0] + 1 #+1 para coger el que ya ha pasado por encima del peso
                #data[ini] #peso
            except:
                ini = 0 #por si no encuentra el criterio
            return float(ini)
        """
        """
        data = daData[6,2].data
        plt.plot(data)        
        weight = daWeight[6,2].sel(stat='media').data
        sdpeso = (daWeight[6,2].sel(stat='sd')*SDx).data
        weight_threshold =  (daWeight[6,2].sel(stat='sd')*SDx).data
        iinianalisis = daEvents[6,2].sel(event='iniAnalisis').data
        idespegue = daEvents[6,2].sel(event='despegue').data
        win_down = 0.05
        win_up = 0.1
        """
        func_detect = _detect_iniMov_peso_mayor_menor  # detect_iniMov_peso_pcto
        daCorte = xr.apply_ufunc(
            func_detect,
            daData,
            daWeight.sel(ID=daData.ID, stat="media"),
            daWeight.sel(ID=daData.ID, stat="media") * SDx / 100,
            daEvents.sel(ID=daData.ID).sel(event="iniAnalisis"),
            daEvents.sel(ID=daData.ID).sel(event="despegue"),
            0.05,
            0.1,  # ventanas down y up, en segundos
            daData.ID,
            input_core_dims=[["time"], [], [], [], [], [], [], []],
            # output_core_dims=[['weight']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

    elif jump_test == "SJPreac":
        # Como no se observa un patrón estable, coge un tiempo prefijado antes del despegue, basado en la media de fase concéntrica en SJ (#((daEventosForces.sel(event='finImpPos') - daEventosForces.sel(event='iniImpPos'))/daSJ.freq).mean())
        daCorte = daEvents.sel(event="despegue") - 0.25 * daData.freq

        """
        from detecta import detect_cusum        
        def detect_iniMov_peso_pcto(data, weight, weight_threshold, idespegue):
            if np.count_nonzero(~np.isnan(data))==0:
                return np.nan
            try:
                _, ini, _, _ = detect_cusum(data[:int(idespegue)], threshold=80, drift=1, ending=True, show=show)
                ini = ini[0]
                
            except:
                ini = 0 #por si no encuentra el criterio
                
            return float(ini)
        
        """
        """
        data = daData[0,0].data
        weight = daWeight[0,0].sel(stat='media').data
        pcto = 10
        sdpeso = (daWeight[0,0].sel(stat='sd')*SDx).data
        threshold = (daWeight[0].sel(stat='sd')*SDx).data
        idespegue = daEvents[0,0].sel(event='despegue').data
        """
        """
        daCorte = xr.apply_ufunc(detect_iniMov_peso_pcto, daData, daWeight.sel(ID=daData.ID, stat='media'), daWeight.sel(ID=daData.ID, stat='media')*SDx/100, daEvents.sel(ID=daData.ID).sel(event='despegue'),
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['weight']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """

    elif jump_test == "DJ2P":

        def _detect_iniMov_peso_pcto(data, weight, pcto_peso, idespegue):
            # Rastrea una ventana del peso (entre peso*(100-SDx)/100 a peso*(100+SDx)/100)
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            try:
                idespegue = int(idespegue)
                # Busca primer aterrizaje
                aterr = detect_onset(
                    -data[:idespegue],
                    threshold=-threshold,
                    n_above=int(0.05 * daData.freq),
                    show=show,
                )[0, 0]
                # Escanea desde el peso hacia abajo en porcentajes
                for pct in range(0, -pcto_peso, -1):
                    ini = detect_onset(
                        data[aterr::-1],
                        threshold=weight * (100 - pct) / 100,
                        n_above=int(0.05 * daData.freq),
                        show=show,
                    )  #
                    # print(idx[0,1] - idx[0,0])
                    if len(ini) > 1:
                        if ini[0, 1] - ini[0, 0] < 0.3 * daData.freq:
                            break
                    else:
                        continue

                ini = aterr - ini[0, 1] if len(ini) > 1 else aterr - ini[0, 0]

            except:
                ini = 0  # por si no encuentra el criterio
            return float(ini)

        """
        data = daData[0,0].data        
        weight = daWeight[0,0].sel(stat='media').data
        pcto = 10
        sdpeso = (daWeight[0,0].sel(stat='sd')*SDx).data
        weight_threshold = daWeight[0,0].sel(stat='media').data*SDx/100
        idespegue = daEvents[0,0].sel(event='despegue').data
        
        data = daData.sel(ID='S02_DJ_30', repe=1).data
        weight = daWeight.sel(ID='S02_DJ_30', repe=1, stat='media').data
        weight_threshold = daWeight.sel(ID='S02_DJ_30', repe=1).sel(stat='media').data*SDx/100
        idespegue = daEvents.sel(ID='S02_DJ_30', repe=1).sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_iniMov_peso_pcto,
            daData,
            daWeight.sel(stat="media"),
            SDx,
            daEvents.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            # output_core_dims=[['weight']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

        """
        def detect_iniMov_peso_XSD(data, weight, sdpeso, idespegue):
            #Parte del despegue hacia atrás buscando cuándo supera el umbral del peso - sd*SDx
            #ini = detect_onset(-data[:int(idespegue):-1], threshold=-threshold, n_above=50, show=True)[0]
            #ini = detect_onset(data[int(idespegue)::-1], threshold=threshold, n_above=5, show=True)
            #ini = idespegue - ini[1,0] + 1 #+1 para coger el que ya ha superado el umbral
            
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral weight+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(weight-sdpeso), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el weight
                ini2 = detect_onset(data[ini1[0,0]:ini1[0,0]-100:-1], threshold=weight, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha superado el umbral
                
            except:
                ini = 0 #por si no encuentra el criterio
            return ini
        daCorte = xr.apply_ufunc(detect_iniMov_peso_XSD, daData.sel(axis='z'), daWeight.sel(stat='media'), daWeight.sel(stat='sd')*SDx, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['weight']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """

        # Comprobaciones
        # daData.sel(axis='z').isel(time=daCorte-1) #

    # Si hay datos, ajusta el límite al inicio análisis
    daCorte = xr.where(
        daCorte.notnull(),
        daCorte.where(
            daCorte > daEvents.sel(event="iniAnalisis"),
            daEvents.sel(event="iniAnalisis"),
        ),
        np.nan,
    )

    return daCorte


def detect_end_mov(
    daData: xr.DataArray,
    daWeight: xr.DataArray | None = None,
    daEvents: xr.DataArray | None = None,
    jump_test: str = "CMJ",
    kind: str = "force",
    SDx: int = 2,
    show: bool = False,
) -> xr.DataArray:
    """
    kind can be 'velocity', 'force' or 'flat_window'
    """
    if kind not in ["velocity", "force", "flat_window"]:
        raise ValueError(r"kind must be 'velocity', 'force' or 'flat_window'")

    # #Con umbral velocidad cero. No funciona bien cuando la velocidad se queda al final por encima de cero
    # def detect_onset_aux(data, threshold, iaterrizaje, ID):
    #     # print(ID)
    #     # plt.plot(data)
    #     if np.count_nonzero(~np.isnan(data))==0:
    #         return np.nan

    #     try:
    #         fin = detect_onset(data[int(iaterrizaje):], threshold=0.0, n_above=int(0.1*daData.freq), show=show)
    #         fin = iaterrizaje + fin[0,1] + 1 #+1 para coger el que ya ha superado el umbral
    #                            #fin[0,0] o fin[1,1] ?????
    #     except:
    #         fin = len(data[~np.isnan(data)]) #por si no encuentra el criterio
    #     return float(fin)

    # TODO: IMPLEMENT 'flat_window', THAT LOOKS FOR WHEN A WINDOW HAS SD LESS THAN THRESHOLD

    # Calcula la diferencia en velocidad dato a dato y detecta cuándo está por encima de umbral pequeño
    def _detect_onset_dif_v(data, threshold, sd, iaterrizaje, finAn, ID):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        try:
            # plt.plot(data[int(fin):int(iaterrizaje):-1])
            dif = np.diff(data[int(finAn) : int(iaterrizaje) : -1])
            fin = detect_onset(
                dif, threshold=0.00005, n_above=int(0.05 * daData.freq), show=show
            )
            fin = finAn - fin[0, 0] + 1  # +1 para coger el que ya ha superado el umbral
        except Exception as e:
            print(e)
            fin = finAn  # len(data[~np.isnan(data)]) #por si no encuentra el criterio

        return float(fin)

    def _detect_onset_fuerza(data, threshold, sd, iaterrizaje, finAn, ID):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        try:
            fin = detect_onset(
                -data[int(finAn) : int(iaterrizaje) : -1],
                threshold=-threshold + sd * SDx,
                n_above=int(0.1 * daData.freq),
                show=show,
            )
            fin = (
                finAn - fin[-1, 0] + 1
            )  # +1 para coger el que ya ha superado el umbral
            # fin[0,0] o fin[1,1] ?????
        except Exception as e:
            print(e)
            fin = finAn  # len(data[~np.isnan(data)]) #por si no encuentra el criterio
        return float(fin)

    if kind == "velocity":
        f_calculo = _detect_onset_dif_v
        datos = calculate_variables(
            daData,
            daWeight=daWeight,
            daEvents=daEvents.sel(event=["iniMov", "finAnalisis"]),
        )["v"]
    elif kind == "force":
        f_calculo = _detect_onset_fuerza
        datos = daData
    else:
        raise Exception(f"Calculation method '{kind}' not implemented")

    # datos.plot.line(x='time', col='ID', col_wrap=3)

    """
    #data = daData[0,0].data
    data = datos[0,0].data
    iaterrizaje = daEvents.sel(event='aterrizaje')[0,0].data
    finAn = daEvents.sel(event='finAnalisis')[0,0].data
    threshold = daWeight.sel(stat='media')[0,0].data
    sd = daWeight.sel(stat='sd')[0,0].data
    """
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    daCorte = xr.apply_ufunc(
        f_calculo,
        datos,
        daWeight.sel(stat="media"),
        daWeight.sel(stat="sd"),
        daEvents.sel(event="aterrizaje"),
        daEvents.sel(event="finAnalisis"),
        daData.ID,
        input_core_dims=[["time"], [], [], [], [], []],
        # output_core_dims=[['weight']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(show=show)
    )

    # Ajusta el límite al inicio análisis
    daCorte = xr.where(
        daCorte.notnull(),
        daCorte.where(
            daCorte < daEvents.sel(event="finAnalisis"),
            daEvents.sel(event="finAnalisis"),
        ),
        np.nan,
    )

    return daCorte


# Encuentra fin cuando fuerza baja del peso y vuelve a subir
def detect_end_mov_conventional(
    daData: xr.DataArray,
    daWeight: xr.DataArray = None,
    daEvents: xr.DataArray = None,
    jump_test: str = "CMJ",
    SDx=2,
) -> xr.DataArray:
    def _detect_onset_aux(data, threshold, sd, iaterrizaje, ID):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan
        fin = detect_onset(
            -data[int(iaterrizaje) :],
            threshold=-threshold + sd * SDx,
            n_above=int(0.1 * daData.freq),
            show=False,
        )
        try:
            fin = (
                iaterrizaje + fin[0, 1] + 1
            )  # +1 para coger el que ya ha superado el umbral
            # fin[0,0] o fin[1,1] ?????
        except:
            fin = len(data[~np.isnan(data)])  # por si no encuentra el criterio
        return float(fin)

    """    
    data = daData[0,0].data
    threshold = daWeight.sel(stat='media')[0,0].data
    sd =  daWeight.sel(stat='sd')[0,0].data
    iaterrizaje = daEvents.sel(event='aterrizaje')[0,0].data
    """
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")
    daCorte = xr.apply_ufunc(
        _detect_onset_aux,
        daData,
        daWeight.sel(stat="media"),
        daWeight.sel(stat="sd"),
        daEvents.sel(event="aterrizaje"),
        daData.ID,
        input_core_dims=[["time"], [], [], [], []],
        # output_core_dims=[['weight']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )

    # Ajusta el límite al inicio análisis
    daCorte = xr.where(
        daCorte.notnull(),
        daCorte.where(
            daCorte < daEvents.sel(event="finAnalisis"),
            daEvents.sel(event="finAnalisis"),
        ),
        np.nan,
    )

    return daCorte


def detect_maxFz(
    daData: xr.DataArray,
    daWeight: xr.DataArray = None,
    daEvents: xr.DataArray = None,
    jump_test: str = "CMJ",
) -> xr.DataArray:
    # from detecta import detect_peaks
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    if jump_test in ["SJ", "SJPreac", "CMJ", "DJ", "DJ2P"]:

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ind = float(
                    np.argmax(data[ini:fin]) + ini
                )  # con -1 es el anterior a superar el umbral de 0. Coincide mejor con maxFz???
                # plt.plot(data[ini:fin])
                # plt.show()
                # detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return np.array([ind])

        """      
        data = daData[0,1,-1].data
        ini = daEvents[0,1].sel(event='iniMov').data
        fin = daEvents[0,1].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daEvents.sel(event="iniMov").data,
            daEvents.sel(event="despegue").data,
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daWeight.sel(stat='media'), n_above=50, show=False)
        )  # .assign_coords(event=['minFz'])

        return daCorte


def detect_minFz(
    daData: xr.DataArray,
    daWeight: xr.DataArray | None = None,
    daEvents: xr.DataArray | None = None,
    jump_test: str = "CMJ",
    threshold: float = 10.0,
    show: bool = False,
) -> xr.DataArray:
    # from detecta import detect_peaks
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    if jump_test == "CMJ":

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                if ini >= fin:  # puede pasar en SJ bien hechos
                    ind = ini
                else:
                    ind = float(np.argmin(data[ini:fin]) + ini)
                    # plt.plot(data[ini:fin])
                    # plt.show()
                # detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except Exception as e:
                ind = np.nan  # por si no encuentra el criterio
            return np.array([ind])

        """       
        data = daData[0].data
        ini = daEvents[0].sel(event='iniMov').data
        fin = daEvents[0].sel(event='maxFlex').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daEvents.sel(event="iniMov").data,
            daEvents.sel(event="maxFlex").data,
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daWeight.sel(stat='media'), n_above=50, show=False)
        )  # .assign_coords(event=['minFz'])

    elif jump_test in ["SJ", "SJPreac"]:

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                if ini >= fin:  # puede pasar en SJ bien hechos
                    ind = ini
                else:
                    ind = np.argmin(data[ini:fin]) + ini
                    # plt.plot(data[ini:fin])
                    # plt.plot(int(ind-ini), data[int(ind)] ,'o')
                    # plt.show()
                # detect_peaks(data[ini:fin], valley=True, mpd=100, show=True)
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
                # return np.nan
            return np.array([ind]).astype("float")

        """       
        data = daData[0,0].data
        ini = daEvents[0,0].sel(event='iniMov').data
        fin = daEvents[0,0].sel(event='maxFz').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daEvents.sel(event="iniMov").data,
            daEvents.sel(event="maxFz").data,
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daWeight.sel(stat='media'), n_above=50, show=False)
        )  # .assign_coords(event=['minFz'])

    elif jump_test in ["DJ", "DJ2P"]:

        def _detect_onset_aux(data, fin, **args_func_cortes):
            # plt.plot(data)
            if np.count_nonzero(~np.isnan(data)) == 0:
                return np.nan
            try:
                fin = int(fin)
                data = data[fin::-1]
                ind = detect_onset(-data, **args_func_cortes)[0, 0]
                ind = float(fin - ind)
            except:
                ind = np.nan  # por si no encuentra el criterio
            return ind  # +1 para que se quede con el que ya ha pasado el umbral

        """
        data= daData[0,1].data
        ini = daEvents[0,1].sel(event='iniMov').data
        fin = daEvents[0,1].sel(event='iniImpPos').data
        args_func_cortes = dict(threshold=-threshold, n_above=int(0.1*daData.freq), show=True)
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daEvents.sel(event="iniImpPos").data,
            input_core_dims=[["time"], []],
            # output_core_dims=[['evento']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            kwargs=dict(
                threshold=-threshold, n_above=int(0.1 * daData.freq), show=show
            ),
        )

    return daCorte


def detect_ini_end_impulse(
    daData: xr.DataArray,
    daWeight: xr.DataArray = None,
    daEvents: xr.DataArray = None,
    jump_test: str = "CMJ",
    show: bool = False,
) -> xr.DataArray:
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    if jump_test in ["SJ", "SJPreac", "CMJ", "DJ"]:

        def _detect_onset_aux(data, weight, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ini1 = detect_onset(
                    data[ini:fin],
                    threshold=weight,
                    n_above=int(0.1 * daData.freq),
                    show=show,
                )
                ind = (
                    ini + ini1[-1]
                )  # se queda con el último paso por el peso, próximo al despegue
                ind[1] += 1  # +1 para coger el que ya ha pasado por debajo del peso

                # #Evaluando si hay más de un evento
                # ind = ini + ini1[0] #si sólo ha detectado una subida y bajada
                # if len(ini1) > 1: #si ha detectado más de una subida y bajada
                #     ind[1] = ini + ini1[-1,-1]
                # ind[1] += 1 #+1 para coger el que ya ha pasado por debajo del peso

                # data[ini[1]+1] #peso
            except Exception as e:
                ind = np.array([np.nan, np.nan])  # por si no encuentra el criterio
            return ind.astype("float")

        """
        data = daData[5,0].data
        weight = daWeight[5,0].sel(stat='media').data
        ini = daEvents[5,0].sel(event='iniMov').data
        fin = daEvents[5,0].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daWeight.sel(stat="media").data,
            daEvents.sel(event="iniMov"),
            daEvents.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daWeight.sel(stat='media'), n_above=50, show=False)
        ).assign_coords(event=["iniImpPos", "finImpPos"])

    elif jump_test in ["DJ2P"]:

        def _detect_onset_aux(data, weight, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)

                # busca cuándo inicia primer despegue
                ini0 = detect_onset(
                    data, threshold=30.0, n_above=int(0.1 * daData.freq), show=False
                )[1, 0]
                ini1 = detect_onset(
                    data[ini0:fin],
                    threshold=weight,
                    n_above=int(0.1 * daData.freq),
                    show=False,
                )
                ind = ini0 + ini1[0]
                ind[1] += 1  # +1 para coger el que ya ha pasado por debajo del peso
                # data[ind[0]-1] #peso
            except Exception as e:
                ind = np.array([np.nan, np.nan])  # por si no encuentra el criterio
            return ind.astype("float")

        """
        data = daData[1,0].data
        weight = daWeight[1,0].sel(stat='media').data
        ini = daEvents[1,0].sel(event='iniMov').data
        fin = daEvents[1,0].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            daWeight.sel(stat="media").data,
            daEvents.sel(event="iniMov"),
            daEvents.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            output_core_dims=[["event"]],
            # exclude_dims=set(('evento',)),
            vectorize=True,
            # kwargs=dict(threshold=daWeight.sel(stat='media'), n_above=50, show=False)
        ).assign_coords(event=["iniImpPos", "finImpPos"])

    return daCorte


def detect_max_flex(
    daData: xr.DataArray,
    daWeight: xr.DataArray = None,
    daEvents: xr.DataArray = None,
    jump_test: str = "CMJ",
    v=None,
) -> xr.DataArray:
    """
    To calculate from 'DJ' it has to come reversed and with speed as a parameter.
    parameter.
    """
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    if jump_test in ["SJ", "SJPreac", "CMJ", "DJ", "DJ2P"]:

        def _detect_onset_aux(data, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ind = detect_onset(
                    data[ini:fin],
                    threshold=0,
                    n_above=int(0.01 * daData.freq),
                    show=False,
                )  # los datos que llegan de velocidad están cortados desde el iniMov
                ind = ind[0, 0] + ini
                # data[ind-5:ind+5] #data[ind]
                ind = float(ind)
                # data[ind-1] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return np.array(ind)

        # Calcula la velocidad, OJO, sin haber hecho el ajuste de offsetFz
        # TODO: CALCULATE SPEED FROM EXTERNAL FUNCTION
        if not isinstance(v, xr.DataArray):
            v = calculate_variables(
                daData,
                daWeight=daWeight,
                daEvents=daEvents.sel(event=["iniMov", "finMov"]),
            )["v"]
        # v = v.sel(axis='z') #se queda solo con axis z la haya calculado aquí o venga calculada del reversed
        # v.plot.line(x='time', col='ID', col_wrap=5, sharey=False)
        """
        data = v[0,:].data
        weight = daWeight[0].sel(stat='media').data
        ini = daEvents[0].sel(event='iniMov').data
        fin = daEvents[0].sel(event='despegue').data
        """
        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            v,
            daEvents.sel(event="iniImpPos"),
            daEvents.sel(event="despegue"),
            input_core_dims=[["time"], [], []],
            # output_core_dims=[['weight']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )

    return daCorte


def detect_max_flex_fromV(
    daData: xr.DataArray,
    daWeight: xr.DataArray = None,
    daEvents: xr.DataArray = None,
    jump_test: str = "CMJ",
) -> xr.DataArray:
    if jump_test == "DJ":
        return

    elif jump_test == "CMJ":

        def _detect_onset_aux(data, weight, ini, fin):
            try:
                ini = int(ini)
                fin = int(fin)
                ind = detect_onset(
                    data[ini:],
                    threshold=weight,
                    n_above=int(0.01 * daData.freq),
                    show=False,
                )
                # ind += ini
                # data[int(ind)-1:int(ind)+2] #data[ind]
            except:
                ind = np.nan  # por si no encuentra el criterio
            return ind

        # data = daData[0,1,-1].data
        # ini = daEvents[0,1].sel(event='iniMov').data
        # fin = daEvents[0,1].sel(event='despegue').data

        daCorte = xr.apply_ufunc(
            _detect_onset_aux,
            daData.sel(axis="z"),
            daWeight.sel(stat="media").data,
            daEvents.sel(event="iniMov"),
            daEvents.sel(event="despegue"),
            input_core_dims=[["time"], [], [], []],
            # output_core_dims=[['weight']],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=10, n_above=50, show=False)
        )
        """
        def detect_iniMov_peso_XSD(data, weight, sdpeso, idespegue):
            #Parte del despegue hacia atrás buscando cuándo supera el umbral del peso - sd*SDx
            #ini = detect_onset(-data[:int(idespegue):-1], threshold=-threshold, n_above=50, show=True)[0]
            #ini = detect_onset(data[int(idespegue)::-1], threshold=threshold, n_above=5, show=True)
            #ini = idespegue - ini[1,0] + 1 #+1 para coger el que ya ha superado el umbral
            
            try:
                #Pasada inicial para ver cuándo baja por debajo del umbral weight+XSD
                ini1 = detect_onset(-data[:int(idespegue)], threshold=-(weight-sdpeso), n_above=50, show=False)
                #Pasada hacia atrás buscando ajuste fino que supera el weight
                ini2 = detect_onset(data[ini1[0,0]:ini1[0,0]-100:-1], threshold=weight, n_above=5, show=False)
            
                ini = ini1[0,0] - ini2[0,0] + 1 #+1 para coger el que ya ha superado el umbral
                
            except:
                ini = 0 #por si no encuentra el criterio
            return ini
        daCorte = xr.apply_ufunc(detect_iniMov_peso_XSD, daData.sel(axis='z'), daWeight.sel(stat='media'), daWeight.sel(stat='sd')*SDx, daDespegue,
                                   input_core_dims=[['time'], [], [], []],
                                   #output_core_dims=[['weight']],
                                   #exclude_dims=set(('time',)),
                                   vectorize=True,
                                   #kwargs=dict(threshold=10, n_above=50, show=False)
                                   )
        """

        # Comprobaciones
        # daData.sel(axis='z').isel(time=daCorte-1) #

    return daCorte


def detect_ini_zero(daData: xr.DataArray, threshold: float = 100) -> xr.DataArray:
    """Detects the end part if the platform has been exited prematurely.
    Files already detected as having the wrong end should be passed to it.
    """
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")
        # print("Adjusted")

    def _detect_ini_zero_aux(data, threshold):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        try:
            ind = detect_onset(
                data, threshold=threshold, n_above=int(0.1 * daData.freq), show=False
            )
            ind = ind[0, 0]  # +1 para coger el que ya ha superado el umbral
        except:
            ind = len(data)  # por si no encuentra el criterio. Poner nan?
        return float(ind)

    """
    data = daData[0].data    
    """
    da = xr.apply_ufunc(
        _detect_ini_zero_aux,
        daData,
        threshold,
        input_core_dims=[["time"], []],
        # output_core_dims=[['weight']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )
    return da


def detect_final_zero(daData: xr.DataArray, threshold: float = 100) -> xr.DataArray:
    """Detects the end part if the platform has been exited prematurely.
    Files already detected as having the wrong end should be passed to it.
    """
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")

    def _detect_final_zero_aux(data, threshold):
        # print(ID)
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan
        try:
            ind = detect_onset(
                data, threshold=threshold, n_above=int(0.1 * daData.freq), show=False
            )
            ind = ind[-1, 1]
        except:
            ind = len(data)  # por si no encuentra el criterio. Poner nan?
        return float(ind)

    da = xr.apply_ufunc(
        _detect_final_zero_aux,
        daData,
        threshold,
        input_core_dims=[["time"], []],
        # output_core_dims=[['weight']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )
    return da


def detect_standard_events(
    daData: xr.DataArray,
    daWeight: xr.DataArray | None = None,
    daEvents: xr.DataArray | None = None,
    jump_test: str = "CMJ",
    kind_end_mov: str = "force",
    threshold: float = 30.0,
    SDx: int = 5,
) -> xr.DataArray:
    """
    kind_end_mov can be 'velocity', 'force' or 'flat_window'
    """
    if daWeight is None:
        raise Exception("You have not entered the weight data")

    if daEvents is None:
        daEvents = create_standard_jump_events(daData)

    # Despegue y aterrizaje definitivo
    daEvents.loc[dict(event=["despegue", "aterrizaje"])] = detect_takeoff_landing(
        daData, jump_test=jump_test, threshold=threshold
    )  # , events=daEventosCMJ)

    # Inicio movimiento, después de detectar el despegue
    daEvents.loc[dict(event="iniMov")] = detect_ini_mov(
        daData,
        jump_test=jump_test,
        daWeight=daWeight,
        daEvents=daEvents,
        threshold=threshold,
        SDx=SDx,
    )  # .sel(event='despegue'))

    # Final del movimiento
    daEvents.loc[dict(event="finMov")] = detect_end_mov(
        daData,
        jump_test=jump_test,
        daWeight=daWeight,
        daEvents=daEvents,
        kind=kind_end_mov,
    )  # .sel(event='aterrizaje'))

    # Ini y fin del impulso positivo
    daEvents.loc[dict(event=["iniImpPos", "finImpPos"])] = detect_ini_end_impulse(
        daData, jump_test=jump_test, daWeight=daWeight, daEvents=daEvents
    )  # .sel(event='iniMov'))

    # Maxima flexión rodillas batida
    if jump_test not in ["DJ", "SJPreac"]:
        daEvents.loc[dict(event="maxFlex")] = detect_max_flex(
            daData, jump_test=jump_test, daWeight=daWeight, daEvents=daEvents
        )

    # MaxFz, entre de iniMov y despegue
    daEvents.loc[dict(event="maxFz")] = detect_maxFz(
        daData, jump_test=jump_test, daWeight=daWeight, daEvents=daEvents
    )

    # MinFz, entre de iniMov y maxFlex
    daEvents.loc[dict(event="minFz")] = detect_minFz(
        daData,
        jump_test=jump_test,
        daWeight=daWeight,
        daEvents=daEvents,
        threshold=threshold,
    )

    if "preactiv" in daEvents.event:
        daEvents.loc[dict(event="preactiv")] = (
            daEvents.loc[dict(event="iniMov")] - np.array(0.5) * daData.freq
        )  # para calcular preactivación en ventana de 0.5 s

    return daEvents


colr = {
    "iniAnalisis": "grey",
    "finAnalisis": "grey",
    "iniMov": "C0",
    "finMov": "C0",
    "preactiv": "dodgerblue",
    "iniPeso": "deepskyblue",
    "finPeso": "deepskyblue",
    #'iniPeso2':'deepskyblue', 'finPeso2':'deepskyblue', #para cuando se ponen a la vez el peso al inicio y al final
    "iniImpPos": "orange",
    "finImpPos": "orange",
    "despegue": "r",
    "aterrizaje": "r",
    "maxFz": "brown",
    "minFz": "cyan",
    "maxFlex": "g",
}


def _complete_in_graph_xr(g, adjust_iniend, daWeight, daEvents, test="generic") -> None:
    for h, ax in enumerate(g.axs):  # .axes): #extrae cada fila
        for i in range(len(ax)):  # extrae cada axis (gráfica)
            dimensiones = g.name_dicts[h, i]
            freq = g.data.freq

            if (
                dimensiones is None
            ):  # para cuando quedan huecos al final de la cuadrícula
                continue
            if g.data.sel(g.name_dicts[h, i]).isnull().all():
                continue
            # print(dimensiones)
            # plt.plot(g.data.sel(g.name_dicts[h, i]))
            # ID = str(g.data.loc[g.name_dicts[h, i]].ID.data)
            if "repe" not in g.data.dims:
                ax[i].set_title(
                    str(g.data.loc[g.name_dicts[h, i]].ID.data)
                )  # pone el nombre completo porque a veces lo recorta

            if (
                daWeight is not None and g.data.name == "Forces"
            ):  # isinstance(daWeight, xr.DataArray):
                # Pasar pesos solamente cuando se grafiquen fuerzas absolutas
                ax[i].axhline(
                    daWeight.sel(dimensiones).sel(stat="media").data,
                    color="C0",
                    lw=0.7,
                    ls="--",
                    dash_capstyle="round",
                    alpha=0.7,
                )

            if isinstance(daEvents, xr.DataArray):
                daEvents.isel(ID=0).to_dataframe()
                # Fill areas
                if g.data.name in ["Forces", "BW"] and daWeight is not None:
                    if g.data.name == "BW":
                        weight = 1.0
                    else:
                        weight = daWeight.sel(dimensiones).sel(stat="media").data

                    Fz = g.data.sel(dimensiones)
                    t = np.arange(len(Fz)) / freq
                    # Imp neg descent
                    if not any(
                        daEvents.sel(dimensiones)
                        .sel(event=["iniMov", "iniImpPos"])
                        .isnull()
                    ):
                        ini = int(daEvents.sel(dimensiones).sel(event="iniMov").data)
                        end = int(daEvents.sel(dimensiones).sel(event="iniImpPos").data)
                        ax[i].fill_between(
                            t[ini:end],
                            weight,
                            Fz[ini:end],
                            color=colr["iniMov"],
                            alpha=0.5,
                        )
                    # Imp posit descent
                    if not any(
                        daEvents.sel(dimensiones)
                        .sel(event=["iniImpPos", "maxFlex"])
                        .isnull()
                    ):
                        ini = int(daEvents.sel(dimensiones).sel(event="iniImpPos").data)
                        end = int(daEvents.sel(dimensiones).sel(event="maxFlex").data)
                        ax[i].fill_between(
                            t[ini:end],
                            weight,
                            Fz[ini:end],
                            color=colr["iniImpPos"],
                            alpha=0.5,
                        )
                    # Imp posit ascent
                    if test == "preacsj":
                        if not any(
                            daEvents.sel(dimensiones)
                            .sel(event=["iniImpPos", "finImpPos"])
                            .isnull()
                        ):
                            ini = int(
                                daEvents.sel(dimensiones).sel(event="iniImpPos").data
                            )
                            end = int(
                                daEvents.sel(dimensiones).sel(event="finImpPos").data
                            )
                            ax[i].fill_between(
                                t[ini:end],
                                weight,
                                Fz[ini:end],
                                color=colr["maxFlex"],
                                alpha=0.5,
                            )
                    elif test == "generic":
                        if not any(
                            daEvents.sel(dimensiones)
                            .sel(event=["maxFlex", "finImpPos"])
                            .isnull()
                        ):
                            ini = int(
                                daEvents.sel(dimensiones).sel(event="maxFlex").data
                            )
                            end = int(
                                daEvents.sel(dimensiones).sel(event="finImpPos").data
                            )
                            ax[i].fill_between(
                                t[ini:end],
                                weight,
                                Fz[ini:end],
                                color=colr["maxFlex"],
                                alpha=0.5,
                            )

                    # Imp negat ascent
                    if not any(
                        daEvents.sel(dimensiones)
                        .sel(event=["finImpPos", "despegue"])
                        .isnull()
                    ):
                        ini = int(daEvents.sel(dimensiones).sel(event="finImpPos").data)
                        end = int(daEvents.sel(dimensiones).sel(event="despegue").data)
                        ax[i].fill_between(
                            t[ini:end],
                            weight,
                            Fz[ini:end],
                            color=colr["finImpPos"],
                            alpha=0.5,
                        )
                    # Flight
                    if not any(
                        daEvents.sel(dimensiones)
                        .sel(event=["despegue", "aterrizaje"])
                        .isnull()
                    ):
                        ini = int(daEvents.sel(dimensiones).sel(event="despegue").data)
                        end = int(
                            daEvents.sel(dimensiones).sel(event="aterrizaje").data
                        )
                        ax[i].fill_between(
                            t[ini:end],
                            weight,
                            Fz[ini:end],
                            color=colr["despegue"],
                            alpha=0.5,
                        )

                # Draw vertical lines
                for ev in daEvents.sel(dimensiones):  # .event:
                    # print(ev.data)
                    if (
                        ev.isnull().all() or ev.count() > 1
                    ):  # np.isnan(ev): #si no existe el evento
                        continue

                    # No muestra ventana búsqueda peso si ajusta ini-fin
                    if (
                        str(ev.event.data) in ["iniPeso", "finPeso"] and adjust_iniend
                    ):  # se salta estos dos porque el array viene cortado por sus valores y tienen escala distinta
                        continue
                    # print(str(ev.data))

                    # Si no es un evento conocido le pone un color cualquiera
                    try:
                        col = colr[str(ev.event.data)]
                    except:
                        col = "k"
                    ax[i].axvline(
                        x=ev / g.data.freq,
                        c=col,
                        lw=1,
                        ls="--",
                        dashes=(5, 5),
                        dash_capstyle="round",
                        alpha=0.6,
                    )

                    # Ajusta altura de etiquetas
                    if str(ev.event.data) in ["iniImpPos", "finImpPos"]:
                        y_texto = ax[i].get_ylim()[1] * 0.7
                    elif str(ev.event.data) in ["minFz", "maxFlex"]:
                        y_texto = ax[i].get_ylim()[1] * 0.8
                    else:
                        y_texto = ax[i].get_ylim()[1] * 0.97

                    ax[i].text(
                        ev / g.data.freq,
                        y_texto,
                        ev.event.data,
                        ha="right",
                        va="top",
                        rotation="vertical",
                        c="k",
                        alpha=0.7,
                        fontsize="xx-small",
                        bbox=dict(
                            facecolor=col,
                            alpha=0.3,
                            edgecolor="none",
                            boxstyle="round,pad=0.3" + ",rounding_size=.5",
                        ),
                        transform=ax[i].transData,
                    )
            if adjust_iniend:
                fr = g.data.freq  # .loc[g.name_dicts[0, 0]]
                plt.xlim(
                    [
                        daEvents.sel(dimensiones).sel(event="iniAnalisis") / fr,
                        daEvents.sel(dimensiones).sel(event="finAnalisis") / fr,
                    ]
                )


def graphs_events(
    daData: xr.DataArray,
    daEvents: xr.DataArray | None = None,
    daWeight: xr.DataArray | None = None,
    num_per_block: int = 4,
    show_in_console: bool = True,
    adjust_iniend: bool = False,
    sharey: bool = False,
    test: str = "generic",
    work_path: Path | str | None = None,
    n_file_graph_global: str | None = None,
) -> None:
    """
    num_per_block is used to adjust the number of graphs per sheet. If the data
    have repe dimension, num_per_block indicates the number of rows per sheet with repeat columns.
    If they do not have repe, each sheet has num_per_block x num_per_block plots.
    """
    # if isinstance(work_path, str):
    #     work_path = Path(work_path)

    timerGraf = time.perf_counter()
    print("\nCreating graphs...")

    # import seaborn as sns

    # Si no se incluye nombre archivo no guarda el pdf
    if n_file_graph_global is not None and work_path is not None:
        if isinstance(work_path, str):
            work_path = Path(work_path)
        nompdf = (work_path / n_file_graph_global).with_suffix(".pdf")
        pdf_pages = PdfPages(nompdf)

    if "axis" in daData.dims:  # por si se envía un da filtrado por axis
        daData = daData.sel(axis="z")
    if (
        daEvents is not None and "axis" in daEvents.dims
    ):  # por si se envía un da filtrado por eje
        daEvents = daEvents.sel(axis="z")
    if (
        daWeight is not None and "axis" in daWeight.dims
    ):  # por si se envía un da filtrado por eje
        daWeight = daWeight.sel(axis="z")

    if adjust_iniend and daEvents is not None:
        # ini = daEvents.isel(event=daEvents.argmin(dim="event"))
        daData = trim_analysis_window(
            daData, daEvents.sel(event=["iniAnalisis", "finAnalisis"])
        )
        if daEvents is not None:
            daEvents = daEvents - daEvents.sel(event="iniAnalisis")

    # Por si no hay dimensión 'repe'
    if "repe" in daData.dims:  # dfDatos.columns:
        fils_cols = dict(row="ID", col="repe")
        distribuidor = num_per_block
    else:
        fils_cols = dict(col="ID", col_wrap=num_per_block)
        distribuidor = num_per_block**2

    # if "repe" in daData.dims:
    #     distribuidor = num_per_block
    # else:
    #     distribuidor = num_per_block**2

    # daData.drop_vars('tipo').plot.line(x='time',col='ID')
    # daData.to_pandas().T.plot()

    for n in range(0, len(daData.ID), distribuidor):
        dax = daData.isel(ID=slice(n, n + distribuidor))
        g = dax.plot.line(
            x="time", alpha=0.8, aspect=1.5, sharey=sharey, **fils_cols
        )  # , lw=1)
        _complete_in_graph_xr(g, adjust_iniend, daWeight, daEvents, test=test)

        if n_file_graph_global is not None and work_path is not None:
            pdf_pages.savefig(g.fig)

        print(f"Graphs Completed {n} at {n + distribuidor} of {len(daData.ID)}")

        if not show_in_console:
            plt.close()  # para que no muestre las gráficas en consola y vaya más rápido

    # if 'repe' not in daData.dims and num_per_block is not None:
    #     #for n, dax in daData.assign_coords(ID=np.arange(len(daData.ID))).groupby_bins('ID', bins=range(0, len(daData.ID) + num_per_block**2, num_per_block**2), include_lowest=True):
    #     for n in range(0,len(daData.ID), num_per_block**2):
    #         dax = daData.isel(ID=slice(n,n+num_per_block**2))

    #         g=dax.plot.line(x='time', alpha=0.8, aspect=1.5, sharey=False, **fils_cols, lw=1)
    #         completa_grafica_xr(g)

    #         if n_file_graph_global is not None:
    #             pdf_pages.savefig(g.fig)

    # else:
    #     for n in range(0,len(daData.ID), num_per_block):
    #         dax = daData.isel(ID=slice(n,n+num_per_block))

    #         g=dax.plot.line(x='time', alpha=0.8, aspect=1.5, sharey=False, **fils_cols, lw=1)
    #         completa_grafica_xr(g)

    #     if n_file_graph_global is not None:
    #         pdf_pages.savefig(g.fig)

    """
    def fun(x,y): #prueba para dibujar con xarray directamente
        print(x,y)
    g=daData.isel(ID=slice(None,3)).plot.line(x='time', **fils_cols)
    g.map_dataarray_line(fun, x='time', y='fuerza', hue='repe')
    """
    # g = sns.relplot(data=dfDatos, x='time', y='Fuerza', col='ID', col_wrap=4, hue='repe',
    #                 estimator=None, ci=95, units='repe',
    #                 facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
    #                 palette=sns.color_palette(col), alpha=0.7)

    """
    #Versión Seaborn
    def dibuja_X(x,y, color, **kwargs):   
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        #print(y, ID, repe, color, kwargs.keys())
        #plt.vlines(daEvents.sel(ID=ID, repe=repe)/daData.freq, ymin=kwargs['data'].loc[:,'Fuerza'].min(), ymax=kwargs['data'].loc[:,'Fuerza'].max(), colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6) # plt.gca().get_ylim()[1] transform=plt.gca().transData)
        #Líneas del peso
        if daWeight is not None: #isinstance(daWeight, xr.DataArray):
            plt.axhline(daWeight.sel(ID=ID, repe=repe, stat='media').data, color='C0', lw=1, ls='--', dash_capstyle='round', alpha=0.6)
       
        for ev in daEvents.sel(ID=ID, repe=repe).event:
            if str(ev.data) not in ['iniAnalisis', 'finAnalisis']: #se salta estos dos porque el array viene cortado por sus valores y tienen escala distinta
                #print(str(ev.data))
                #print(daEvents.sel(ID=ID, repe=repe,event=ev))
            # for num, ev in daEvents.sel(ID=ID, repe=repe).groupby('evento'):
            #     print('\n',num)
                if not np.isnan(daEvents.sel(ID=ID, repe=repe, event=ev)): #si existe el evento
                    plt.axvline(x=daEvents.sel(ID=ID, repe=repe, event=ev)/daData.freq, c=col[str(ev.data)], lw=0.5, ls='--', dashes=(5, 5), dash_capstyle='round', alpha=0.5)
                    y_texto = plt.gca().get_ylim()[1] if str(ev.data) not in ['minFz', 'despegue', 'maxFz'] else plt.gca().get_ylim()[1]*0.8
                    plt.text(daEvents.sel(ID=ID, repe=repe, event=ev).data/daData.freq, y_texto, ev.data,
                             ha='right', va='top', rotation='vertical', c='k', alpha=0.6, fontsize=8, 
                             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), transform=plt.gca().transData)
    
    g = sns.relplot(data=dfDatos[dfDatos['Fuerza'].notnull()], x='time', y='Fuerza', #ATENCIÓN en esta versión de seaborn (12.1) falla con datos nan, por eso se seleccionan los notnull()                    
                    #estimator=None, ci=95, units='repe',
                    facet_kws={'sharey': False, 'legend_out':True}, solid_capstyle='round', kind='line',
                    alpha=0.7, aspect=1.5,
                    **fils_cols) #palette=sns.color_palette(col), 
    if daEvents is not None:
        g.map_dataframe(dibuja_X, x='time', y='Fuerza', lw=0.25, alpha=0.3)
    """

    """
    def dibuja_xr(x,y, **kwargs):
        ID = kwargs['data'].loc[:,'ID'].unique()[0]
        repe = kwargs['data'].loc[:,'repe'].unique()
        print(y, ID, repe, color, kwargs.keys())
    
    g=daData.sel(axis='z').plot.line(x='time', col='ID', col_wrap=4, hue='repe', sharey=False)
    #g = xr.plot.FacetGrid(self.datos, col='ID', col_wrap=4)
    g.map_dataarray_line(dibuja_xr, x='time', y=None, hue='repe')#, y='trial')
    col=['C0', 'C1', 'C2']
    for h, ax in enumerate(g.axs): #extrae cada fila
        for i in range(len(ax)): #extrae cada axis (gráfica)     
            try:
                idn = g.data.loc[g.name_dicts[h, i]].ID
                #print('weight=', daWeight.sel(ID=idn).data)#idn)
                #Rango medida weight
                #ax[i].axvspan(g.data.time[int(window[0]*self.datos.freq)], g.data.time[int(window[1]*self.datos.freq)], alpha=0.2, color='C1')
                #for j in daData.repe:
                for e in daEvents.sel(ID=idn):
                    #print(e)
                    ax[i].vlines(e/daData.freq, ymin=g.data.sel(ID=idn).min(), ymax=g.data.sel(ID=idn).max(), colors=['C0', 'C1', 'C2'], lw=1, ls='--', alpha=0.6) # plt.gca().get_ylim()[1] transform=plt.gca().transData)
            except:
                print("No va el", h,i)
    """

    # Cierra el pdf
    if n_file_graph_global is not None:
        # pdf_pages.savefig(g.fig)
        pdf_pages.close()
        print(f"\nGraph saved {nompdf}")
    print("Graphs created in {0:.3f} s \n".format(time.perf_counter() - timerGraf))


def graphs_all_variables(
    dsData: xr.Dataset,
    show_events: bool = False,
    daWeight: xr.DataArray = None,
    num_per_block: int = 4,
    show_in_console: bool = True,
    adjust_iniend: bool = False,
    work_path: Path | str = None,
    n_file_graph_global: str = None,
) -> None:
    """
    Plots all variables in the dataset, with options for block size, console display,
    adjustment, working directory, and global file name.
    adjust_iniend: si es True recorta según iniAnalisis y fin Analisis.
    """

    if "axis" in dsData.dims:  # por si se envía un da filtrado por axis
        dsData = dsData.sel(axis="z")

    if num_per_block > len(dsData.ID):
        num_per_block = len(dsData.ID)

    daData = dsData[["BW", "v", "s", "P"]].to_array()

    daEvents = dsData["events"] if "events" in list(dsData.keys()) else None
    daWeight = dsData["weight"] if "weight" in list(dsData.keys()) else None

    # daData.loc[dict(variable='P')] = daData.loc[dict(variable='P')] / daWeight.sel(stat='media')

    timerGraf = time.time()
    print("\nCreating graphs...")

    # import seaborn as sns

    # Si no se incluye nombre archivo no guarda el pdf
    if n_file_graph_global is not None:
        if not isinstance(work_path, Path):
            work_path = Path(work_path)
        nompdf = (work_path / n_file_graph_global).with_suffix(".pdf")
        pdf_pages = PdfPages(nompdf)

    # if 'axis' in daData.dims: #por si se envía un da filtrado por axis
    #     daData=daData.sel(axis='z')
    # if daEvents is not None and 'axis' in daEvents.dims: #por si se envía un da filtrado por eje
    #     daEvents=daEvents.sel(axis='z')
    # if daWeight is not None and 'axis' in daWeight.dims: #por si se envía un da filtrado por eje
    #     daWeight=daWeight.sel(axis='z')

    if adjust_iniend:
        daData = trim_analysis_window(
            daData, daEvents.sel(event=["iniAnalisis", "finAnalisis"])
        )
        if daEvents:  # is not None:
            daEvents = daEvents - daEvents.sel(event="iniAnalisis")

    # Por si no hay dimensión 'repe'
    if "repe" in daData.dims:  # dfDatos.columns:
        fils_cols = dict(row="ID", col="repe")
        distribuidor = num_per_block
    else:
        fils_cols = dict(col="ID", col_wrap=num_per_block)
        distribuidor = num_per_block**2

    for n in range(0, len(daData.ID), distribuidor):
        dax = daData.isel(ID=slice(n, n + distribuidor))

        g = dax.plot.line(
            x="time", alpha=0.8, aspect=1.5, sharey=False, **fils_cols
        )  # , lw=1)
        _complete_in_graph_xr(g, adjust_iniend, daWeight, daEvents)

        if n_file_graph_global is not None:
            pdf_pages.savefig(g.fig)

        print(f"Completed graphs {n} to {n + distribuidor} of {len(daData.ID)}")

        if not show_in_console:
            plt.close()  # para que no muestre las gráficas en consola y vaya más rápido

    # Cierra el pdf
    if n_file_graph_global is not None:
        # pdf_pages.savefig(g.fig)
        pdf_pages.close()
        print(f"Graph saved {nompdf}")
    print("Graphs created in {0:.3f} s \n".format(time.time() - timerGraf))


def graphs_weight_check(
    daData: xr.DataArray,
    daEvents: xr.DataArray,
    dsWeights: xr.Dataset | None = None,
    daWeight: xr.DataArray | None = None,
    threshold_weight_diff: float = 20,
    daWeight_mean: xr.DataArray | None = None,
    allowed_window: float = 0.3,
    num_per_block: int = 4,
    show_in_console: bool = True,
    work_path: Path | str | None = None,
    n_file_graph_global: str | None = None,
) -> None:
    """
    allowed_window: time in seconds ahead and ahead and behind iniPeso and endPeso
    """
    timer_graf = time.time()
    print("\nCreating graphs...")

    # import seaborn as sns

    # Si no se incluye nombre archivo no guarda el pdf
    if n_file_graph_global != None and work_path is not None:
        if isinstance(work_path, str):
            work_path = Path(work_path)
        nompdf = (work_path / n_file_graph_global).with_suffix(".pdf")
        pdf_pages = PdfPages(nompdf)

    if isinstance(allowed_window, float):
        allowed_window = allowed_window * daData.freq

    if "axis" in daData.dims:  # por si se envía un da filtrado por axis
        daData = daData.sel(axis="z")
    if (
        daEvents is not None and "axis" in daEvents.dims
    ):  # por si se envía un da filtrado por eje
        daEvents = daEvents.sel(axis="z")
    if (
        dsWeights is not None and "axis" in dsWeights.dims
    ):  # por si se envía un da filtrado por eje
        dsWeights = dsWeights.sel(axis="z")
    if (
        daWeight is not None and "axis" in daWeight.dims
    ):  # por si se envía un da filtrado por eje
        daWeight = daWeight.sel(axis="z")
    if (
        daWeight_mean is not None and "axis" in daWeight_mean.dims
    ):  # por si se envía un da filtrado por eje
        daWeight_mean = daWeight_mean.sel(axis="z")

    # Por si no hay dimensión 'repe'
    if "repe" in daData.dims:  # dfDatos.columns:
        fils_cols = dict(row="ID", col="repe")
    else:
        fils_cols = dict(col="ID", col_wrap=num_per_block)

    if "repe" in daData.dims:
        distribuidor = num_per_block
    else:
        distribuidor = num_per_block**2

    """
    def completa_peso(g, daEvents, daWeight, daWeight_mean):
        for h, ax in enumerate(g.axs):#.axes): #extrae cada fila
            for i in range(len(ax)): #extrae cada axis (gráfica)                    
                dimensiones = g.name_dicts[h, i]
                peso_afinado = daWeight.sel(dimensiones).sel(stat='media').data
                peso_media = daWeight_mean.sel(dimensiones).sel(stat='media').data
                
                #print(dimensiones)
                if dimensiones is None: #para cuando quedan huecos al final de la cuadrícula
                    continue
                #plt.plot(g.data.sel(g.name_dicts[h, i]))
                #ID = str(g.data.loc[g.name_dicts[h, i]].ID.data)
                if 'repe' not in g.data.dims:
                    ax[i].set_title(str(g.data.loc[g.name_dicts[h, i]].ID.data)) #pone el nombre completo porque a veces lo recorta
                
                
                if daWeight is not None: #isinstance(daWeight, xr.DataArray):
                    ax[i].axhline(peso_afinado, color='C0', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
                    ax[i].text(0.0, peso_afinado, 'peso afinado', 
                                ha='left', va='bottom', rotation='horizontal', c='C0', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transData
                                )
                    ax[i].text(0.05, 0.1, f'Peso afinado={peso_afinado:.1f} N', 
                                ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transAxes
                                )
            
                
                if daWeight_mean is not None: #isinstance(daWeight, xr.DataArray):
                    ax[i].axhline(peso_media, color='C1', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
                    ax[i].text(0.3, peso_media, 'peso media', 
                                ha='left', va='bottom', rotation='horizontal', c='C1', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transData
                                )
                    ax[i].text(0.05, 0.05, f'Peso media={peso_media:.1f} N', 
                                ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                                transform=ax[i].transAxes
                                )
                    
                #Si la diferencia es mayor que el umbral, avisa
                if abs(peso_media-peso_afinado) > threshold_weight_diff:
                    ax[i].text(0.5, 0.9, f'REVISAR (dif={peso_media-peso_afinado:.1f} N)', 
                                ha='center', va='center', rotation='horizontal', c='r', alpha=0.7, fontsize='large', 
                                bbox=dict(facecolor='lightgrey', alpha=0.3, edgecolor='r', boxstyle='round,pad=0.3'+',rounding_size=.5'), 
                                transform=ax[i].transAxes
                                )
                
                
                if isinstance(daEvents, xr.DataArray):
                    for ev in daEvents.sel(dimensiones):#.event:
                        if ev.isnull().all():#np.isnan(ev): #si no existe el evento
                            continue                       
                        
                        ax[i].axvline(x=ev/g.data.freq, c=colr[str(ev.event.data)], lw=1, ls='--', dashes=(5, 5), dash_capstyle='round', alpha=0.7)
                        
                        y_texto = ax[i].get_ylim()[1]*0.97
                        ax[i].text(ev / g.data.freq, y_texto, ev.event.data,
                                 ha='right', va='top', rotation='vertical', c='k', alpha=0.7, fontsize='small', 
                                 bbox=dict(facecolor=colr[str(ev.event.data)], alpha=0.2, edgecolor='none', boxstyle='round,pad=0.3'+',rounding_size=.5'), 
                                 transform=ax[i].transData)
    """

    daWindow = daEvents.sel(event=["iniPeso", "finPeso"]).copy()
    daWindow.loc[dict(event="iniPeso")] = xr.where(
        daWindow.loc[dict(event="iniPeso")] - allowed_window > 0,
        daWindow.loc[dict(event="iniPeso")] - allowed_window,
        0,
    )  # daWindow.loc[dict(event='iniPeso')] - allowed_window
    daWindow.loc[dict(event="finPeso")] = xr.where(
        daWindow.loc[dict(event="finPeso")] + allowed_window
        < daData.time.size,  # daEvents.loc[dict(event="finAnalisis")],
        daWindow.loc[dict(event="finPeso")] + allowed_window,
        daData.time.size,  # daEvents.loc[dict(event="finAnalisis")],
    )  # daWindow.loc[dict(event='finPeso')] + allowed_window
    # #TODO: ESTO NO SE AJUSTA BIEN CUANDO LA VENTANA ESTÁ CERCA DE CERO Y SALE NEGATIVO
    # daWindow = xr.where(daWindow.loc[dict(event='iniPeso')] < 0, daWindow - daWindow.loc[dict(event='iniPeso')], daWindow)
    # daWindow = xr.where(daWindow.loc[dict(event='finPeso')] > len(daData.time), daWindow - (daWindow.loc[dict(event='finPeso')] - len(daData.time)), daWindow)

    # daWindow.loc[dict(event=["iniPeso", "finPeso"])] = [19000,19500]
    daDat = trim_analysis_window(daData, daWindow)
    daEvents = daEvents - daWindow.loc[dict(event="iniPeso")]

    # daWindow = daWindow - daWindow.sel(event='iniPeso') #+ 0.5*daData.freq
    # daWindow.loc[dict(event='iniPeso')] = daWindow.loc[dict(event='iniPeso')] + allowed_window*daDat.freq
    # daWindow.loc[dict(event='finPeso')] = daWindow.loc[dict(event='finPeso')] - allowed_window*daDat.freq

    """
    for n in range(0,len(daData.ID), distribuidor):
        dax = daDat.isel(ID=slice(n, n + distribuidor))
        
        g=dax.plot.line(x='time', alpha=0.8, aspect=1.5, color='lightgrey', sharey=False, **fils_cols) #, lw=1)
        completa_peso(g, daEvents.sel(event=['iniPeso', 'finPeso']), daWeight, daWeight_mean)
                        
        if n_file_graph_global is not None:
            pdf_pages.savefig(g.fig)
            
        print(f'Completadas gráficas {n} a {n + distribuidor} de {len(daData.ID)}')
        
        if not show_in_console: plt.close() #para que no muestre las gráficas en consola y vaya más rápido
    """

    def _completa_graf_xr(*args, color):
        if len(args) == 3:
            ID, repe, time = args
            coords_graph = dict(ID=ID, repe=repe)
        else:
            ID, time = args
            coords_graph = dict(ID=ID)

        # data = g.data.loc[coords_graph]
        # print(ID, repe, time)
        # peso_afinado = daWeight.sel(ID=ID).sel(stat='media').data
        # peso_media = daWeight_mean.sel(ID=ID).sel(stat='media').data

        if "repe" not in g.data.dims:
            plt.title(ID)  # pone el nombre completo porque a veces lo recorta

        """
        if peso_afinado is not None: #isinstance(daWeight, xr.DataArray):
            plt.axhline(peso_afinado, color='C0', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
            plt.text(0.0, peso_afinado, 'peso afinado', 
                        ha='left', va='bottom', rotation='horizontal', c='C0', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transData
                        )
            plt.text(0.05, 0.1, f'Peso afinado={peso_afinado:.1f} N', 
                        ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transAxes
                        )
        if peso_media is not None: #isinstance(daWeight, xr.DataArray):
            plt.axhline(peso_media, color='C1', lw=1, ls='--', dash_capstyle='round', alpha=0.7)
            plt.text(0.3, peso_media, 'peso media', 
                        ha='left', va='bottom', rotation='horizontal', c='C1', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transData
                        )
            plt.text(0.05, 0.05, f'Peso media={peso_media:.1f} N', 
                        ha='left', va='top', rotation='horizontal', c='k', alpha=0.7, fontsize='x-small', 
                        transform=plt.gca().transAxes
                        )
        #Si la diferencia es mayor que el umbral, avisa
        if abs(peso_media-peso_afinado) > threshold_weight_diff:
            plt.text(0.5, 0.9, f'REVISAR (dif={peso_media-peso_afinado:.1f} N)', 
                        ha='center', va='center', rotation='horizontal', c='r', alpha=0.7, fontsize='large', 
                        bbox=dict(facecolor='lightgrey', alpha=0.3, edgecolor='r', boxstyle='round,pad=0.3'+',rounding_size=.5'), 
                        transform=plt.gca().transAxes
                        )
        """
        # Dibuja líneas distintos tipos de cálculo peso
        if isinstance(dsWeights, xr.Dataset):
            color = ["C0", "C1", "C2"]
            for n, n_tipo_peso in enumerate(dsWeights.sel(coords_graph)):
                weight = dsWeights[n_tipo_peso].sel(coords_graph).data
                plt.axhline(
                    weight,
                    color=color[n],
                    lw=1,
                    ls="--",
                    dash_capstyle="round",
                    alpha=0.7,
                )
                plt.text(
                    0.0,
                    weight,
                    f"weight {n_tipo_peso}={weight:.1f} N",
                    ha="left",
                    va="bottom",
                    rotation="horizontal",
                    c=color[n],
                    alpha=0.7,
                    fontsize="x-small",
                    transform=plt.gca().transData,
                )

                if n > 0:
                    # Si la diferencia es mayor que el umbral, avisa
                    if (
                        abs(weight - dsWeights["media"].sel(coords_graph))
                        > threshold_weight_diff
                    ):
                        plt.text(
                            0.5,
                            0.9,
                            f"REVIEW (delta={weight - dsWeights['media'].sel(coords_graph):.1f} N)",
                            ha="center",
                            va="center",
                            rotation="horizontal",
                            c="r",
                            alpha=0.7,
                            fontsize="large",
                            bbox=dict(
                                facecolor="lightgrey",
                                alpha=0.3,
                                edgecolor="r",
                                boxstyle="round,pad=0.3" + ",rounding_size=.5",
                            ),
                            transform=plt.gca().transAxes,
                        )

        if isinstance(daEvents, xr.DataArray):
            # coords_graph_events = coords_graph.copy()
            # coords_graph_events["event"] = ["iniPeso", "finPeso"]

            for ev in daEvents.sel(
                dict(coords_graph, event=["iniPeso", "finPeso"])
            ):  # .event:
                if ev.isnull().all():  # np.isnan(ev): #si no existe el evento
                    continue
                plt.axvline(
                    x=ev / g.data.freq,
                    c=colr[str(ev.event.data)],
                    lw=1,
                    ls="--",
                    dashes=(5, 5),
                    dash_capstyle="round",
                    alpha=0.7,
                )
                y_texto = (
                    plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
                ) * 0.97 + plt.gca().get_ylim()[
                    0
                ]  # escala a las coordenadas de la variable en cada gráfica
                plt.text(
                    ev / g.data.freq,
                    y_texto,
                    ev.event.data,
                    ha="right",
                    va="top",
                    rotation="vertical",
                    c="k",
                    alpha=0.7,
                    fontsize="small",
                    bbox=dict(
                        facecolor=colr[str(ev.event.data)],
                        alpha=0.2,
                        edgecolor="none",
                        boxstyle="round,pad=0.3" + ",rounding_size=.5",
                    ),
                    transform=plt.gca().transData,
                )

    for n in range(0, len(daData.ID), distribuidor):
        dax = daDat.isel(ID=slice(n, n + distribuidor))
        """
        ID = dax.ID[0].data
        repe = dax.repe[0].data
        """
        g = dax.plot.line(
            x="time",
            alpha=0.8,
            lw=1,
            aspect=1.5,
            color="lightgrey",
            sharey=False,
            **fils_cols,
        )  # , lw=1)
        if "repe" in dax.dims:
            g.map(_completa_graf_xr, "ID", "repe", "time", color=0)
        else:
            g.map(_completa_graf_xr, "ID", "time", color=0)
        # _completa_graf_xr(g, daEvents.sel(event=['iniPeso', 'finPeso']), daWeight, daWeight_mean)

        if n_file_graph_global is not None:
            pdf_pages.savefig(g.fig)

        print(f"Completadas gráficas {n} a {n + distribuidor} de {len(daData.ID)}")

        if not show_in_console:
            plt.close()  # para que no muestre las gráficas en consola y vaya más rápido

    # Cierra el pdf
    if n_file_graph_global is not None:
        # pdf_pages.savefig(g.fig)
        pdf_pages.close()
        print(f"Graph saved {nompdf}")
    print("Graphs created in {0:.3f} s \n".format(time.time() - timer_graf))


# TODO: REPLACE WITH slice_time_series_phases.trim_window
def trim_analysis_window(
    daData: xr.DataArray | xr.Dataset, daEvents: xr.DataArray, window=None
) -> xr.DataArray:
    """
    If a value is passed to window, only one event should be passed.
    Add the window (in seconds) to the initial event.
    """
    # TODO: TRY WITH DA.PAD

    def _trim_window(datos, ini, fin):
        # print(datos.shape, ini,fin)
        d2 = np.full(
            datos.shape, np.nan
        )  # rellena con nan al final para que tengan mismo tamaño
        try:
            ini = int(ini)
            fin = int(fin)
            if ini < 0:
                ini = 0
            if fin > len(datos):
                fin = len(datos)
            d2[: fin - ini] = datos[ini:fin]
        except:
            pass
        return d2  # datos[int(ini):int(fin)]

    if window is not None:
        if window > 0:
            daIni = daEvents
            daFin = daEvents + window * daData.freq
        else:
            daIni = daEvents + window * daData.freq
            daFin = daEvents

    else:
        daIni = daEvents.isel(event=0)
        daFin = daEvents.isel(event=1)

    daCortado = (
        xr.apply_ufunc(
            _trim_window,
            daData,
            daIni,
            daFin,  # .sel(ID=daData.ID, repe=daData.repe)
            input_core_dims=[["time"], [], []],
            output_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            vectorize=True,
            # join='outer'
        )
        .assign_coords({"time": daData.time})
        .dropna(dim="time", how="all")
    )
    daCortado.attrs = daData.attrs

    if not isinstance(daCortado, xr.Dataset):
        daCortado.name = daData.name
        daCortado = daCortado.astype(daData.dtype)
    else:
        for var in list(daCortado.data_vars):  # ['F', 'v', 's', 'P', 'RFD']:
            daCortado[var].attrs = daData[var].attrs

    # daCortado.plot.line(x='time', row='ID', col='axis')
    return daCortado


def adjust_offsetFz_flight_min(
    daData: xr.DataArray,
    jump_test: str = None,
    threshold: float = 20.0,
    pct_window: int = 5,
    show: bool = False,
) -> xr.DataArray:
    # Hace media de valores por debajo del threshold. Si es DJ2P considera los dos tramos en vuelo
    # Mejor aplicarlo antes de filtrar
    daReturn = daData.copy()

    if "axis" in daData.dims:
        daReturn = daReturn.sel(axis="z")

    minim = daReturn.min("time")
    # minim.sel(ID='S06_DJ_30', repe=2)
    offset = minim + threshold
    # offset.sel(ID='S06_DJ_30', repe=2)
    resta = daReturn.where(daReturn < offset).mean("time")
    # resta.sel(ID='S06_DJ_30', repe=2)
    # daReturn.sel(ID='S06_DJ_30', repe=2).plot.line(x='time')
    daReturn = daReturn - resta

    if show:
        if "plat" in daData.dims:
            if "repe" in daData.dims:
                daReturn.stack(ID_repe=("ID", "repe")).where(
                    daReturn.stack(ID_repe=("ID", "repe"))
                    < offset.stack(ID_repe=("ID", "repe"))
                ).plot.line(x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7)
                # Suspects
                sospechosos = daReturn.stack(ID_repe=("ID", "repe", "plat")).where(
                    np.abs(resta.stack(ID_repe=("ID", "repe", "plat"))) > threshold,
                    drop=True,
                )
                sospechosos.plot.line(
                    x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                )
                print(f"Suspects {sospechosos.ID_repe.data}")
        else:
            if "repe" in daData.dims:
                daReturn.stack(ID_repe=("ID", "repe")).where(
                    daReturn.stack(ID_repe=("ID", "repe"))
                    < offset.stack(ID_repe=("ID", "repe"))
                ).plot.line(x="time", col="ID", col_wrap=4)
            else:
                daReturn.where(daReturn < threshold * 2.5).plot.line(
                    x="time", col="ID", col_wrap=4
                )

    daReturn = daReturn - resta

    if "axis" in daData.dims:
        daData.loc[dict(axis="z")] = daReturn
    else:
        daData = daReturn

    return daData


def adjust_offsetFz(
    daData: xr.DataArray,
    jump_test: str | None = None,
    threshold: float = 20.0,
    pct_window: int = 5,
    kind: str = "flight",
    show: bool = False,
    show_detail: bool = False,
) -> xr.DataArray:
    """
    Apply in case of erroneous force plate offset.
    Better to apply it before filtering.

    kind: one of "mean", "flight", "threshold"
    show: show grid of plots
    show_detail: show evey single detection plot

    Reference: "The offset voltage of the unloaded force platform was determined by finding the 0.4 s
    moving average during the flight phase with the smallest standard deviation."
    Street, G., McMillan, S., Board, W., Rasmussen, M., & Heneghan, J. M. (2001).
    Sources of Error in Determining Countermovement Jump Height with the Impulse Method. Journal of Applied Biomechanics, 17(1), 43-54. https://doi.org/10.1123/jab.17.1.43
    """

    if kind not in ["mean", "flight", "threshold"]:
        raise ValueError(r"kind must be one of 'mean', 'flight' or 'threshold'")

    daReturn = daData.copy()
    if "axis" in daData.dims:
        daReturn = daReturn.sel(axis="z")

    if kind == "mean":
        if jump_test == "DJ2PApart":
            # Se asume que es cero plat1 al principio y plat2 al final
            offset_plat1 = daReturn.sel(plat=1, time=slice(0, 1)).mean("time")
            offset_plat2 = daReturn.sel(
                plat=2, time=slice(daReturn.time[-1] - 1.5, daReturn.time[-1] - 0.5)
            ).mean("time")

            daReturn.loc[dict(plat=1)] -= offset_plat1
            daReturn.loc[dict(plat=2)] -= offset_plat2

            if show:
                if "plat" in daData.dims:
                    if "repe" in daData.dims:
                        daReturn.stack(ID_repe=("ID", "repe")).where(
                            daReturn.stack(ID_repe=("ID", "repe")) < threshold
                        ).plot.line(
                            x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                        )

                else:
                    if "repe" in daData.dims:
                        daReturn.stack(ID_repe=("ID", "repe")).where(
                            daReturn.stack(ID_repe=("ID", "repe")) < threshold
                        ).plot.line(x="time", col="ID", col_wrap=4)
                    else:
                        daReturn.where(daReturn < threshold).plot.line(
                            x="time", col="ID", col_wrap=4
                        )
        else:
            raise NotImplementedError("'mean' adjustment only for 'DJ2PApart' test")

    if kind == "flight":  # sin comprobar
        # Ajusta buscando los vuelos concretamente
        if jump_test == "DJ2P":
            # busca despegue y aterrizaje provisionales
            vuelo = detect_takeoff_landing(
                daReturn, jump_test, threshold=threshold, show=show_detail
            )
            split_window = (
                (
                    vuelo.loc[dict(event="aterrizaje")]
                    - vuelo.loc[dict(event="despegue")]
                )
                * pct_window
                / 100
            ).round()  # astype(np.int32)
            vuelo.loc[dict(event="despegue")] += split_window
            vuelo.loc[dict(event="aterrizaje")] -= split_window
            offset_flight = trim_analysis_window(daReturn, vuelo).mean(dim="time")

        else:
            # busca despegue y aterrizaje provisionales
            vuelo = detect_takeoff_landing(
                daReturn, jump_test, threshold=threshold, show=show_detail
            )
            # reduces the window a little to avoid possible bounces from filtering
            split_window = (
                (
                    vuelo.loc[dict(event="aterrizaje")]
                    - vuelo.loc[dict(event="despegue")]
                )
                * pct_window
                / 100
            ).round()  # .astype(np.int32)  # TODO: CHECK FOR NAN
            vuelo.loc[dict(event="despegue")] += split_window
            vuelo.loc[dict(event="aterrizaje")] -= split_window

            offset_flight = trim_analysis_window(daReturn, vuelo).mean(dim="time")
            # trim_analysis_window(daData, vuelo).sel(axis='x').plot.line(x='time', col='ID', col_wrap=4)
            # offset_flight.plot()
            # offset_flight.sel(axis='z').plot.line(col='ID', col_wrap=4, hue='repe')
            # daData -= offset_flight
        daReturn = daReturn - offset_flight
        daReturn.attrs = daData.attrs

        if show:
            if "plat" in daData.dims:
                if "repe" in daData.dims:
                    offset_flight.plot.line(
                        col="ID", col_wrap=4, hue="axis", sharey=False
                    )
            else:
                trim_analysis_window(
                    daReturn, vuelo.sel(event=["despegue", "aterrizaje"])
                ).plot.line(col="ID", col_wrap=4, hue="repe", sharey=False)

    elif kind == "threshold":
        # Hace media de valores por debajo del umbral. Si es DJ2P considera los dos tramos en vuelo

        offset = daReturn.where(daReturn < threshold).mean("time")

        if show:
            if "plat" in daData.dims:
                if "repe" in daData.dims:
                    daReturn.stack(ID_repe=("ID", "repe")).where(
                        daReturn.stack(ID_repe=("ID", "repe")) < threshold * 2.5
                    ).plot.line(
                        x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                    )
            else:
                daReturn.where(daReturn < threshold * 2.5).plot.line(
                    x="time", col="ID", col_wrap=4
                )

        daReturn = daReturn - offset

    elif kind == "min":
        # Hace media de valores por debajo del umbral. Si es DJ2P considera los dos tramos en vuelo

        minim = daReturn.min("time")
        # minim.sel(ID='S06_DJ_30', repe=2)
        offset = minim + threshold
        # offset.sel(ID='S06_DJ_30', repe=2)
        resta = daReturn.where(daReturn < offset).mean("time")
        # resta.sel(ID='S06_DJ_30', repe=2)
        # daReturn.sel(ID='S06_DJ_30', repe=2).plot.line(x='time')
        daReturn = daReturn - resta

        if show:
            if "plat" in daData.dims:
                if "repe" in daData.dims:
                    daReturn.stack(ID_repe=("ID", "repe")).where(
                        daReturn.stack(ID_repe=("ID", "repe"))
                        < offset.stack(ID_repe=("ID", "repe"))
                    ).plot.line(
                        x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                    )
                    # Sospechosos
                    sospechosos = daReturn.stack(ID_repe=("ID", "repe", "plat")).where(
                        np.abs(resta.stack(ID_repe=("ID", "repe", "plat"))) > threshold,
                        drop=True,
                    )
                    sospechosos.plot.line(
                        x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7
                    )
                    print(f"Sospechosos {sospechosos.ID_repe.data}")
            else:
                if "repe" in daData.dims:
                    daReturn.stack(ID_repe=("ID", "repe")).where(
                        daReturn.stack(ID_repe=("ID", "repe"))
                        < offset.stack(ID_repe=("ID", "repe"))
                    ).plot.line(x="time", col="ID", col_wrap=4)
                else:
                    daReturn.where(daReturn < threshold * 2.5).plot.line(
                        x="time", col="ID", col_wrap=4
                    )

        daReturn = daReturn - resta

    if "axis" in daData.dims:
        daData.loc[dict(axis="z")] = daReturn
    else:
        daData = daReturn

    return daData


def adjust_offsetFz_flight_conventional(
    daData: xr.DataArray,
    jump_test: str = "CMJ",
    threshold: float = 20.0,
    pct_window: int = 5,
    show: bool = False,
) -> xr.DataArray:
    # Adjust by searching for specific flights
    if jump_test == "DJ2P":
        vuelo = detect_takeoff_landing(
            daData, jump_test, threshold=threshold
        )  # , show=show)
        split_window = (
            (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
            * pct_window
            / 100
        ).astype("int32")
        vuelo.loc[dict(event="despegue")] += split_window
        vuelo.loc[dict(event="aterrizaje")] -= split_window
        offset_flight = trim_analysis_window(daData, vuelo).mean(dim="time")

    else:
        # busca despegue y aterrizaje provisionales
        vuelo = detect_takeoff_landing(
            daData, jump_test, threshold=threshold
        )  # , show=show)
        # reduces the window a little to avoid possible bounces from filtering
        split_window = (
            (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
            * pct_window
            / 100
        ).astype("int32")
        vuelo.loc[dict(event="despegue")] += split_window
        vuelo.loc[dict(event="aterrizaje")] -= split_window

        offset_flight = trim_analysis_window(daData, vuelo).mean(dim="time")
        # trim_analysis_window(daData, vuelo).sel(axis='x').plot.line(x='time', col='ID', col_wrap=4)
        # offset_flight.sel(axis='z').plot.line(col='ID', col_wrap=4, hue='repe')
        # daData -= offset_flight
        with xr.set_options(keep_attrs=True):
            # datos = daData - offset_flight
            daData = daData - offset_flight

        if show:
            try:  # comprobar si es necesario cuando hay ejes
                trim_analysis_window(
                    daData, vuelo.sel(event=["despegue", "aterrizaje"])
                ).plot.line(col="ID", col_wrap=4, hue="axis", sharey=False)
            except:
                trim_analysis_window(
                    daData, vuelo.sel(event=["despegue", "aterrizaje"])
                ).plot.line(col="ID", col_wrap=4, hue="repe", sharey=False)

    return daData  # datos


def adjust_offset_s(
    daData: xr.DataArray,
    jump_test: str = "DJ",
    threshold: float = 20.0,
    pct_window: int = 5,
    show: bool = False,
) -> xr.DataArray:
    def _último_dato(data, ID):
        # print(ID)
        # plt.plot(data)

        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.nan

        return data[~np.isnan(data)][-1]

    """    
    data = daData[0,0].data
    """
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")
    daUltimo = xr.apply_ufunc(
        _último_dato,
        daData,
        daData.ID,
        input_core_dims=[["time"], []],
        # output_core_dims=[['weight']],
        # exclude_dims=set(('time',)),
        vectorize=True,
        # kwargs=dict(threshold=10, n_above=50, show=False)
    )
    # (daData - daUltimo).plot.line(x='time', col='ID')

    return daData - daUltimo


def reset_Fz_flight(
    daData: xr.DataArray,
    jump_test: str | None = None,
    threshold: float = 20.0,
    pct_window: int = 5,
    show: bool = False,
) -> xr.DataArray:  # , ventana_flight=None):
    """
    If jump_test=None, sets to zero below threshold
    """
    # daReturn = daData.where(daData > threshold, 0.0)
    if jump_test == "DJ2PApart":
        # Para plataforma auxiliar en DJ2Plats. Pone a cero después del despegue inicial
        vuelo = detect_takeoff_landing(
            daData, jump_test, threshold=threshold, show=show
        )

        def _detect_onset_aux(data, vuel, ID):
            if np.count_nonzero(~np.isnan(data)) == 0:
                return data
            # plt.plot(data)
            # plt.show()
            # print(ID, repe)
            data = data.copy()
            data[int(vuel) :] = 0.0
            return data

        """
        data = daData.sel(ID='S07_DJ_30', repe=1).data
        vuel = vuelo.sel(ID='S07_DJ_30', repe=1).data[0]
        """

        daReturn = xr.apply_ufunc(
            _detect_onset_aux,
            daData,
            vuelo.sel(event="despegue"),
            daData.ID,
            input_core_dims=[["time"], [], []],
            output_core_dims=[["time"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            # kwargs=dict(threshold=-threshold, n_above=int(0.1*daData.freq), show=show)
        ).drop_vars("event")

    else:  # if not DJ2PApart. Keeps nans
        daReturn = daData.where(daData.isnull(), daData.where(daData > threshold, 0.0))

    if show:
        if "plat" in daData.dims:
            if "repe" in daData.dims:
                daReturn.sel(axis="z").stack(ID_repe=("ID", "repe")).where(
                    daReturn.sel(axis="z").stack(ID_repe=("ID", "repe"))
                    < threshold * 2.5
                ).plot.line(x="time", col="ID_repe", hue="plat", col_wrap=4, alpha=0.7)
        else:
            daReturn.where(daReturn <= threshold * 2.1).plot.line(
                x="time", col="ID", col_wrap=4
            )

    return daReturn

    """
    #Con ufunc necesario tener despegue y aterrizaje
    def reset_ventana(data, ini, fin):
        dat=data.copy()
        #print(datos.shape, ini,fin)  
        ini=int(ini)
        fin=int(fin)
        dat[ini:fin] = np.full(fin-ini, 0.0)
        return dat
    
    
    # data = daData[0,1].sel(axis='z').data
    # ini = daEvents[0,1].sel(event='iniMov')
    # fin = daEvents[0,1].sel(event='finMov')
    
    
    daCortado = xr.apply_ufunc(reset_ventana, daData, ventana_flight.isel(event=0).sel(ID=daData.ID, repe=daData.repe), ventana_flight.isel(event=1).sel(ID=daData.ID, repe=daData.repe),
                   input_core_dims=[['time'], [], []],
                   output_core_dims=[['time']],
                   #exclude_dims=set(('time',)),
                   vectorize=True,
                   #join='outer'
                   ).dropna(dim='time', how='all')
    daCortado.attrs = daData.attrs
    daCortado.name = daData.name
    #daCortado.sel(axis='z').plot.line(x='time', row='ID', col='axis')
    return daCortado
    """


# TODO: CONTINUE TESTING THIS FUNCTION TO ADJUST THE X AND Y AXES AS WELL.
def reset_F_flight_axis(
    daData: xr.DataArray,
    jump_test: str = None,
    threshold: float = 20.0,
    pct_window: int = 5,
    show: bool = False,
) -> xr.DataArray:  # , ventana_flight=None):
    if "axis" in daData.dims:
        daDatosZ = daData.sel(axis="z")
    else:
        daDatosZ = daData

    vuelo = detect_takeoff_landing(daDatosZ, jump_test, threshold=threshold)
    # reduces the window a little to avoid possible bounces from filtering
    split_window = (
        (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
        * pct_window
        / 100
    ).astype("int32")
    vuelo.loc[dict(event="despegue")] += split_window
    vuelo.loc[dict(event="aterrizaje")] -= split_window

    with xr.set_options(keep_attrs=True):
        daData = xr.where(
            ~daData.isnull(), daData.where(daData > threshold, 0.0), daData
        )
        daData.time.attrs["units"] = "s"  # por alguna razón lo cambiaba a newtons
        # daData.plot.line(row='ID', col='repe', hue='axis', sharey=False)

    if show:
        trim_analysis_window(daData, vuelo + [-50, 50]).plot.line(
            col="ID", col_wrap=4, hue="axis", sharey=False
        )

    return daData


def reset_F_flight_axis_convencional(
    daData: xr.DataArray,
    jump_test: str = None,
    threshold: float = 20.0,
    pct_window: int = 5,
    show: bool = False,
) -> xr.DataArray:  # , ventana_flight=None):
    if "axis" in daData.dims:
        daDatosZ = daData.sel(axis="z")
    else:
        daDatosZ = daData

    vuelo = detect_takeoff_landing(daDatosZ, jump_test, threshold=threshold)
    # reduces the window a little to avoid possible bounces from filtering
    split_window = (
        (vuelo.loc[dict(event="aterrizaje")] - vuelo.loc[dict(event="despegue")])
        * pct_window
        / 100
    ).astype("int32")
    vuelo.loc[dict(event="despegue")] += split_window
    vuelo.loc[dict(event="aterrizaje")] -= split_window

    with xr.set_options(keep_attrs=True):
        daData = xr.where(
            ~daData.isnull(), daData.where(daData > threshold, 0.0), daData
        )
        daData.time.attrs["units"] = "s"  # por alguna razón lo cambiaba a newtons
        # daData.plot.line(row='ID', col='repe', hue='axis', sharey=False)

    if show:
        trim_analysis_window(daData, vuelo + [-50, 50]).plot.line(
            col="ID", col_wrap=4, hue="axis", sharey=False
        )

    return daData


def calculate_variables(
    daData: xr.DataArray, daWeight: xr.DataArray = None, daEvents: xr.DataArray = None
) -> xr.Dataset:
    """
    Calculates force / time related variables: v, s, P, RFD
    daEvents: receives the initial and final event for the calculation. You can pass
               iniMov/finMov (to avoid drifts) or iniAnalisis/finAnalisis for
               variable full plots.
    """

    daBW = daData / daWeight.sel(stat="media").drop_vars("stat")

    def _integrate(data, time, weight, ini, fin):
        # if np.count_nonzero(~np.isnan(data))==0:
        #     return np.nan
        dat = np.full(len(data), np.nan)
        try:
            ini = int(ini)
            fin = int(fin)
            # plt.plot(data[ini:fin])
            dat[ini:fin] = integrate.cumulative_trapezoid(
                data[ini:fin] - weight, time[ini:fin], initial=0
            )
            # plt.plot(dat)
        except Exception as e:
            print(f"Error calculando la integral. {e}")
            pass  # dat = np.full(len(data), np.nan)
        return dat

    """
    data = daData[2,0].data #.sel(axis='z').data
    time = daData.time.data
    weight=daWeight[2,0].sel(stat='media').data
    ini = daEvents[2,0].sel(event='iniMov').data
    fin = daEvents[2,0].sel(event='finMov').data
    plt.plot(data[int(ini):int(fin)])
    """
    """daV = (
        xr.apply_ufunc(
            _integra,
            daData,
            daData.time,
            daWeight.sel(stat="media"),
            daEvents.isel(event=0),
            daEvents.isel(
                event=1
            ),  # events 0 y 1 para que sirva con reversed, se pasa iniMov y finMov en el orden adecuado
            input_core_dims=[["time"], ["time"], [], [], []],
            output_core_dims=[["time"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            join="exact",
        )
        / (daWeight.sel(stat="media") / g)
    ).drop_vars("stat")
    """
    daV = (
        daData.biomxr.integrate_window(
            daEvents, daOffset=daWeight.sel(stat="media"), result_return="continuous"
        )
        / (daWeight.sel(stat="media") / g)
    ).drop_vars("stat")
    # daV.isel(ID=slice(None, 8)).plot.line(x='time', col='ID', col_wrap=4)

    daS = daV.biomxr.integrate_window(daEvents, result_return="continuous")
    """daS = xr.apply_ufunc(
        _integra,
        daV,
        daData.time,
        0,
        daEvents.isel(event=0),
        daEvents.isel(event=1),
        input_core_dims=[["time"], ["time"], [], [], []],
        output_core_dims=[["time"]],
        # exclude_dims=set(('time',)),
        vectorize=True,
    )
    """
    # daS.isel(ID=slice(None, 8)).plot.line(x='time', col='ID', col_wrap=4)

    daP = daData * daV
    daRFD = daData.differentiate(coord="time")

    # daV.attrs['units']='m/s'
    # daS.attrs['units']='m'
    # daP.attrs['units']='W'
    # daRFD.attrs['units']='N/s'

    daBW = daBW.assign_attrs({"freq": daData.freq, "units": "N/kg"})
    daV = daV.assign_attrs({"freq": daData.freq, "units": "m/s"})
    daS = daS.assign_attrs({"freq": daData.freq, "units": "m"})
    daP = daP.assign_attrs({"freq": daData.freq, "units": "W"})
    daRFD = daRFD.assign_attrs({"freq": daData.freq, "units": "N/s"})

    return (
        xr.Dataset(
            {"BW": daBW, "v": daV, "s": daS, "P": daP, "RFD": daRFD}  # F normalizada
        )
        .astype(daData.dtype)
        .assign_attrs({"freq": daData.freq})
    )


def calculate_results(
    daCinet: xr.DataArray = None,
    dsCinem: xr.Dataset = None,
    daWeight: xr.DataArray = None,
    daResults: xr.DataArray = None,
    daEvents: xr.DataArray = None,
) -> xr.DataArray:
    if not isinstance(daResults, xr.DataArray):
        daResults = (
            xr.full_like(daCinet.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {
                    "n_var": [
                        "tVuelo",
                        "tFaseInicioDesc",
                        "tFaseExc",
                        "tFaseConc",
                        "FzMax",
                        "FzMin",
                        "FzMaxCaida",
                        "FzTransicion",
                        "vDespegue",
                        "vAterrizaje",
                        "vMax",
                        "vMin",
                        "vMinCaida",
                        "sIniMov",
                        "sFinMov",
                        "sDespegue",
                        "sAterrizaje",
                        "sDifDespAter",
                        "sMax",
                        "sMin",
                        "sMinCaida",
                        "hTVuelo",
                        "hVDespegue",
                        "hS",
                        "PMax",
                        "PMin",
                        "PMinCaida",
                        "RFDMax",
                        "RFDMed",
                        "impNegDescenso",
                        "ImpPositDescenso",
                        "ImpPositAscenso",
                        "ImpNegAscenso",
                        "RSI",
                        "landingStiffness",
                        "tIniMov",
                        "tDespegue",
                        "tAterrizaje",
                        "tFzMax",
                        "tFzMin",
                        "tFzMaxCaida",
                        "tFzTransicion",
                        "tVMax",
                        "tVMin",
                        "tVMinCaida",
                        "tSMax",
                        "tSMin",
                        "tSMinCaida",
                        "tPMax",
                        "tPMin",
                        "tPMinCaida",
                        "tRFDMax",
                    ]
                },
                axis=-1,
            )
        ).copy()
    daResults.name = "results"
    del daResults.attrs["freq"]
    del daResults.attrs["units"]
    if "freq_ref" in daResults.attrs:
        del daResults.attrs["freq_ref"]

    if "axis" in daCinet.dims:
        daCinet = daCinet.sel(axis="z")  # en principio solo interesa el eje z

    dsBatida = trim_analysis_window(
        dsCinem[["BW", "v", "s", "P", "RFD"]],
        daEvents.sel(event=["iniMov", "despegue"]),
    )
    dsCaida = trim_analysis_window(
        dsCinem[["BW", "v", "s", "P", "RFD"]],
        daEvents.sel(event=["aterrizaje", "finMov"]),
    )

    # Tiempos de fase
    daResults.loc[dict(n_var="tFaseInicioDesc")] = (
        daEvents.sel(event="iniImpPos") - daEvents.sel(event="iniMov")
    ) / dsCinem.freq
    daResults.loc[dict(n_var="tFaseExc")] = (
        daEvents.sel(event="maxFlex") - daEvents.sel(event="iniImpPos")
    ) / dsCinem.freq
    daResults.loc[dict(n_var="tFaseConc")] = (
        daEvents.sel(event="despegue") - daEvents.sel(event="maxFlex")
    ) / dsCinem.freq
    daResults.loc[dict(n_var="tVuelo")] = (
        daEvents.sel(event="aterrizaje") - daEvents.sel(event="despegue")
    ) / dsCinem.freq

    # Fuerzas batida
    daResults.loc[dict(n_var="FzMax")] = dsBatida["BW"].max(
        dim="time"
    )  # trim_analysis_window(daCinet, daEvents.sel(event=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(n_var="FzMin")] = trim_analysis_window(
        dsCinem["BW"],
        daEvents.sel(event=["iniMov", "maxFz"]),
    ).min(
        dim="time"
    )  # trim_analysis_window(daCinet, daEvents.sel(event=['iniMov', 'despegue'])).min(dim='time')
    daResults.loc[dict(n_var="FzTransicion")] = daCinet.sel(
        time=daEvents.sel(event="maxFlex") / dsCinem.freq, method="nearest"
    )

    # Fuerzas caída
    daResults.loc[dict(n_var="FzMaxCaida")] = dsCaida["BW"].max(dim="time")

    # Velocidades
    daResults.loc[dict(n_var="vDespegue")] = dsCinem["v"].sel(
        time=daEvents.sel(event="despegue") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="vAterrizaje")] = dsCinem["v"].sel(
        time=daEvents.sel(event="aterrizaje") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="vMax")] = dsBatida["v"].max(dim="time")
    daResults.loc[dict(n_var="vMin")] = dsBatida["v"].min(dim="time")
    daResults.loc[dict(n_var="vMinCaida")] = dsCaida["v"].min(dim="time")

    # Posiciones
    daResults.loc[dict(n_var="sIniMov")] = dsCinem["s"].sel(
        time=daEvents.sel(event="iniMov") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sFinMov")] = dsCinem["s"].sel(
        time=(daEvents.sel(event="finMov") - 1) / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sDespegue")] = dsCinem["s"].sel(
        time=daEvents.sel(event="despegue") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sAterrizaje")] = dsCinem["s"].sel(
        time=daEvents.sel(event="aterrizaje") / dsCinem.freq, method="nearest"
    )
    daResults.loc[dict(n_var="sDifDespAter")] = (
        daResults.loc[dict(n_var="sDespegue")]
        - daResults.loc[dict(n_var="sAterrizaje")]
    )
    daResults.loc[dict(n_var="sMax")] = trim_analysis_window(
        dsCinem["s"], daEvents.sel(event=["despegue", "aterrizaje"])
    ).max(dim="time")
    daResults.loc[dict(n_var="sMin")] = dsBatida["s"].min(dim="time")
    daResults.loc[dict(n_var="sMinCaida")] = dsCaida["s"].min(dim="time")

    # Altura salto
    daResults.loc[dict(n_var="hTVuelo")] = (
        g / 8 * daResults.loc[dict(n_var="tVuelo")] ** 2
    )
    daResults.loc[dict(n_var="hVDespegue")] = daResults.loc[
        dict(n_var="vDespegue")
    ] ** 2 / (2 * g)
    daResults.loc[dict(n_var="hS")] = (
        daResults.loc[dict(n_var="sMax")] - daResults.loc[dict(n_var="sDespegue")]
    )

    # Potencias
    daResults.loc[dict(n_var="PMax")] = dsBatida["P"].max(
        dim="time"
    )  # trim_analysis_window(dsCinem['P'], daEvents.sel(event=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(n_var="PMin")] = dsBatida["P"].min(
        dim="time"
    )  # trim_analysis_window(dsCinem['P'], daEvents.sel(event=['iniMov', 'despegue'])).min(dim='time')
    daResults.loc[dict(n_var="PMinCaida")] = dsCaida["P"].min(dim="time")

    # RFD
    daResults.loc[dict(n_var="RFDMax")] = dsBatida["RFD"].max(
        dim="time"
    )  # trim_analysis_window(dsCinem['RFD'], daEvents.sel(event=['iniMov', 'despegue'])).max(dim='time')
    daResults.loc[dict(n_var="RFDMed")] = (
        daCinet.sel(time=daEvents.sel(event="maxFlex") / dsCinem.freq, method="nearest")
        - daCinet.sel(time=daEvents.sel(event="minFz") / dsCinem.freq, method="nearest")
    ) / ((daEvents.sel(event="maxFlex") - daEvents.sel(event="minFz")) / dsCinem.freq)

    # ---- Impulsos. Como la fuerza viene en BW, el peso que resta es 1. Con fuerza en newtons restar daWeight.sel(stat='media').drop_vars('stat')
    from biomdp.general_processing_functions import integrate_window

    daResults.loc[dict(n_var="impNegDescenso")] = integrate_full(
        daCinet - 1, daEvents=daEvents.sel(event=["iniMov", "iniImpPos"])
    )
    daResults.loc[dict(n_var="ImpPositDescenso")] = integrate_full(
        daCinet - 1, daEvents=daEvents.sel(event=["iniImpPos", "maxFlex"])
    )
    daResults.loc[dict(n_var="ImpPositAscenso")] = integrate_full(
        daCinet - 1, daEvents=daEvents.sel(event=["maxFlex", "finImpPos"])
    )
    daResults.loc[dict(n_var="ImpNegAscenso")] = integrate_full(
        daCinet - 1, daEvents=daEvents.sel(event=["finImpPos", "despegue"])
    )

    # ---- Tiempos de events clave
    daResults.loc[dict(n_var="tIniMov")] = (
        dsCinem["events"].sel(event="iniMov").data[0] / dsCinem.freq
    )
    daResults.loc[dict(n_var="tDespegue")] = (
        dsCinem["events"].sel(event="despegue").data[0] / dsCinem.freq
    )
    daResults.loc[dict(n_var="tAterrizaje")] = (
        dsCinem["events"].sel(event="aterrizaje").data[0] / dsCinem.freq
    )
    daResults.loc[dict(n_var="tFzMax")] = (
        dsBatida["BW"].biomxr.nanargmax_xr(dim="time") / dsCinem.freq
    )

    daResults.loc[dict(n_var="tFzMin")] = (
        trim_analysis_window(
            dsCinem["BW"],
            daEvents.sel(event=["iniMov", "maxFz"]),
        ).biomxr.nanargmin_xr(dim="time")
        / dsCinem.freq
    )

    daResults.loc[dict(n_var="tFzMaxCaida")] = (
        dsCaida["BW"].biomxr.nanargmax_xr(dim="time")
        + daEvents.sel(event="aterrizaje")
        - daEvents.sel(event="iniMov")
    ) / dsCinem.freq

    daResults.loc[dict(n_var="tFzTransicion")] = (
        daEvents.sel(event="maxFlex") - daEvents.sel(event="iniMov")
    ) / dsCinem.freq

    # from biomdp.general_processing_functions import nanargmax_xr, nanargmin_xr

    daResults.loc[dict(n_var="tVMax")] = (
        dsBatida["v"].biomxr.nanargmax_xr(dim="time") / dsCinem.freq
    )
    daResults.loc[dict(n_var="tVMin")] = (
        dsBatida["v"].biomxr.nanargmin_xr(dim="time") / dsCinem.freq
    )

    daResults.loc[dict(n_var="tSMax")] = (
        trim_analysis_window(
            dsCinem["s"],
            daEvents.sel(event=["iniMov", "aterrizaje"]),
        ).biomxr.nanargmax_xr(dim="time")
        / dsCinem.freq
    )
    daResults.loc[dict(n_var="tSMin")] = (
        dsBatida["s"].biomxr.nanargmin_xr(dim="time") / dsCinem.freq
    )
    daResults.loc[dict(n_var="tSMinCaida")] = (
        dsCaida["s"].biomxr.nanargmin_xr(dim="time")
        + daEvents.sel(event="aterrizaje")
        - daEvents.sel(event="iniMov")
    ) / dsCinem.freq

    daResults.loc[dict(n_var="tPMax")] = (
        dsBatida["P"].biomxr.nanargmax_xr(dim="time") / dsCinem.freq
    )
    daResults.loc[dict(n_var="tPMin")] = (
        dsBatida["P"].biomxr.nanargmin_xr(dim="time") / dsCinem.freq
    )

    daResults.loc[dict(n_var="tRFDMax")] = (
        dsBatida["RFD"].biomxr.nanargmax_xr(dim="time") / dsCinem.freq
    )

    # ---- Otras
    # TODO: EN POSICIÓN MÁS BAJA CAÍDA, DIVIDIR Fz / DESPL DESDE CONTACTO A MIN S
    # TODO: AÑADIR ALTURA EN ATERRIZAJE AL SMINCAIDA
    FzSMin = daResults.loc[dict(n_var="FzMin")]
    tSMin = daResults.loc[dict(n_var="tSMinCaida")]
    # TODO: COMPROBAR QUE COINCIDE EN EL TIEMPO FZMAX CON SMIN
    daResults.loc[dict(n_var="landingStiffness")] = dsCaida["BW"].max(dim="time") / (
        abs(daResults.loc[dict(n_var="sMinCaida")])
        + daResults.loc[dict(n_var="sAterrizaje")]
    )

    # TODO: ALTURA DIVIDIDA ENTRE TIEMPO BATIDA. DIFERENTE EN CMJ Y DJ
    daResults.loc[dict(n_var="RSI")] = daResults.loc[dict(n_var="hVDespegue")] / (
        daEvents.sel(event=["iniMov", "despegue"]).diff(dim="event") / dsCinem.freq
    ).squeeze("event")

    return daResults


def results_to_table(daResults: xr.DataArray, dsSalto: xr.Dataset) -> pd.DataFrame:
    # TODO: los datos en blanco con nan o '--'?
    dfResultsForces = (
        daResults.isel(ID=0).to_dataframe()  # .transpose('n_var') #'modalidad', 'repe'
        # .reset_index()
        # .set_index(['modalidad', 'n_var', 'ID', 'repe'])
    )

    tabla_results = pd.DataFrame(
        columns=["Valor", "Instante (s)"],
        index=[
            "Inicio Salto",
            "Despegue",
            "Aterrizaje",
            "Máxima fuerza vertical en la batida (BW)",
            "Máxima potencia en la batida (W/BW)",
            "Tiempo de vuelo (s)",
            "Velocidad despegue (m/s)",
            "Desplazamiento CG despegue (m)",
            "Altura según v despegue (m)",
            "Altura según desplazamiento CG (m)",
            "Altura según tiempo de vuelo (m)",
            "Máxima fuerza vertical en la caída (BW)",
        ],
    )

    tabla_results.loc["Inicio Salto", :] = [
        np.nan,  # "--",
        (daResults.sel(n_var="tIniMov").data[0]),  # .round(3),
    ]

    tabla_results.loc["Despegue", :] = [
        np.nan,  # "--",
        daResults.sel(n_var=["tIniMov", "tDespegue"])
        .diff("n_var")[0]
        .data[0],  # .round(3),
    ]

    tabla_results.loc["Aterrizaje", :] = [
        np.nan,  # "--",
        daResults.sel(n_var=["tIniMov", "tAterrizaje"])
        .diff("n_var")[0]
        .data[0],  # .round(3),
    ]

    tabla_results.loc["Máxima fuerza vertical en la batida (BW)", :] = daResults.sel(
        n_var=["FzMax", "tFzMax"]
    )  # .round(3)

    tabla_results.loc["Máxima potencia en la batida (W/BW)", :] = daResults.sel(
        n_var=["PMax", "tPMax"]
    )  # .round(3)

    tabla_results.loc["Tiempo de vuelo (s)", :] = [
        daResults.sel(n_var="tVuelo").data[0],  # .round(3),
        np.nan,  # "--",
    ]

    tabla_results.loc["Velocidad despegue (m/s)", :] = [
        daResults.sel(n_var="vDespegue").data[0],  # .round(3),
        daResults.sel(n_var=["tIniMov", "tDespegue"]).diff("n_var")[0].data[0],
        # (dsSalto["events"].sel(event="despegue").data[0] / dsSalto.freq),  # .round(3),
    ]

    tabla_results.loc["Desplazamiento CG despegue (m)", :] = [
        daResults.sel(n_var="sDespegue").data[0],  # .round(3),
        daResults.sel(n_var=["tIniMov", "tDespegue"]).diff("n_var")[0].data[0],
        # (dsSalto["events"].sel(event="despegue").data[0] / dsSalto.freq),  # .round(3),
    ]

    tabla_results.loc["Altura según v despegue (m)", :] = [
        daResults.sel(n_var="hVDespegue").data[0],  # , "tDespegue"]
        np.nan,
    ]

    tabla_results.loc["Altura según desplazamiento CG (m)", :] = [
        daResults.sel(n_var="hS").data[0],
        np.nan,
    ]  # .round(3)  # "--",

    tabla_results.loc["Altura según tiempo de vuelo (m)", :] = [
        daResults.sel(n_var="hTVuelo").data[0],  # .round(3),
        np.nan,  # daResults.sel(n_var=["tIniMov", "tSMax"]).diff("n_var")[0].data[0],  # "--",
    ]

    tabla_results.loc["Máxima fuerza vertical en la caída (BW)", :] = daResults.sel(
        n_var=["FzMaxCaida", "tFzMaxCaida"]
    )  # .round(3)

    # Redondea decimales
    tabla_results = tabla_results.astype(float).round({"Valor": 2, "Instante (s)": 3})

    return tabla_results


def calculate_results_EMG(
    daEMG: xr.DataArray = None,
    daResults: xr.DataArray = None,
    daEvents: xr.DataArray = None,
) -> xr.DataArray:
    if not isinstance(daResults, xr.DataArray):
        if "axis" in daEvents.coords:
            daEvents = daEvents.drop_vars("axis")
        daResults = (
            xr.full_like(
                daEMG.isel(time=0).drop_vars(["time", "axis"]), np.nan
            ).expand_dims(
                {
                    "n_var": [
                        "EMGPreIniMean",
                        "EMGExcMean",
                        "EMGConcMean",
                        "EMGVueloMean",
                        "EMGPreIniInteg",
                        "EMGExcInteg",
                        "EMGConcInteg",
                        "EMGVueloInteg",
                        "EMGPreIniRMS",
                        "EMGExcRMS",
                        "EMGConcRMS",
                        "EMGVueloRMS",
                    ]
                },
                axis=-1,
            )
        ).copy()

    daResults.name = "results"
    del daResults.attrs["freq"]
    del daResults.attrs["freq_ref"]
    del daResults.attrs["units"]

    # Medias
    daResults.loc[dict(n_var="EMGPreIniMean")] = trim_analysis_window(
        daData=daEMG, daEvents=daEvents.sel(event=["preactiv", "iniMov"])
    ).mean("time")
    daResults.loc[dict(n_var="EMGExcMean")] = trim_analysis_window(
        daEMG, daEvents.sel(event=["iniMov", "maxFlex"])
    ).mean("time")
    daResults.loc[dict(n_var="EMGConcMean")] = trim_analysis_window(
        daEMG, daEvents.sel(event=["maxFlex", "despegue"])
    ).mean("time")
    daResults.loc[dict(n_var="EMGVueloMean")] = trim_analysis_window(
        daEMG, daEvents.sel(event=["despegue", "aterrizaje"])
    ).mean("time")

    # Integrales
    daResults.loc[dict(n_var="EMGPreIniInteg")] = integrate_full(
        daEMG, daEvents=daEvents.sel(event=["preactiv", "iniMov"])
    )
    daResults.loc[dict(n_var="EMGExcInteg")] = integrate_full(
        daEMG, daEvents=daEvents.sel(event=["iniMov", "maxFlex"])
    )
    daResults.loc[dict(n_var="EMGConcInteg")] = integrate_full(
        daEMG, daEvents=daEvents.sel(event=["maxFlex", "despegue"])
    )
    daResults.loc[dict(n_var="EMGVueloInteg")] = integrate_full(
        daEMG, daEvents=daEvents.sel(event=["despegue", "aterrizaje"])
    )

    # RMS
    from biomdp.general_processing_functions import RMS

    daResults.loc[dict(n_var="EMGPreIniRMS")] = RMS(
        daData=daEMG, daWindow=daEvents.sel(event=["preactiv", "iniMov"])
    )
    daResults.loc[dict(n_var="EMGExcRMS")] = RMS(
        daData=daEMG, daWindow=daEvents.sel(event=["iniMov", "maxFlex"])
    )
    daResults.loc[dict(n_var="EMGConcRMS")] = RMS(
        daEMG, daWindow=daEvents.sel(event=["maxFlex", "despegue"])
    )
    daResults.loc[dict(n_var="EMGVueloRMS")] = RMS(
        daEMG, daWindow=daEvents.sel(event=["despegue", "aterrizaje"])
    )

    return daResults


# =============================================================================
# TEST INSIDE A CLASS
"""
Clase con funciones para tratar fuerzas de saltos desde archivos de plataforma
de fuerzas.
"""


class jump_forces_utils:
    def __init__(
        self,
        data: xr.DataArray = xr.DataArray(),
        jump_test: str = "CMJ",
        events: xr.DataArray | None = None,
    ):
        self.data = data
        self.jump_test = jump_test
        self.events = (
            xr.full_like(self.data.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {"event": BASIC_EVENTS}, axis=-1
            )
        ).copy()
        self.weight = None

    def load_preprocessed(self, work_path, n_preprocessed_file):
        if Path((work_path / (n_preprocessed_file)).with_suffix(".nc")).is_file():
            tpo = time.time()
            self.data = xr.load_dataarray(
                (work_path / (n_preprocessed_file)).with_suffix(".nc")
            ).sel(tipo=self.jump_test)
            print(
                "\nLoading preprocessed file ",
                n_preprocessed_file
                + "_Vicon.nc en {0:.3f} s.".format(time.time() - tpo),
            )
        else:
            raise Exception("Vicon preprocessed file not found")

    def calculate_weight(self, window=[100, 600], show=False):
        self.weight = (
            self.data.sel(axis="z")
            .isel(time=slice(window[0], window[1]))
            .mean(dim="time")
        )

        if show:

            def plot_weight(x, y, **kwargs):  # provisional
                print(x)  # kwargs['data'])
                # plt.plot()

            g = (
                self.data.sel(axis="z")
                .stack(ID_plate=("ID", "plate"))
                .plot.line(col="ID_plate", col_wrap=4, sharey=False)
            )
            # g = xr.plot.FacetGrid(self.data, col='ID', col_wrap=4)
            # g.map_dataarray(dibuja_peso, x='time', y=None)#, y='trial')

            for h, ax in enumerate(g.axes):  # extrae cada fila
                for i in range(len(ax)):  # extrae cada axis (gráfica)
                    try:
                        idn = g.data.loc[g.name_dicts[h, i]].ID
                        # print('weight=', self.weight.sel(ID=idn).data)#idn)
                        # Mean weight range
                        # ax[i].axvspan(g.data.time[int(window[0]*self.data.freq)], g.data.time[int(window[1]*self.data.freq)], alpha=0.2, color='C1')
                        ax[i].axvspan(
                            (len(self.data.time) + window[0]) / self.data.freq,
                            (len(self.data.time) + window[1]) / self.data.freq,
                            alpha=0.2,
                            color="C1",
                        )
                        # Weight lines
                        ax[i].hlines(
                            self.weight.sel(ID=idn).data,
                            xmin=self.data.time[0],
                            xmax=self.data.time[-1],
                            colors=["C0", "C1", "C2"],
                            lw=1,
                            ls="--",
                            alpha=0.6,
                        )
                    except:
                        print("Error in", h, i)


# =============================================================================


# =============================================================================
# %% PRUEBAS
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr

    from biomdp.filter_butter import filter_butter
    from biomdp.io import read_kistler_txt

    work_path = Path(r"src\datasets")
    file = work_path / "kistler_CMJ_1plate.txt"

    daCMJ = read_kistler_txt(file)
    daCMJ.plot.line(x="time", col="plate")

    daCMJ = daCMJ.expand_dims("ID").assign_coords(ID=["S01_CMJ_1"])

    daCMJ = daCMJ.sel(axis="z")
    # Filtra
    daCMJ = filter_butter(dat_orig=daCMJ, fr=daCMJ.freq, fc=400)

    # Offsets flight adjustment, even before weight---------------------------
    daCMJ = adjust_offsetFz(
        daData=daCMJ,
        jump_test="CMJ",
        kind="flight",
        threshold=40,
        pct_window=5,
    )  # , show=True)

    # Replacewith zero the flight  window-------------
    daCMJ = reset_Fz_flight(
        daData=daCMJ, jump_test="CMJ", threshold=30, pct_window=5
    )  # , show=True)

    daEventsForces = (
        xr.full_like(daCMJ.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": BASIC_EVENTS}, axis=-1
        )
    ).copy()

    # Estimates the start and end of the analysis adjustment
    daEventsForces.loc[dict(event=["iniAnalisis", "finAnalisis"])] = (
        guess_iniend_analysis(
            daData=daCMJ,
            jump_test="CMJ",
            daEvents=daEventsForces.sel(event=["iniAnalisis", "finAnalisis"]),
            window=[1.5, 1.2],
            threshold=30,
        )
    )

    # ----Calculate weight
    # Windows for calculating the weight of each jump
    daEventsForces.loc[dict(event=["iniPeso", "finPeso"])] = (
        np.array([0.0, 0.5]) * daCMJ.freq
    )

    # First calculate the weight of the average of the selected stable zone
    daWeight_mean = calculate_weight(
        daData=daCMJ,
        weight_window=daEventsForces.sel(event=["iniPeso", "finPeso"]),
    )  # , show=True)

    # To fine-tune the weight, it first detects tentative events
    daEventsForces = detect_standard_events(
        daData=daCMJ,
        daEvents=daEventsForces,
        daWeight=daWeight_mean,
        jump_test="CMJ",
        threshold=30.0,
    )
    daWeightCMJ = finetune_weight(
        daData=daCMJ,
        daWeight=daWeight_mean,
        daEvents=daEventsForces,
        kind="iter",
    )  # , show=True)

    # =============================================================================
    # TEST AS CLASS
    # =============================================================================

    r"""work_path = Path(r"src\datasets")
    file = work_path / "kistler_CMJ_1plate.txt"
    daCMJ = (
        read_kistler_txt(file)
        .expand_dims("ID")
        .assign_coords(ID=["S01_CMJ_1"])
    )

    cmj = jump_forces_utils(daCMJ, jump_test="CMJ")

    cmj.calculate_weight()
    cmj.data
    cmj.jump_test

    cmj.calculate_weight(window=[-1500, -1000], show=True)
    cmj.weight.sel(ID="01", trial="1")
    """

# %%
