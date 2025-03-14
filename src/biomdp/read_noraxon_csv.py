# -*- coding: utf-8 -*-
"""
Created on Fry Mar 8 13:15:58 2024

@author: jose.lopeze

Reading of .csv files exported from Noraxon with IMU.
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.1"
__date__ = "13/03/2025"

"""
Updates:
    13/03/2025, v0.1.1
        - Adapted to biomdp with translations.

    08/03/2024, v0.1.0
        - Empezado tomando trozos sueltos.
            
"""

from typing import List
import numpy as np
import pandas as pd
import xarray as xr
import polars as pl

# import matplotlib.pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages #para guardar gráficas en pdf
# import seaborn as sns

from pathlib import Path
import time  # para cuantificar tiempos de procesado


# import sys
# sys.path.append('F:\Programacion\Python\Mios\Functions')
# #sys.path.append('G:\Mi unidad\Programacion\Python\Mios\Functions')

# from readViconCsv import read_vicon_csv


# =============================================================================
# %% Processing functions
# =============================================================================


def split_dim_axis(da: xr.DataArray) -> xr.DataArray:
    # Separa el xarray en ejes creando dimensión axis
    if "Accel " in da.n_var.to_series().iloc[0]:
        sensor_type = "Accel "
    elif "Gyro " in da.n_var.to_series().iloc[0]:
        sensor_type = "Gyro "
    elif "Mag " in da.n_var.to_series().iloc[0]:
        sensor_type = "Mag "
    else:
        raise Exception('Sensor type not found, try "Accel", "Gyro" or "Mag"')

    x = da.sel(
        n_var=da.n_var.str.endswith(f"{sensor_type[0]}x")
    )  # .rename({"n_var": "axis"})
    x = x.assign_coords(
        n_var=[
            s[:-3] for s in x.n_var.to_series()
        ]  # [s.split(sensor_type)[1][:-3] for s in x.n_var.to_series()]
    )

    y = da.sel(
        n_var=da.n_var.str.endswith(f"{sensor_type[0]}y")
    )  # .rename({"n_var": "axis"})
    y = y.assign_coords(n_var=[s[:-3] for s in y.n_var.to_series()])

    z = da.sel(
        n_var=da.n_var.str.endswith(f"{sensor_type[0]}z")
    )  # .rename({"n_var": "axis"})
    z = z.assign_coords(n_var=[s[:-3] for s in z.n_var.to_series()])

    da = (
        xr.concat([x, y, z], dim="axis")
        # .assign_coords(n_var="plat1")
        .assign_coords(axis=["x", "y", "z"])
        # .expand_dims({"n_var": 1})
    )

    return da


def split_dim_side(da: xr.DataArray) -> xr.DataArray:
    # Separa el xarray en ejes creando dimensión axis

    L = da.sel(n_var=da.n_var.str.contains("izquierda"))
    L = L.assign_coords(n_var=[s.split(" izquierda")[0] for s in L.n_var.to_series()])

    R = da.sel(n_var=da.n_var.str.contains("derecha"))  # .rename({"n_var": "axis"})
    R = R.assign_coords(n_var=[s.split(" derecha")[0] for s in R.n_var.to_series()])

    da = (
        xr.concat([L, R], dim="side")
        # .assign_coords(n_var="plat1")
        .assign_coords(side=["L", "R"])
        # .expand_dims({"n_var": 1})
    )

    return da


def assign_subcategories_xr(
    da: xr.DataArray, n_project: str | None = None
) -> xr.DataArray:

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


def df_to_da(
    dfAll: pl.DataFrame | pd.DataFrame, n_project: str | None = None, show: bool = False
) -> xr.DataArray:
    if isinstance(dfAll, pl.DataFrame):
        # Transforma df polars a dataarray con todas las variables cargadas
        vars_leidas = dfAll.select(
            pl.exclude(
                ["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
            ),
        ).columns

        dfpd = dfAll.melt(
            id_vars=["ID", "time"], value_vars=vars_leidas, variable_name="n_var"
        ).to_pandas()

    else:  # from pandas
        vars_leidas = dfAll.drop(
            columns=["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
        ).columns
        dfpd = dfAll.drop(
            columns=["estudio", "tipo", "subtipo", "particip", "repe"]
        ).melt(id_vars=["ID", "time"], var_name="n_var")

    daAll = (
        # dfpd.drop(columns=["estudio", "tipo", "subtipo", "particip", "repe"])
        dfpd  # .melt(id_vars=["ID", "time"], var_name="n_var")
        # pd.melt(dfAllArchivos.to_pandas().drop(columns=['estudio','tipo','subtipo']), id_vars=['ID', 'repe', 'time'], var_name='axis')
        .set_index(["ID", "n_var", "time"])
        .to_xarray()
        .to_array()
        .squeeze("variable")
        .drop_vars("variable")
    )

    # Assign extra coordinates
    daAll = assign_subcategories_xr(da=daAll, n_project=n_project)
    # daAll = daAll.assign_coords(estudio=('ID', dfAll.filter(pl.col('time')==0.000).get_column('estudio').to_list()),
    #                                          particip=('ID', dfAll.filter(pl.col('time')==0.000).get_column('particip').to_list()),
    #                                          tipo=('ID', dfAll.filter(pl.col('time')==0.000).get_column('tipo').to_list()),
    #                                          subtipo=('ID', dfAll.filter(pl.col('time')==0.000).get_column('subtipo').to_list()),
    #                                          repe=('ID', dfAll.filter(pl.col('time')==0.000).get_column('repe').to_list()),
    #                                          )
    # Ajusta tipo de coordenada tiempo, necesario??
    ###########daAllArchivos = daAllArchivos.assign_coords(time=('time', daAllArchivos.time.astype('float32').values))

    # daAllArchivos.sel(ID='PCF_SCT05', axis='z').plot.line(x='time', col='repe')
    # daAllArchivos.assign_coords(time=daAllArchivos.time.astype('float32'))

    # daAll.attrs = {
    #     "freq": (np.round(1 / (daAll.time[1] - daAll.time[0]), 1)).data,
    #     "units": "N",
    # }
    daAll.attrs["freq"] = (np.round(1 / (daAll.time[1] - daAll.time[0]), 1)).data
    daAll.time.attrs["units"] = "s"

    return daAll


def read_noraxon_pd(
    file: str | Path, n_vars_load: List[str] | None = None, to_dataarray: bool = False
) -> xr.DataArray | pl.DataFrame:
    # file = Path(r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")
    df = pd.read_csv(
        file, skiprows=3, header=0, engine="c"
    ).drop(  # .astype(np.float64)
        columns=["Activity", "Marker"]
    )

    # df = df.drop(columns=df.filter(regex="(Normal)") + df.filter(regex="(Tiempo.)"))

    #       df.filter(regex="(Normal)").values)).drop(columns=df.filter(regex="(Tiempo.)"))
    # df.filter(regex="(Normal)")+df.filter(regex="(Tiempo.)")
    # df.dropna(axis="columns", how="all")

    # df.filter(regex="Tiempo.")

    if to_dataarray:
        print("_to_dataarray not implemented yet. Try using read_noraxon_pl")
        da = xr.DataArray()
        return da

    return df


def read_noraxon_pl(
    file: str | Path, n_vars_load: List[str] | None = None, to_dataarray: bool = False
) -> xr.DataArray | pl.DataFrame:
    # file = Path(r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")

    df = (
        pl.read_csv(
            file,
            has_header=True,
            skip_rows=3,
            # skip_rows_after_header=0,
            columns=n_vars_load,
            # separator=",",
        )
        # .select(pl.exclude("^.*_duplicated_.*$"))  # quita columnas de tiempo duplicadas
        .select(pl.exclude(pl.String))  # quita columnas de texto con datos (Normal)
        # .with_columns(pl.all().cast(pl.Float64()))
    )

    """
    cadera = df.select(
            pl.col("^*Flexo-extensión cadera .*::Y$")
        )  # .to_numpy()
    """
    # df = df.to_pandas()

    # ----Transform polars to xarray
    if to_dataarray:

        # # Separa ejes articulares
        # x = df.select(pl.col("^Flexo-extensión.*$")).to_numpy()
        # y = df.select(pl.col("^Aducción-abducción.*$")).to_numpy()
        # z = df.select(pl.col("^Rotación.*$")).to_numpy()
        # data = np.stack([x, y, z])

        freq = 1 / (df[1, "time"] - df[0, "time"])

        # coords = {
        #     "axis": ["x", "y", "z"],
        #     "time": np.arange(data.shape[1]) / freq,
        #     "n_var": ["Force"],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        # }
        coords = {
            "time": np.arange(df.shape[0]) / freq,
            "n_var": df.columns,  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        }
        da = (
            xr.DataArray(
                data=df.to_numpy(),
                dims=coords.keys(),
                coords=coords,
            )
            # .astype(float)
            # .transpose("n_var", "time")
        )
        da.name = "EMG"
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        da.attrs["units"] = "mV"

        return da

    return df


def load_merge_noraxon_csv(
    path: str | Path,
    section: str = "EMG",
    n_project: str | None = None,
    data_type: str | None = None,
    show: bool = False,
) -> xr.DataArray:
    """
    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    n_vars_load : TYPE, optional
        DESCRIPTION. The default is None.
    n_project : string, optional
        DESCRIPTION. The name of the study.
    data_type:
        Conversión al tipo de datos indicado. Por defecto es None, que quiere
        decir que se mantiene el tipo original ('float64')
    show : bool, optional
        DESCRIPTION. The default is False.


    Returns
    -------
    daAllArchivos : xarray DataArray
        DESCRIPTION. DataArray with the loaded files concatenated.


    """
    if data_type is None:
        data_type = float

    files_list = sorted(
        list(path.glob("*.csv"))  # "**/*.csv"
    )  #'**/*.txt' incluye los que haya en subcarpetas
    files_list = [
        x
        for x in files_list
        if "error" not in x.name and "info" not in x.name and "_error" not in x.name
    ]  # selecciona archivos

    all_vars = [
        "Ultium EMG.EMG 1 (uV)",
        "Ultium EMG.Internal Accel 1 Ax (mG)",
        "Ultium EMG.Internal Accel 1 Ay (mG)",
        "Ultium EMG.Internal Accel 1 Az (mG)",
        "Ultium EMG.Internal Gyro 1 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 1 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 1 Gz (deg/s)",
        "Ultium EMG.Internal Mag 1 Mx (mGauss)",
        "Ultium EMG.Internal Mag 1 My (mGauss)",
        "Ultium EMG.Internal Mag 1 Mz (mGauss)",
        "Ultium EMG.EMG 2 (uV)",
        "Ultium EMG.Internal Accel 2 Ax (mG)",
        "Ultium EMG.Internal Accel 2 Ay (mG)",
        "Ultium EMG.Internal Accel 2 Az (mG)",
        "Ultium EMG.Internal Gyro 2 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 2 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 2 Gz (deg/s)",
        "Ultium EMG.Internal Mag 2 Mx (mGauss)",
        "Ultium EMG.Internal Mag 2 My (mGauss)",
        "Ultium EMG.Internal Mag 2 Mz (mGauss)",
        "Ultium EMG.EMG 3 (uV)",
        "Ultium EMG.Internal Accel 3 Ax (mG)",
        "Ultium EMG.Internal Accel 3 Ay (mG)",
        "Ultium EMG.Internal Accel 3 Az (mG)",
        "Ultium EMG.Internal Gyro 3 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 3 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 3 Gz (deg/s)",
        "Ultium EMG.Internal Mag 3 Mx (mGauss)",
        "Ultium EMG.Internal Mag 3 My (mGauss)",
        "Ultium EMG.Internal Mag 3 Mz (mGauss)",
        "Ultium EMG.EMG 4 (uV)",
        "Ultium EMG.Internal Accel 4 Ax (mG)",
        "Ultium EMG.Internal Accel 4 Ay (mG)",
        "Ultium EMG.Internal Accel 4 Az (mG)",
        "Ultium EMG.Internal Gyro 4 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 4 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 4 Gz (deg/s)",
        "Ultium EMG.Internal Mag 4 Mx (mGauss)",
        "Ultium EMG.Internal Mag 4 My (mGauss)",
        "Ultium EMG.Internal Mag 4 Mz (mGauss)",
        "Ultium EMG.EMG 5 (uV)",
        "Ultium EMG.Internal Accel 5 Ax (mG)",
        "Ultium EMG.Internal Accel 5 Ay (mG)",
        "Ultium EMG.Internal Accel 5 Az (mG)",
        "Ultium EMG.Internal Gyro 5 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 5 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 5 Gz (deg/s)",
        "Ultium EMG.Internal Mag 5 Mx (mGauss)",
        "Ultium EMG.Internal Mag 5 My (mGauss)",
        "Ultium EMG.Internal Mag 5 Mz (mGauss)",
        "Ultium EMG.EMG 6 (uV)",
        "Ultium EMG.Internal Accel 6 Ax (mG)",
        "Ultium EMG.Internal Accel 6 Ay (mG)",
        "Ultium EMG.Internal Accel 6 Az (mG)",
        "Ultium EMG.Internal Gyro 6 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 6 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 6 Gz (deg/s)",
        "Ultium EMG.Internal Mag 6 Mx (mGauss)",
        "Ultium EMG.Internal Mag 6 My (mGauss)",
        "Ultium EMG.Internal Mag 6 Mz (mGauss)",
        "Ultium EMG.Internal Accel 7 Ax (mG)",
        "Ultium EMG.Internal Accel 7 Ay (mG)",
        "Ultium EMG.Internal Accel 7 Az (mG)",
        "Ultium EMG.Internal Gyro 7 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 7 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 7 Gz (deg/s)",
        "Ultium EMG.Internal Mag 7 Mx (mGauss)",
        "Ultium EMG.Internal Mag 7 My (mGauss)",
        "Ultium EMG.Internal Mag 7 Mz (mGauss)",
        "Ultium EMG.Internal Accel 8 Ax (mG)",
        "Ultium EMG.Internal Accel 8 Ay (mG)",
        "Ultium EMG.Internal Accel 8 Az (mG)",
        "Ultium EMG.Internal Gyro 8 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 8 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 8 Gz (deg/s)",
        "Ultium EMG.Internal Mag 8 Mx (mGauss)",
        "Ultium EMG.Internal Mag 8 My (mGauss)",
        "Ultium EMG.Internal Mag 8 Mz (mGauss)",
        "Ultium EMG.Internal Accel 9 Ax (mG)",
        "Ultium EMG.Internal Accel 9 Ay (mG)",
        "Ultium EMG.Internal Accel 9 Az (mG)",
        "Ultium EMG.Internal Gyro 9 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 9 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 9 Gz (deg/s)",
        "Ultium EMG.Internal Mag 9 Mx (mGauss)",
        "Ultium EMG.Internal Mag 9 My (mGauss)",
        "Ultium EMG.Internal Mag 9 Mz (mGauss)",
        "Ultium EMG.Internal Accel 10 Ax (mG)",
        "Ultium EMG.Internal Accel 10 Ay (mG)",
        "Ultium EMG.Internal Accel 10 Az (mG)",
        "Ultium EMG.Internal Gyro 10 Gx (deg/s)",
        "Ultium EMG.Internal Gyro 10 Gy (deg/s)",
        "Ultium EMG.Internal Gyro 10 Gz (deg/s)",
        "Ultium EMG.Internal Mag 10 Mx (mGauss)",
        "Ultium EMG.Internal Mag 10 My (mGauss)",
        "Ultium EMG.Internal Mag 10 Mz (mGauss)",
    ]

    if section == "EMG":
        sect = "EMG "
    else:
        sect = section

    n_vars_load = ["time"] + [s for s in all_vars if sect in s]

    # file = Path("F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")

    print("\nLoading files...")
    timerCarga = time.perf_counter()  # inicia el contador de tiempo

    num_processed_files = 0
    dfAll = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daAll = []
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(files_list):
        print(f"Cargando archivo nº {nf+1}/{len(files_list)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            dfProvis = read_noraxon_pl(file, n_vars_load)
            n_file = file.stem.split("_")[
                1:
            ]  # para quitar lo primero que pone, que es la fecha
            if len(n_file) == 5:
                n_project = n_file[0] if n_project is None else n_project
                particip = n_file[-4]
                tipo = n_file[-3]
                subtipo = n_file[-2]
            elif len(n_file) == 4:
                n_project = n_file[0]
                particip = n_file[1]
                tipo = n_file[2]
                subtipo = "X"
            elif len(n_file) == 3:
                particip = n_file[0]
                tipo = n_file[-2]
                subtipo = "X"
            if n_project is None:
                n_project = "EstudioX"

            repe = str(int(n_file[-1]))  # int(file.stem.split('.')[0][-1]
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

            dfAll.append(dfProvis)

            print(f"{dfAll[-1].shape[0]} rows and {dfAll[-1].shape[1]} columns")
            print("Time {0:.3f} s \n".format(time.perf_counter() - timerSub))
            num_processed_files += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                f"\nATTENTION. Unable to process {file.parent.name}, {file.name}, {err}\n"
            )
            error_files.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    dfAll = pl.concat(dfAll)

    print(
        f"Loaded {num_processed_files} files in {time.perf_counter()-timerCarga:.3f} s \n"
    )

    # Show file errors
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    if isinstance(data_type, str):
        cast = pl.Float32() if data_type == "float32" else pl.Float64()
        dfAll = dfAll.select(
            # pl.col(['n_project', 'tipo', 'subtipo', 'ID', 'repe']),
            pl.exclude(n_vars_load),
            pl.col(n_vars_load).cast(cast),
        )

    # Rename files
    n_vars_load2 = [s.split(".")[-1] for s in n_vars_load[1:] if sect in s]

    # Rename columnas
    # if section == 'EMG':
    #     rename = [s.split('Internal ')[-1][:-5] for s in n_vars_load2]
    # elif section == 'Accel':
    #     rename = [s.split('Internal ')[-1][:-5] for s in n_vars_load2]
    # elif section == 'Gyro':
    rename = [s.split("Internal ")[-1].split(" (")[0] for s in n_vars_load2]

    dfAll = dfAll.rename(dict(zip(dfAll.columns[1 : len(rename) + 1], rename)))

    daAll = df_to_da(dfAll, n_project=n_project)
    daAll.name = section
    daAll.attrs["units"] = n_vars_load2[0].split("(")[-1].split(")")[0]

    # daAll = split_dim_axis(daAll)

    # daAll = split_dim_side(daAll)

    return daAll


# =============================================================================
# %% Prueba las funciones
# =============================================================================
if __name__ == "__main__":
    ###################################
    # PROBAR CÁLCULOS ÁNGULOS IMUS CON PYKINEMATICS
    # pip install pykinematics
    # TAMBIÉN PROBAR CON pip install pyjamalib, conda install conda-forge::pyjamalib
    # TAMBIÉN SCIKIT-KINEMATICS http://work.thaslwanter.at/skinematics/html/
    # pip install scikit-kinematics
    ###################################
    # work_path = Path('F:\Programacion\Python\Mios\TratamientoDatos\EjemploBikefitting')
    work_path = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000"
    )

    # Carga EMG
    daEMG = load_merge_noraxon_csv(path=work_path, section="EMG")
    # daEMG.attrs['units'] = 'uV'
    daEMG.plot.line(x="time", col="n_var", row="ID", sharey=False)

    # Load Acelerometer
    daAcc = load_merge_noraxon_csv(path=work_path, section="Accel")
    daAcc = split_dim_axis(da=daAcc)
    daAcc.plot.line(x="time", col="n_var", row="ID", sharey=False)

    # Load Gyroscope
    daGyro = load_merge_noraxon_csv(path=work_path, section="Gyro")
    daGyro = split_dim_axis(da=daGyro)
    daGyro.plot.line(x="time", col="n_var", row="ID", sharey=False)
    daGyro.where(daGyro.tipo == "salto", drop=True).isel(ID=0).plot.line(
        x="time", col="n_var", col_wrap=4, sharey=False
    )
    daGyro.sel(ID=daGyro.ID.str.contains("salto")).isel(ID=0).plot.line(
        x="time", col="n_var", col_wrap=4, sharey=False
    )

    # Load Magnetometer
    daMag = load_merge_noraxon_csv(path=work_path, section="Mag")
    daMag = split_dim_axis(da=daMag)
    daMag.plot.line(x="time", col="n_var", row="ID", sharey=False)
    daMag.sel(ID=daMag.ID.str.contains("salto")).isel(ID=0).plot.line(
        x="time", col="n_var", col_wrap=4, sharey=False
    )

    """
    file = Path(r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-43_PO_S000_carrera_001.csv")
    #Sigue siendo más rápido con polars que pd engine='c'
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(10):
        dfProvis = read_noraxon_pd(file)
    print(time.perf_counter()-t)
    
    
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(10):
        dfProvis = read_noraxon_pl(file, n_vars_load=['Ultium EMG.Internal Accel 1 Ax (mG)'], to_dataarray=False)
    print(time.perf_counter()-t)
    
    read_noraxon_pl(file, to_dataarray=False)
    """
