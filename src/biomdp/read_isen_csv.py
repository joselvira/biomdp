# -*- coding: utf-8 -*-
"""
Created on Fry Mar 1 13:15:58 2024

@author: jose.lopeze

Lectura de archivos .csv exportados de iSen.

"""


#
# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.0"
__date__ = "17/08/2024"

"""
Updates:
     11/03/2025, v0.1.1
            - Adapted to biomdp with translations.
    
    17/08/2024, v0.1.0
        - Basado en el usado para registros 2021 para TFM de María Aracil. 

        """
from typing import List
import numpy as np
import pandas as pd
import xarray as xr
import polars as pl

import matplotlib.pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages #para guardar gráficas en pdf
# import seaborn as sns

from pathlib import Path
import time  # para cuantificar tiempos de procesado


# import sys
# sys.path.append('F:\Programacion\Python\Mios\Functions')
# #sys.path.append('G:\Mi unidad\Programacion\Python\Mios\Functions')

# from readViconCsv import read_vicon_csv


# =============================================================================
# %% Functions
# =============================================================================


def split_dim_axis(da: xr.DataArray) -> xr.DataArray:
    # Separa el xarray en ejes creando dimensión axis

    x = da.sel(
        n_var=da.n_var.str.contains("Flexo-extensión")
    )  # .rename({"n_var": "axis"})
    x = x.assign_coords(
        n_var=[s.split("Flexo-extensión ")[1][:-3] for s in x.n_var.to_series()]
    )

    y = da.sel(
        n_var=da.n_var.str.contains("Abducción-aducción")
    )  # .rename({"n_var": "axis"})
    y = y.assign_coords(
        n_var=[s.split("Abducción-aducción ")[1][:-3] for s in y.n_var.to_series()]
    )

    z = da.sel(n_var=da.n_var.str.contains("Rotación"))  # .rename({"n_var": "axis"})
    z = z.assign_coords(
        n_var=[s.split("Rotación ")[1][:-3] for s in z.n_var.to_series()]
    )

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


def assign_subcategories_xr(da: xr.DataArray, n_project: str = None) -> xr.DataArray:

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
    dfAll: pd.DataFrame, n_project: str | None = None, show: bool = False
) -> xr.DataArray:
    if isinstance(dfAll, pl.DataFrame):
        # Transforma df polars a dataarray con todas las variables cargadas
        vars_leidas = dfAll.select(
            pl.exclude(
                ["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
            ),
        ).columns

        dfpd = dfAll.unpivot(
            index=["ID", "time"], on=vars_leidas, variable_name="n_var"
        ).to_pandas()

    else:  # viene con pandas
        vars_leidas = dfAll.drop(
            columns=["time", "estudio", "tipo", "subtipo", "ID", "particip", "repe"]
        ).columns
        dfpd = dfAll.drop(
            columns=["estudio", "tipo", "subtipo", "particip", "repe"]
        ).melt(id_vars=["ID", "time"], var_name="n_var")
    daAll = (
        # dfpd.drop(columns=["estudio", "tipo", "subtipo", "particip", "repe"])
        dfpd  # .melt(id_vars=["ID", "time"], var_name="n_var")
        # pd.melt(dfAllFiles.to_pandas().drop(columns=['estudio','tipo','subtipo']), id_vars=['ID', 'repe', 'time'], var_name='axis')
        .set_index(["ID", "n_var", "time"])
        .to_xarray()
        .to_array()
        .squeeze("variable")
        .drop_vars("variable")
    )

    # Rename columns
    # dfAll = dfAll.rename({'abs time (s)':'time', 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    # Assign extra coordinates
    daAll = assign_subcategories_xr(da=daAll, n_project=n_project)
    # daAll = daAll.assign_coords(estudio=('ID', dfAll.filter(pl.col('time')==0.000).get_column('estudio').to_list()),
    #                                          particip=('ID', dfAll.filter(pl.col('time')==0.000).get_column('particip').to_list()),
    #                                          tipo=('ID', dfAll.filter(pl.col('time')==0.000).get_column('tipo').to_list()),
    #                                          subtipo=('ID', dfAll.filter(pl.col('time')==0.000).get_column('subtipo').to_list()),
    #                                          repe=('ID', dfAll.filter(pl.col('time')==0.000).get_column('repe').to_list()),
    #                                          )
    # Set time coordinate type, required??
    ###########daAllFiles = daAllFiles.assign_coords(time=('time', daAllFiles.time.astype('float32').values))

    # daAllFiles.sel(ID='PCF_SCT05', axis='z').plot.line(x='time', col='repe')
    # daAllFiles.assign_coords(time=daAllFiles.time.astype('float32'))
    daAll.attrs = {
        "freq": (np.round(1 / (daAll.time[1] - daAll.time[0]), 1)).data,
        "units": "N",
    }
    daAll.time.attrs["units"] = "s"
    daAll.name = "Forces"

    return daAll


def read_isen_pd(
    file: str | Path, n_vars_load: List[str] | None = None, to_dataarray: bool = False
) -> pd.DataFrame:
    # file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-CaderasRodillas_00.csv')

    df = pd.read_csv(file, dtype=np.float64, engine="c")  # .astype(np.float64)
    # df = df.drop(columns=df.filter(regex="(Normal)") + df.filter(regex="(Tiempo.)"))

    #       df.filter(regex="(Normal)").values)).drop(columns=df.filter(regex="(Tiempo.)"))
    # df.filter(regex="(Normal)")+df.filter(regex="(Tiempo.)")
    # df.dropna(axis="columns", how="all")

    # df.filter(regex="Tiempo.")

    if to_dataarray:
        da = xr.DataArray()
        return da

    return df


def read_isen_pl(
    file: str | Path, n_vars_load: List[str] | None = None, to_dataarray: bool = False
) -> xr.DataArray | pl.DataFrame:
    # file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-CaderasRodillas_00.csv')
    # file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-Sensores_00.csv')
    df = (
        pl.read_csv(
            file,
            # has_header=True,
            # skip_rows=0,
            # skip_rows_after_header=0,
            columns=n_vars_load,
            # separator=",",
        )  # , columns=nom_vars_cargar)
        .select(pl.exclude("^.*_duplicated_.*$"))  # quita columnas de tiempo duplicadas
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

        # Separa ejes articulares
        x = df.select(pl.col("^Flexo-extensión.*$")).to_numpy()
        y = df.select(pl.col("^Aducción-abducción.*$")).to_numpy()
        z = df.select(pl.col("^Rotación.*$")).to_numpy()
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


def load_merge_iSen_angles_csv(
    path: str | Path = None,
    file_list: List[str | Path] | None = None,
    n_vars_load: List[str] | None = None,
    n_project: str | None = None,
    data_type: str | None = None,
    show: bool = False,
) -> xr.DataArray:
    """
    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    file_list : TYPE, optional
        DESCRIPTION. List of files to read, overrides path.The default is None.
    n_vars_load : TYPE, optional
        DESCRIPTION. The default is None.
    n_project : string, optional
        DESCRIPTION. The name of the study.
    data_type:
        Conversión al tipo de datos indicado. Por defecto es None, que quiere
        decir que se mantiene el tipo original ('float64')
    show : bool, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    daAllFiles : xarray DataArray
        DESCRIPTION.
    dfAllFiles : pandas DataFrame
        DESCRIPTION.

    """
    if data_type is None:
        data_type = float

    if file_list is None:
        file_list = sorted(
            list(path.glob("*.csv"))  # "**/*.csv"
        )  #'**/*.txt' incluye los que haya en subcarpetas
        file_list = [
            x
            for x in file_list
            if "error" not in x.name
            and "Sensores" not in x.name
            and "Actual" not in x.name
        ]  # selecciona archivos
    else:
        file_list = file_list

    # file = Path("F:/Investigacion/Proyectos/Tesis/TesisCoralPodologa/Registros/PRUEBASiSEN/ANGULOS.csv")

    print("\nLoading files...")
    timer_load = time.perf_counter()  # inicia el contador de tiempo

    n_processed_files = 0
    dfAll = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daAll = []
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(file_list):
        print(f"Loading file num. {nf+1}/{len(file_list)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            dfProvis = read_isen_pl(file, n_vars_load)

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
            if n_project is None:
                n_project = "EstudioX"

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

            dfAll.append(dfProvis)

            print(f"{dfAll[-1].shape[0]} filas y {dfAll[-1].shape[1]} columnas")
            print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))
            n_processed_files += 1

        except Exception as err:  # Si falla anota un error y continúa
            print(
                "\nATTENTION. Unable to process {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            error_files.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    dfAll = pl.concat(dfAll)

    print(
        f"Loaded {n_processed_files} files in {time.perf_counter()-timer_load:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
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

    dfAll = dfAll.rename({"Tiempo": "time"})  # , 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    daAll = df_to_da(dfAll, n_project=n_project)

    daAll = split_dim_axis(daAll)

    daAll = split_dim_side(daAll)

    # TODO: CONTINUE TO MAKE IT SO THAT IF IT LOADS VARIABLES OTHER THAN THE CONVENTIONAL ONES, IT PASSES THEM TO DAARRAY BUT WITHOUT AXES
    # if  dfAll.columns == ['abs time (s)', 'Fx', 'Fy', 'Fz'] or n_vars_load == ['abs time (s)', 'Fx', 'Fy', 'Fz', 'Fx_duplicated_0', 'Fy_duplicated_0', 'Fz_duplicated_0']:
    #     daAll = pasa_dfpl_a_da(dfAll, merge_2_plats=merge_2_plats, show=show)

    """    
    #Transforma a pandas y a dataarray
    daAll = (dfAll.drop(['estudio','tipo','subtipo'])
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
    
    daAll = asigna_subcategorias_xr(da=daAll, n_project=estudio, subtipo='2PAP')
    
    daAll.name = 'Forces'
    daAll.attrs['units'] = 'N'
    daAll.attrs['freq'] = freq
    daAll.time.attrs['units'] = 's'
    """
    return daAll


def load_merge_iSen_sensors_csv(
    path: str | Path,
    n_vars_load: List[str] | None = None,
    n_project: str | None = None,
    data_type: str | None = None,
    show: bool = False,
) -> xr.DataArray:
    """
    Load data from iSen CSV files and merge them into a single xarray DataArray.

    Parameters
    ----------
    path : str or Path
        Path to the folder containing the iSen CSV files.
    n_vars_load : None or list of str, optional
        List of variables to load from the CSV files. If None, all variables
        are loaded. The default is None.
    n_project : str, optional
        Name of the study. The default is None.
    data_type : type, optional
        Data type to which the values will be converted. If None, the original
        data type is kept. The default is None.
    show : bool, optional
        If True, the resulting DataArray is plotted. The default is False.

    Returns
    -------
    daAll : xarray DataArray
        DataArray containing all the data from the CSV files.

    """

    if data_type is None:
        data_type = float

    file_list = sorted(
        list(path.glob("*.csv"))  # "**/*.csv"
    )  #'**/*.txt' incluye los que haya en subcarpetas
    file_list = [
        x for x in file_list if "error" not in x.name and "aceleración local" in x.name
    ]  # selecciona archivos

    # file = Path("F:/Investigacion/Proyectos/Tesis/TesisCoralPodologa/Registros/PRUEBASiSEN/ANGULOS.csv")

    print("\nLoading files...")
    timer_load = time.perf_counter()  # inicia el contador de tiempo

    num_processed_files = 0
    dfAll = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    daAll = []
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error
    for nf, file in enumerate(file_list):
        print(f"Loading file num. {nf+1}/{len(file_list)}: {file.name}")
        try:
            timerSub = time.perf_counter()  # inicia el contador de tiempo

            dfProvis = read_isen_pl(file, n_vars_load)

            # ADAPT DIRECTORY LABELS
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
            else:
                particip = file.stem.split("_")[0]
                tipo = "X"
                subtipo = "X"
                repe = 0

            if n_project is None:
                n_project = "EstudioX"

            try:
                repe = str(
                    int(file.stem.split("_")[-1])
                )  # int(file.stem.split('.')[0][-1]
            except:
                repe = 0
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
                "\nATTENTION. Unable to process {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            error_files.append(file.parent.name + " " + file.name + " " + str(err))
            continue

    dfAll = pl.concat(dfAll)

    print(
        f"Loaded {num_processed_files} files in {time.perf_counter()-timer_load:.3f} s \n"
    )

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    if isinstance(
        data_type, str
    ):  # si se ha definido algún tipo de datos, por defecto es 'float64'
        cast = pl.Float32() if data_type == "float32" else pl.Float64()
        dfAll = dfAll.select(
            # pl.col(['n_project', 'tipo', 'subtipo', 'ID', 'repe']),
            pl.exclude(n_vars_load),
            pl.col(n_vars_load).cast(cast),
        )

    dfAll = dfAll.rename({"Tiempo": "time"})  # , 'Fx':'x', 'Fy':'y', 'Fz':'z'})

    daAll = df_to_da(dfAll, n_project=n_project)

    return daAll


def load_merge_iSen_csvxxxx(path, n_preprocessed_file):
    """TESTING"""
    print("\nLoading  iSen files...")
    timer_load = time.time()  # inicia el contador de tiempo
    # nomVarsACargar = nomVarsContinuas#+nomVarsDiscretas

    # Selecciona los archivos en la carpeta
    file_list = sorted(
        path.glob("*.csv")  # "**/*.csv"
    )  # list((path).glob('**/*.csv'))#incluye los que haya en subcarpetas
    file_list = [
        x for x in file_list if "error" not in x.name and "SENSORES" not in x.name
    ]  # selecciona archivos

    print("\nLoading files...")
    timer_load = time.perf_counter()  # inicia el contador de tiempo

    num_processed_files = 0
    dfAll = (
        []
    )  # guarda los dataframes que va leyendo en formato lista y al final los concatena
    error_files = []  # guarda los nombres de archivo que no se pueden abrir y su error

    for nf, file in enumerate(file_list[:]):
        print(f"Loading file num. {nf}/{len(file_list)}: {file.name}")
        """
        #Selecciona archivos según potencia
        if file.parent.parts[-1] not in ['2W', '6W']: #si no es de marcha se lo salta
            #print('{} archivo estático'.format(file.stem))
            continue
        """
        # # Selecciona archivos según variable
        # if file.stem not in [
        #     r"Flexo-extensión caderas",
        #     r"Flexo-extensión rodillas",
        #     r"Dorsiflexión-plantarflexión tobillos",
        # ]:  # si no es de marcha se lo salta
        #     # print('{} archivo estático'.format(file.stem))
        #     continue
        # print(file)

        try:
            timerSub = time.time()  # inicia el contador de tiempo
            dfprovis = pd.read_csv(
                file, sep=","
            )  # .filter(regex="::Y")  # se queda solo con las columnas de datos, descarta las de tiempos

            # Ajusta los nombres R y L
            dfprovis.rename(
                columns={
                    str(dfprovis.filter(regex="derech").columns.values)[2:-2]: "R",
                    str(dfprovis.filter(regex="izquierd").columns.values)[2:-2]: "L",
                },
                inplace=True,
            )
            # dfprovis.columns=['R', 'L']

            freq = (
                1 / dfprovis["Tiempo"].diff().mean()
            )  # (dfprovis["Tiempo"][1] - dfprovis["Tiempo"][0])

            # Añade columna tiempo
            t = np.arange(0, len(dfprovis) / freq)[
                : len(dfprovis)
            ]  # ajuste al cargar el tiempo porque en algunos t sale con un dato de más

            nom = file.parent.parts[-2]
            pot = file.parent.parts[-1]
            artic = file.stem.split()[-1][:-1]
            if "flex" in file.stem.lower():
                eje = "x"

            """
            if file.stem == 'Dorsiflexión-plantarflexión tobillos':
                artic = 'tobillo'
                eje='x'
            elif file.stem == 'Flexo-extensión rodillas':
                artic = 'rodilla'
                eje='x'
            elif file.stem == 'Flexo-extensión caderas':
                artic = 'cadera'
                eje='x'
            """

            # Añade etiquetas
            dfprovis = dfprovis.assign(
                **{
                    "nombre": nom,
                    "potencia": pot,
                    "articulacion": artic,
                    "eje": eje,
                    "time": t,
                }
            ).reset_index(drop=True)

            # Para formato tidy
            dfprovis = dfprovis.reindex(
                columns=["nombre", "potencia", "articulacion", "eje", "time", "R", "L"]
            )

            # Transforma a formato long, mejor dejarlo en tidy para que ocupe menos al guardarlo
            # dfprovis = pd.melt(dfprovis, id_vars=['nombre', 'potencia', 'articulacion', 'eje', 'time'], var_name='lado')#, value_vars=dfprovis[pot].iloc[:, :-4])
            # dfprovis = dfprovis.reindex(columns=['nombre', 'potencia', 'articulacion', 'eje', 'lado', 'time', 'value'])

            dfAllFiles.append(dfprovis)

            print(
                "{0} filas y {1} columnas".format(
                    dfAllFiles[-1].shape[0], dfAllFiles[-1].shape[1]
                )
            )
            print("Tiempo {0:.3f} s \n".format(time.time() - timerSub))
            num_processed_files += 1

        except Exception as err:  # Si falla anota un error y continua
            print(
                "\nATTENTION. Unable to process {0}, {1}, {2}".format(
                    file.parent.name, file.name, err
                ),
                "\n",
            )
            error_files.append(file.parent.name + " " + file.name + " " + str(err))
            continue
    dfAllFiles = pd.concat(dfAllFiles)
    print(f"Loaded {num_processed_files} files in {time.time() - timer_load:.3f} s \n")

    # Si no ha podido cargar algún archivo, lo indica
    if len(error_files) > 0:
        print("\nATTENTION. Unable to load:")
        for x in range(len(error_files)):
            print(error_files[x])

    # =============================================================================
    # Lo pasa a DataArray
    # =============================================================================
    # Transforma a formato long
    dfAllMulti = (
        pd.melt(
            dfAllFiles,
            id_vars=["nombre", "potencia", "articulacion", "eje", "time"],
            var_name="lado",
        )
        .reindex(
            columns=[
                "nombre",
                "potencia",
                "articulacion",
                "eje",
                "lado",
                "time",
                "value",
            ]
        )
        .set_index(["nombre", "potencia", "articulacion", "eje", "lado", "time"])
    )

    try:
        dfAllSubjects = dfAllMulti.to_xarray().to_array()
        dfAllSubjects = dfAllSubjects.sel(time=slice(0, 63))
        del dfAllSubjects["variable"]  # la quita de coordenadas
        dfAllSubjects = dfAllSubjects.squeeze("variable")  # la quita de dimensiones
        dfAllSubjects.attrs["freq"] = freq
        dfAllSubjects.attrs["units"] = "degrees"
        dfAllSubjects.time.attrs["units"] = "s"

        if False:
            dfAllSubjects.sel(
                nombre="Javi", potencia="2W", articulacion="rodilla", eje="x"
            ).plot.line(x="time", hue="lado")
            dfAllSubjects.sel(potencia="2W", articulacion="cadera").plot.line(
                x="time", row="nombre", col="lado"
            )

    except:
        print(
            "There may be a duplicate file. Search for the file with different duration."
        )
        # Si no funciona el data array, comprueba si hay duplicados
        for n, df in dfAllFiles.set_index(list(dfAllFiles.columns[:-2])).groupby(
            list(dfAllFiles.columns[:-2])
        ):
            print(n, len(df))

    if False:
        # Comparativa Todas las variables de side L y R juntas en una misma hoja de cada sujeto
        nompdf = work_path / "CamparacionVicon_iSen_PorArtics_iSen.pdf"
        with PdfPages(nompdf) as pdf_pages:
            for n, gda in dfAllSubjects.sel(eje="x").groupby("nombre"):
                g = gda.plot.line(
                    x="time",
                    row="potencia",
                    col="articulacion",
                    hue="lado",
                    sharey=False,
                    aspect=1.5,
                )
                for h, ax in enumerate(g.axes):  # extrae cada fila
                    for i in range(len(ax)):  # extrae cada axis (gráfica)
                        nom = str(g.data.loc[g.name_dicts[h, i]].nombre.data)
                        pot = str(g.data.loc[g.name_dicts[h, i]].potencia.data)
                        # print(nom, pot)
                        try:
                            ax[i].axvline(
                                x=dfFrames.loc[(nom, pot), "ini"] / freq, c="r", ls="--"
                            )
                            ax[i].axvline(
                                x=dfFrames.loc[(nom, pot), "fin"] / freq, c="r", ls="--"
                            )
                        except:
                            continue

                g.fig.subplots_adjust(top=0.95)
                g.fig.suptitle(n)
                pdf_pages.savefig(g.fig)
                plt.show()

    # =============================================================================
    #   Guarda archivos cargados
    # =============================================================================
    # Guarda xarray
    tpoGuardar = time.time()
    dfAllSubjects.to_netcdf(
        (work_path / (n_preprocessed_file + "_iSen")).with_suffix(".nc")
    )
    print(
        "\nPreprocessed Dataframe saved {0} en {1:0.2f} s.".format(
            n_preprocessed_file + "_iSen.nc", time.time() - tpoGuardar
        )
    )

    # Guarda dataframetpoGuardar = time.time()
    tpoGuardar = time.time()
    dfAllFiles.to_csv(
        (work_path / (n_preprocessed_file + "_iSen_tidy")).with_suffix(".csv"),
        index=False,
    )
    print(
        "\nPreprocessed DataArray saved {0} en {1:0.2f} s.".format(
            n_preprocessed_file + "_iSen_tidy.csv", time.time() - tpoGuardar
        )
    )
    # Transforma a formato long
    # dfAllFiles = pd.melt(dfAllFiles, id_vars=['nombre', 'potencia', 'articulacion', 'eje', 'time'], var_name='lado')#, value_vars=dfprovis[pot].iloc[:, :-4])
    # dfAllFiles = dfAllFiles.reindex(columns=['nombre', 'potencia', 'articulacion', 'eje', 'lado', 'time', 'value'])


def load_preprocessed(work_path: str | Path, n_preprocessed_file: str):
    # LOAD VICON
    if Path(
        (work_path / (n_preprocessed_file + "_Vicon")).with_suffix(".nc")
    ).is_file():
        tpo = time.time()
        ds_Vicon = xr.load_dataset(
            (work_path / (n_preprocessed_file + "_Vicon")).with_suffix(".nc")
        )
        # daAllFiles = xr.load_dataarray((work_path / (n_preprocessed_file+'_Vicon')).with_suffix('.nc'))
        print(
            "\nLoading preprocessed file ",
            n_preprocessed_file + "_Vicon.nc en {0:.3f} s.".format(time.time() - tpo),
        )
    else:
        raise Exception("Preprocessed Vicon file not found.")

    # LOAD ISEN
    if Path((work_path / (n_preprocessed_file + "_iSen")).with_suffix(".nc")).is_file():
        tpo = time.time()
        da_iSen = xr.load_dataarray(
            (work_path / (n_preprocessed_file + "_iSen")).with_suffix(".nc")
        )
        print(
            "\nSaved preprocessed file ",
            (work_path / (n_preprocessed_file + "_iSen")).with_suffix(".nc").name,
            "en {0:.3f} s.".format(time.time() - tpo),
        )
    else:
        raise Exception("Preprocessed iSen file not found")

    return ds_Vicon, da_iSen


# =============================================================================
# %% TESTS
# =============================================================================
if __name__ == "__main__":

    # Vicon
    # work_path = Path('F:\Programacion\Python\Mios\TratamientoDatos\EjemploBikefitting')
    work_path = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen"
    )
    # dfAllFiles, daAllFiles = carga_preprocesa_vicon(path=work_path)

    # guardar_preprocesados_vicon(path=work_path , nom_archivo='Prueba', df= dfAllFiles, da=daAllFiles)
    daData = load_merge_iSen_angles_csv(path=work_path)
    daData.sel(axis="x", time=slice(30, None)).plot.line(
        x="time", col="n_var", row="ID"
    )

    n_vars_load = (
        None  # ['Tiempo', 'BOLTWOODITE X::Y', 'BOLTWOODITE Y::Y', 'BOLTWOODITE Z::Y',
    )
    #                'LEO X::Y', 'LEO Y::Y', 'LEO Z::Y',
    #                'GASPEITE X::Y', 'GASPEITE Y::Y', 'GASPEITE Z::Y',
    #                'MIARGYRITE X::Y', 'MIARGYRITE Y::Y', 'MIARGYRITE Z::Y',
    #                'KARENAI X::Y', 'KARENAI Y::Y', 'KARENAI Z::Y'
    # ]
    daSensors = load_merge_iSen_sensors_csv(
        path=work_path / "SensoresSeparado", n_vars_load=n_vars_load
    )
    daSensors.sel(time=slice(30, None)).isel(ID=0).plot.line(
        x="time", col="n_var", col_wrap=4
    )

    r"""
    file = Path(r'F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\Piloto0iSen\S00_Todos-CaderasRodillas_00.csv')
    #Sigue siendo más rápido con polars que pd engine='c'
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(100):
        dfProvis = read_isen_pd(file)
    print(time.perf_counter()-t)
    
    
    t = time.perf_counter()  # inicia el contador de tiempo
    for i in range(100):
        dfProvis = read_isen_pl(file)
    print(time.perf_counter()-t)
    """
