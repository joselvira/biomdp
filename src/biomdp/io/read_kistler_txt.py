# -*- coding: utf-8 -*-
"""
Created on Sun Mar 09 11:12:37 2025

@author: Jose L. L. Elvira

Read data from Kistler Bioware .txt exported files.
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.0"
__date__ = "09/03/2025"

"""
Updates:
    09/03/2025, v0.1.0
        - Version imported from jump_forces_utils.

"""

from typing import List, Any
import numpy as np
import pandas as pd
import xarray as xr
import polars as pl

import time
from pathlib import Path


# =============================================================================
# %% Functions
# =============================================================================


def read_kistler_txt(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
    engine="polars",
    raw: bool = False,
) -> Any:

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    if engine == "polars":
        da = read_kistler_txt_pl(
            file,
            lin_header=lin_header,
            n_vars_load=n_vars_load,
            to_dataarray=to_dataarray,
            raw=raw,
        )

    elif engine == "pandas":
        da = read_kistler_txt_pd(
            file,
            lin_header=lin_header,
            n_vars_load=n_vars_load,
            to_dataarray=to_dataarray,
        )

    elif engine == "arrow":
        da = read_kistler_txt_arrow(
            file,
            lin_header=lin_header,
            n_vars_load=n_vars_load,
            to_dataarray=to_dataarray,
        )
    else:
        raise ValueError(
            f"Engine {engine} not valid\nTry with 'polars', 'pandas' or 'arrow'"
        )

    return da


# Carga un archivo de Bioware como dataframe de Polars
def read_kistler_txt_pl(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
    raw: bool = False,
    magnitude: str = "force",
) -> pl.DataFrame | xr.DataArray:
    try:
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
        if raw:
            return df

        else:
            if magnitude == "force":
                try:
                    x = df.select(pl.col("^*Fx.*$")).to_numpy()
                    y = df.select(pl.col("^*Fy.*$")).to_numpy()
                    z = df.select(pl.col("^*Fz.*$")).to_numpy()
                except:
                    raise Exception("Expected header with abs time (s), Fx, Fy, Fz")
                data = np.stack([x, y, z])
                freq = 1 / (df[1, "abs time (s)"] - df[0, "abs time (s)"])
                # ending = -3
                coords = {
                    "axis": ["x", "y", "z"],
                    "time": np.arange(data.shape[1]) / freq,
                    "plate": range(
                        1, x.shape[1] + 1
                    ),  # [x[:ending] for x in df.columns if 'x' in x[-1]],
                }
                da = (
                    xr.DataArray(
                        data=data,
                        dims=coords.keys(),
                        coords=coords,
                    )
                    .astype(float)
                    .transpose("plate", "axis", "time")
                )
                da.name = "forces"
                da.attrs["freq"] = freq
                da.time.attrs["units"] = "s"
                da.attrs["units"] = "N"

            elif magnitude == "cop":
                raise Exception("Not implemented yet")

    except Exception as err:
        print(f"\nATTENTION. Unable to process {file.name}, {err}, \n")

    return da


def read_kistler_txt_arrow(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
) -> pd.DataFrame:
    """In test, at the moment it does not work when there are repeated cols"""
    from pyarrow import csv

    read_options = csv.ReadOptions(
        # column_names=['Fx', 'Fy', 'Fz'],
        skip_rows=lin_header,
        skip_rows_after_names=1,
    )
    parse_options = csv.ParseOptions(delimiter="\t")
    data = csv.read_csv(file, read_options=read_options, parse_options=parse_options)
    return data.to_pandas()


def read_kistler_txt_pd(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
) -> pd.DataFrame | xr.DataArray:

    df = (
        pd.read_csv(
            file,
            header=lin_header,
            usecols=n_vars_load,  # ['Fx', 'Fy', 'Fz', 'Fx.1', 'Fy.1', 'Fz.1'], #n_vars_load,
            # skiprows=18,
            delimiter="\t",
            # dtype=np.float64,
            engine="c",  # "pyarrow" con pyarrow no funciona bien de momento cargar columnas con nombre repetido,
        ).drop(index=0)
        # , columns=nom_vars_cargar)
        # .slice(1, None) #quita la fila de unidades (N) #no hace falta con skip_rows_after_header=1
        # .select(pl.col(n_vars_load))
        # .rename({'abs time (s)':'time'}) #'Fx':'x', 'Fy':'y', 'Fz':'z',
        #          #'Fx_duplicated_0':'x_duplicated_0', 'Fy_duplicated_0':'y_duplicated_0', 'Fz_duplicated_0':'z'
        #          })
    )
    # df.dtypes

    # ----Transform pandas to xarray
    if to_dataarray:
        x = df.filter(regex="Fx*")  # .to_numpy()
        y = df.filter(regex="Fy*")
        z = df.filter(regex="Fx*")
        data = np.stack([x, y, z])
        freq = 1 / (df.loc[2, "abs time (s)"] - df.loc[1, "abs time (s)"])
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


def split_plataforms(da: xr.DataArray) -> xr.DataArray:
    plat1 = da.sel(n_var=da.n_var.str.startswith("F1"))
    plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip("F1"))

    plat2 = da.sel(n_var=da.n_var.str.startswith("F2"))
    plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip("F2"))

    da = xr.concat([plat1, plat2], dim="plat").assign_coords(plat=[1, 2])

    return da


def split_axis(da: xr.DataArray) -> xr.DataArray:
    # NOT NECESSARY WITH COMPUTE_FORCES_AX???
    # TODO: The letter of the axis in the name must be removed
    x = da.sel(n_var=da.n_var.str.contains("x"))
    y = da.sel(n_var=da.n_var.str.contains("y"))
    z = da.sel(n_var=da.n_var.str.contains("z"))
    da = (
        xr.concat([x, y, z], dim="axis")
        # .assign_coords(n_var='plat1')
        .assign_coords(axis=["x", "y", "z"])
        # .expand_dims({'n_var':1})
    )
    return da


def compute_forces_axes(da: xr.DataArray) -> xr.DataArray:
    # da=daForce

    if "plat" not in da.coords:
        da = split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains("x")).sum(dim="n_var")
    Fy = da.sel(n_var=da.n_var.str.contains("y")).sum(dim="n_var")
    Fz = da.sel(n_var=da.n_var.str.contains("z")).sum(dim="n_var")

    daReturn = xr.concat([Fx, Fy, Fz], dim="axis").assign_coords(axis=["x", "y", "z"])
    # daReturn.plot.line(x='time', col='plat')

    return daReturn


def compute_moments_axes(da: xr.DataArray) -> xr.DataArray:
    # da=daForce
    raise Exception("Not implemented yet")
    """
    if 'plat' not in da.coords:
        da = split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains('x')).sum(dim='n_var')
    Fy = da.sel(n_var=da.n_var.str.contains('y')).sum(dim='n_var')
    Fz = da.sel(n_var=da.n_var.str.contains('z')).sum(dim='n_var')
        
    daReturn = (xr.concat([Fx, Fy, Fz], dim='axis')
                .assign_coords(axis=['x', 'y', 'z'])
                )
    #daReturn.plot.line(x='time', col='plat')
    """
    return daReturn


# =============================================================================
# %% TESTS
# =============================================================================
if __name__ == "__main__":

    # from biomdp.io.read_kistler_txt import read_kistler_c3d_xr, read_kistler_ezc3d_xr

    work_path = Path(r"src\biomdp\datasets")
    file = work_path / "kistler_CMJ_1plate.txt"
    daForce = read_kistler_txt(file)
    daForce.isel(plate=0).plot.line(x="time")  # , col="plat")

    file = work_path / "kistler_DJ_2plates.txt"
    daForce = read_kistler_txt(file)
    daForce.plot.line(x="time", col="plate")
    daForce.sum(dim="plate").plot.line(x="time")
