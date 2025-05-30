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
__version__ = "0.2.1"
__date__ = "30/05/2025"

"""
Updates:
    30/05/2025, v0.2.1
        - Incuded "separator" parameter in read_kistler_txt_pl.

    25/03/2025, v0.2.0
        - Incuded general function to distribute according to "engine".
    
    09/03/2025, v0.1.0
        - Version imported from jump_forces_utils.

"""

from typing import List, Any
import numpy as np

import pandas as pd
import xarray as xr

# import polars as pl

import time
from pathlib import Path


# =============================================================================
# %% Functions
# =============================================================================


def read_kistler_txt(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    # to_dataarray: bool = False,
    engine="polars",
    raw: bool = False,
    magnitude: str = "force",
) -> Any:
    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    if engine == "polars":
        ret = read_kistler_txt_pl(
            file,
            lin_header=lin_header,
            n_vars_load=n_vars_load,
            raw=raw,
            magnitude=magnitude,
        )

    elif engine == "pandas":
        ret = read_kistler_txt_pd(
            file,
            lin_header=lin_header,
            n_vars_load=n_vars_load,
            raw=raw,
            magnitude=magnitude,
        )

    elif engine == "arrow":
        ret = read_kistler_txt_arrow(
            file,
            lin_header=lin_header,
            n_vars_load=n_vars_load,
            raw=raw,
            magnitude=magnitude,
        )
    else:
        raise ValueError(
            f"Engine {engine} not valid\nTry with 'polars', 'pandas' or 'arrow'"
        )

    return ret


# Carga un archivo de Bioware como dataframe de Polars
def read_kistler_txt_pl(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    # to_dataarray: bool = False,
    raw: bool = False,
    magnitude: str = "force",
    separator: str = "\t",
) -> xr.DataArray | Any:
    """
    Read .txt files exported from Bioware

    Note: since BioWare Version 5.6.1.0 time column changed from "abs time (s)" to "abs time"
    """
    try:
        import polars as pl
    except:
        raise ImportError(
            "Polars package not instaled. Install it if you want to use the accelerated version"
        )

    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    try:
        df = (
            pl.read_csv(
                file,
                has_header=True,
                skip_rows=lin_header,
                skip_rows_after_header=1,
                columns=n_vars_load,
                separator=separator,
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
            freq = 1 / (
                df[1, 0] - df[0, 0]  # assumes time in first column
            )  # (df[1, "abs time (s)"] - df[0, "abs time (s)"])
            if magnitude == "force":
                try:
                    x = df.select(pl.col("^*Fx.*$")).to_numpy()
                    y = df.select(pl.col("^*Fy.*$")).to_numpy()
                    z = df.select(pl.col("^*Fz.*$")).to_numpy()
                except:
                    raise Exception("Expected header with abs time (s), Fx, Fy, Fz")
                data = np.stack([x, y, z])

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
                da.name = "Forces"
                da.attrs["freq"] = freq
                da.time.attrs["units"] = "s"
                da.attrs["units"] = "N"

            elif magnitude == "cop":

                try:
                    x = df.select(pl.col("^*Ax.*$")).to_numpy()
                    y = df.select(pl.col("^*Ay.*$")).to_numpy()
                except:
                    raise Exception("Expected header with abs time (s), Ax, Ay")
                data = np.stack([x, y])

                coords = {
                    "axis": ["x", "y"],
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
                da.name = "COP"
                da.attrs["freq"] = freq
                da.time.attrs["units"] = "s"
                da.attrs["units"] = "mm"

            else:
                raise ValueError(
                    f"Magnitude {magnitude} not valid.\nTry with 'force' or 'cop'"
                )

            return da

    except Exception as err:
        print(f"\nATTENTION. Unable to read {file.name}, {err}, \n")


def read_kistler_txt_pd(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    # to_dataarray: bool = False,
    raw: bool = False,
    magnitude: str = "force",
) -> xr.DataArray | pd.DataFrame | None:

    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    
    try:
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
        ).astype(float)
        # df.dtypes

        if raw:
            return df
        # ----Transform pandas to xarray
        else:
            try:
                freq = 1 / (df.loc[2, "abs time (s)"] - df.loc[1, "abs time (s)"])
            except:
                freq = 1 / (df.loc[2, "abs time"] - df.loc[1, "abs time"])
            if magnitude == "force":
                x = df.filter(regex="Fx")  # .to_numpy()
                y = df.filter(regex="Fy")
                z = df.filter(regex="Fz")
                data = np.stack([x, y, z])
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
                # coords = {
                #     "axis": ["x", "y", "z"],
                #     "time": np.arange(data.shape[1]) / freq,
                #     "n_var": [
                #         "Force"
                #     ],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
                # }
                # da = (
                #     xr.DataArray(
                #         data=data,
                #         dims=coords.keys(),
                #         coords=coords,
                #     )
                #     .astype(float)
                #     .transpose("n_var", "axis", "time")
                # )
                da.name = "Forces"
                da.attrs["freq"] = freq
                da.time.attrs["units"] = "s"
                da.attrs["units"] = "N"

            elif magnitude == "cop":
                x = df.filter(regex="Ax")  # .to_numpy()
                y = df.filter(regex="Ay")
                data = np.stack([x, y])
                # ending = -3
                coords = {
                    "axis": ["x", "y"],
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
                da.name = "COP"
                da.attrs["freq"] = freq
                da.time.attrs["units"] = "s"
                da.attrs["units"] = "mm"
            else:
                raise ValueError(
                    f"Magnitude {magnitude} not valid.\nTry with 'force' or 'cop'"
                )

            return da

    except Exception as err:
        print(f"\nATTENTION. Unable to read {file.name}, {err}, \n")


def read_kistler_txt_arrow(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    # to_dataarray: bool = False,
    raw: bool = False,
    magnitude: str = "force",
) -> pd.DataFrame | xr.DataArray:
    """Testing, at the moment it does not work when there are repeated cols"""
        
    try:
        from pyarrow import csv
    except:
        raise ImportError("pyarrow not installed")

    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    try:
        read_options = csv.ReadOptions(
            # column_names=['Fx', 'Fy', 'Fz'],
            skip_rows=lin_header,
            skip_rows_after_names=1,
        )
        parse_options = csv.ParseOptions(delimiter="\t")
        data = csv.read_csv(
            file, read_options=read_options, parse_options=parse_options
        )

        df = data.to_pandas()
        if raw:
            return df
        # ----Transform pandas to xarray
        else:
            if magnitude == "force":
                x = df.filter(regex="Fx")  # .to_numpy()
                y = df.filter(regex="Fy")
                z = df.filter(regex="Fz")
                data = np.stack([x, y, z])
                freq = 1 / (df.loc[2, "abs time (s)"] - df.loc[1, "abs time (s)"])
                ending = -3
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
                da.name = "Forces"
                da.attrs["freq"] = freq
                da.time.attrs["units"] = "s"
                da.attrs["units"] = "N"

            elif magnitude == "cop":
                raise Exception("Magnitude "cop" not implemented yet")
            else:
                raise ValueError(
                    f"Magnitude {magnitude} not valid.\nTry with 'force' or 'cop'"
                )

            return da

    except Exception as err:
        print(f"\nATTENTION. Unable to read {file.name}, {err}, \n")


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

    work_path = Path(r"src\datasets")
    file = work_path / "kistler_CMJ_1plate.txt"
    daForce = read_kistler_txt(file)
    daForce.isel(plate=0).plot.line(x="time")  # , col="plat")

    dfForce = read_kistler_txt(file, raw=True, engine="pandas")
    dfForce.plot(x="abs time (s)")

    dfForce = read_kistler_txt(file, raw=True, engine="arrow")
    dfForce.plot(x="abs time (s)")

    file = work_path / "kistler_DJ_2plates.txt"
    daForce = read_kistler_txt(file)
    daForce.plot.line(x="time", col="plate")
    daForce.sum(dim="plate").plot.line(x="time")

    daForce = read_kistler_txt(file, engine="pandas")
    daForce.plot.line(x="time", col="plate")

    daForce = read_kistler_txt(file, engine="arrow")
    daForce.plot.line(x="time", col="plate")

    dfForce = read_kistler_txt(file, raw=True, engine="pandas")
    dfForce[["Fx", "Fy", "Fz"]].plot()

    file = work_path / "kistler_COP.txt"
    daForce = read_kistler_txt(file, magnitude="cop")
    daForce.to_dataset("axis").plot.scatter(x="x", y="y", col="plate")

    daForce = read_kistler_txt(file, magnitude="cop", engine="pandas")
    daForce.to_dataset("axis").plot.scatter(x="x", y="y", col="plate")

    import timeit

    def test_performance():
        result = read_kistler_txt(file, engine="polars")
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )

    def test_performance():
        result = read_kistler_txt(file, engine="pandas")
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )

    def test_performance():
        result = read_kistler_txt(file, engine="arrow")
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )
