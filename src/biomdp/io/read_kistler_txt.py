"""
Created on Sun Mar 09 11:12:37 2025

@author: Jose L. L. Elvira

Read data from Kistler Bioware .txt exported files.
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.3.0"
__date__ = "05/09/2026"

"""
Updates:
    31/08/2026, v0.3.0
        - The functions have been modified to match .dat and .c3d versions.
        - Selected engine='arrow' as default (fastest).

    30/05/2025, v0.2.1
        - Incuded "separator" parameter in read_kistler_txt_pl.

    25/03/2025, v0.2.0
        - Incuded general function to distribute according to "engine".
    
    09/03/2025, v0.1.0
        - Version imported from jump_forces_utils.

"""

# import polars as pl
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

# =============================================================================
# %% Functions
# =============================================================================


_MAGNITUDE_COMPONENTS = {
    "raw_signals": ["Fx12", "Fx34", "Fy14", "Fy23", "Fz1", "Fz2", "Fz3", "Fz4"],
}


def _normalize_header_name(name: str) -> str:
    clean = str(name).strip()
    clean = re.sub(r"_duplicated_\d+$", "", clean)
    clean = re.sub(r"\.\d+$", "", clean)
    return clean


def _find_time_column_index(columns: list[str], time_var: str | None = None) -> int:
    normalized = [_normalize_header_name(col).lower() for col in columns]

    if time_var is not None:
        expected = _normalize_header_name(time_var).lower()
        if expected in normalized:
            return normalized.index(expected)
        raise ValueError(
            f"Expected time column '{time_var}' not found in input headers"
        )

    preferred = ["abs time (s)", "abs time", "time"]
    for key in preferred:
        if key in normalized:
            return normalized.index(key)

    for idx, col in enumerate(normalized):
        if "time" in col:
            return idx

    raise ValueError("Expected a time column (e.g. 'abs time (s)' or 'abs time')")


def _deduplicate_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in labels:
        count = counts.get(label, 0)
        if count == 0:
            unique_labels.append(label)
        else:
            unique_labels.append(f"{label}_{count + 1}")
        counts[label] = count + 1
    return unique_labels


def _ordered_unique(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _select_sensor_columns(columns: list[str]) -> list[str]:
    sensor_names = set(_MAGNITUDE_COMPONENTS["raw_signals"])
    return [col for col in columns if _normalize_header_name(col) in sensor_names]


def _select_non_sensor_columns(columns: list[str]) -> list[str]:
    sensor_names = set(_MAGNITUDE_COMPONENTS["raw_signals"])
    return [col for col in columns if _normalize_header_name(col) not in sensor_names]


def _reorder_with_time_first(columns: list[str], time_col: str | None) -> list[str]:
    if time_col is None:
        return columns
    if time_col not in columns:
        return columns
    return [time_col] + [col for col in columns if col != time_col]


def _mangle_duplicate_columns_pandas_style(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        count = counts.get(col, 0)
        if count == 0:
            out.append(col)
        else:
            out.append(f"{col}.{count}")
        counts[col] = count + 1
    return out


def _select_column_indices(
    columns: list[str],
    raw: bool,
    time_idx: int | None,
) -> list[int]:
    sensor_names = set(_MAGNITUDE_COMPONENTS["raw_signals"])

    if raw:
        data_indices = [
            idx
            for idx, col in enumerate(columns)
            if _normalize_header_name(col) in sensor_names
        ]
    else:
        data_indices = [
            idx
            for idx, col in enumerate(columns)
            if _normalize_header_name(col) not in sensor_names and idx != time_idx
        ]

    if time_idx is not None:
        return [time_idx] + data_indices
    return data_indices


def _extract_kistler_dataarray_from_numpy(
    values: np.ndarray,
    columns: list[str],
    time_values: np.ndarray | None = None,
    time_var: str | None = None,
) -> xr.DataArray:
    if len(columns) != values.shape[1]:
        raise ValueError("Columns length does not match values shape")

    try:
        time_idx = _find_time_column_index(columns, time_var=time_var)
    except ValueError:
        time_idx = None

    if time_values is None:
        if time_idx is not None:
            time_values = np.asarray(values[:, time_idx], dtype=float)
        else:
            time_values = np.arange(values.shape[0], dtype=float)

    if time_values.size < 2:
        raise ValueError("Unable to build DataArray: not enough samples")

    variable_idx = [idx for idx in range(values.shape[1]) if idx != time_idx]
    if not variable_idx:
        raise ValueError("Unable to build DataArray: no data variables found")

    base_labels = [_normalize_header_name(columns[idx]) for idx in variable_idx]
    var_names = _ordered_unique(base_labels)
    var_to_indices: dict[str, list[int]] = {name: [] for name in var_names}
    for idx, base_name in zip(variable_idx, base_labels):
        var_to_indices[base_name].append(idx)

    n_plates = max(len(indices) for indices in var_to_indices.values())
    data = np.full((n_plates, len(var_names), values.shape[0]), np.nan, dtype=float)

    values_float = np.asarray(values, dtype=float)
    for var_pos, var_name in enumerate(var_names):
        indices = var_to_indices[var_name]
        for plate_pos, col_idx in enumerate(indices):
            data[plate_pos, var_pos, :] = values_float[:, col_idx]

    da = xr.DataArray(
        data=data,
        dims=("plate", "n_var", "time"),
        coords={
            "plate": np.arange(1, n_plates + 1),
            "n_var": var_names,
            "time": time_values,
        },
        name="Kistler",
    )

    if time_values.size > 1:
        delta_time = np.diff(time_values)
        if not np.allclose(delta_time, 0):
            da.attrs["freq"] = round(1.0 / float(np.median(delta_time)), 1)
            da.time.attrs["units"] = "s"

    return da


def read_kistler_txt(
    file: str | Path,
    engine="arrow",
    **kwargs,
) -> Any:
    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    if engine == "polars":
        ret = read_kistler_txt_pl(
            file,
            **kwargs,
        )

    elif engine == "pandas":
        ret = read_kistler_txt_pd(
            file,
            **kwargs,
        )

    elif engine == "arrow":
        ret = read_kistler_txt_arrow(
            file,
            **kwargs,
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
    n_vars_load: list[str] | None = None,
    raw: bool = False,
    separator: str = "\t",
    to_dataarray: bool | None = True,
    time_var: str | None = "abs time",
    **kwargs,
) -> xr.DataArray | Any:
    """
    Read .txt files exported from Bioware

    Note: since BioWare Version 5.6.1.0 time column changed from "abs time (s)" to "abs time"
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "Polars package not instaled. Install it if you want to use the accelerated version"
        )

    """
    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")
    """

    if n_vars_load is not None and time_var not in n_vars_load:
        n_vars_load = [time_var] + n_vars_load

    try:
        if "magnitude" in kwargs:
            warnings.warn(
                "'magnitude' is ignored in read_kistler_txt; output now includes all loaded variables.",
                DeprecationWarning,
                stacklevel=2,
            )

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

        time_col = None
        try:
            time_idx = _find_time_column_index(df.columns, time_var=time_var)
            time_col = df.columns[time_idx]
            time_values = np.asarray(df[time_col].to_numpy(), dtype=float)
        except ValueError:
            if time_var is not None:
                raise
            time_idx = None
            time_values = None

        if raw:
            data_columns = _select_sensor_columns(df.columns)
            if not data_columns:
                raise ValueError("No sensor variables were found in the input file")
            selected_columns = data_columns
            if time_idx is not None:
                selected_columns = [time_col] + selected_columns
            df = df.select(selected_columns)
        else:
            data_columns = _select_non_sensor_columns(df.columns)
            selected_columns = _reorder_with_time_first(data_columns, time_col)
            df = df.select(selected_columns)

        if to_dataarray:
            da = _extract_kistler_dataarray_from_numpy(
                values=df.to_numpy(),
                columns=df.columns,
                time_values=time_values,
                time_var=time_var,
            )
            return da
        else:
            return df

    except Exception as err:
        raise RuntimeError(f"Unable to read {file.name}: {err}") from err


def read_kistler_txt_pd(
    file: str | Path,
    lin_header: int = 17,
    separator: str = "\t",
    n_vars_load: list[str] | None = None,
    raw: bool = False,
    to_dataarray: bool | None = True,
    time_var: str | None = "abs time",
    **kwargs,
) -> xr.DataArray | pd.DataFrame | None:
    """
    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")
    """

    if n_vars_load is not None and time_var not in n_vars_load:
        n_vars_load = [time_var] + n_vars_load

    try:
        if "magnitude" in kwargs:
            warnings.warn(
                "'magnitude' is ignored in read_kistler_txt; output now includes all loaded variables.",
                DeprecationWarning,
                stacklevel=2,
            )

        df = (
            pd.read_csv(
                file,
                header=lin_header,
                usecols=n_vars_load,  # ['Fx', 'Fy', 'Fz', 'Fx.1', 'Fy.1', 'Fz.1'], #n_vars_load,
                # skiprows=18,
                delimiter=separator,
                # dtype=np.float64,
                engine="c",  # "pyarrow" con pyarrow no funciona bien de momento cargar columnas con nombre repetido,
            ).drop(index=0)
        ).astype(float)

        try:
            time_idx = _find_time_column_index(df.columns.tolist(), time_var=time_var)
            time_values = np.asarray(df.iloc[:, time_idx], dtype=float)
        except ValueError:
            if time_var is not None:
                raise
            time_idx = None
            time_values = None

        if raw:
            data_columns = _select_sensor_columns(df.columns.tolist())
            if not data_columns:
                raise ValueError("No sensor variables were found in the input file")
            selected_columns = data_columns
            if time_idx is not None:
                time_col = df.columns[time_idx]
                selected_columns = [time_col] + selected_columns
            df = df.loc[:, selected_columns]
        else:
            data_columns = _select_non_sensor_columns(df.columns.tolist())
            time_col = df.columns[time_idx] if time_idx is not None else None
            selected_columns = _reorder_with_time_first(data_columns, time_col)
            df = df.loc[:, selected_columns]

        if to_dataarray:
            da = _extract_kistler_dataarray_from_numpy(
                values=df.to_numpy(),
                columns=df.columns.tolist(),
                time_values=time_values,
                time_var=time_var,
            )
            return da

        return df

    except Exception as err:
        raise RuntimeError(f"Unable to read {file.name}: {err}") from err


def read_kistler_txt_arrow(
    file: str | Path,
    lin_header: int = 17,
    separator: str = "\t",
    n_vars_load: list[str] | None = None,
    raw: bool = False,
    to_dataarray: bool | None = True,
    time_var: str | None = "abs time",
    **kwargs,
) -> pd.DataFrame | xr.DataArray:
    """Testing, at the moment it does not work when there are repeated cols"""

    try:
        from pyarrow import csv
    except ImportError:
        raise ImportError("pyarrow not installed")

    """
    if isinstance(file, str):
        file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")
    """

    if n_vars_load is not None and time_var not in n_vars_load:
        n_vars_load = [time_var] + n_vars_load

    try:
        if "magnitude" in kwargs:
            warnings.warn(
                "'magnitude' is ignored in read_kistler_txt; output now includes all loaded variables.",
                DeprecationWarning,
                stacklevel=2,
            )

        read_options = csv.ReadOptions(
            # column_names=['Fx', 'Fy', 'Fz'],
            skip_rows=lin_header,
            skip_rows_after_names=1,
        )
        parse_options = csv.ParseOptions(delimiter=separator)
        data = csv.read_csv(
            file, read_options=read_options, parse_options=parse_options
        )

        df = data.to_pandas()

        # Emulate pandas usecols behavior with duplicated headers (Fx/Fx.1, ...).
        if n_vars_load is not None:
            raw_columns = df.columns.tolist()
            mangled_columns = _mangle_duplicate_columns_pandas_style(raw_columns)
            selected_indices = [
                idx for idx, name in enumerate(mangled_columns) if name in n_vars_load
            ]
            if not selected_indices:
                requested = ", ".join(n_vars_load)
                raise ValueError(
                    f"n_vars_load did not match any column in file: {requested}"
                )
            df = df.iloc[:, selected_indices]
            df.columns = [mangled_columns[idx] for idx in selected_indices]

        try:
            time_idx = _find_time_column_index(df.columns.tolist(), time_var=time_var)
            time_values = np.asarray(df.iloc[:, time_idx], dtype=float)
        except ValueError:
            if time_var is not None:
                raise
            time_idx = None
            time_values = None

        columns = df.columns.tolist()
        selected_indices = _select_column_indices(
            columns=columns,
            raw=raw,
            time_idx=time_idx,
        )

        if raw and (time_idx is None and not selected_indices):
            raise ValueError("No sensor variables were found in the input file")
        if raw and (time_idx is not None and len(selected_indices) == 1):
            raise ValueError("No sensor variables were found in the input file")

        if not selected_indices:
            raise ValueError("No variables selected from input file")

        df = df.iloc[:, selected_indices]
        df.columns = _mangle_duplicate_columns_pandas_style(df.columns.tolist())

        if raw:
            data_columns = _select_sensor_columns(df.columns.tolist())
            if not data_columns:
                raise ValueError("No sensor variables were found in the input file")

        if to_dataarray:
            da = _extract_kistler_dataarray_from_numpy(
                values=df.to_numpy(),
                columns=df.columns.tolist(),
                time_values=time_values,
                time_var=time_var,
            )
            return da

        return df

    except Exception as err:
        raise RuntimeError(f"Unable to read {file.name}: {err}") from err


def split_plataforms(da: xr.DataArray) -> xr.DataArray:
    if "plate" in da.dims:
        return da

    if "platform" in da.dims:
        return da.rename({"platform": "plate"})

    if "n_var" not in da.dims:
        raise ValueError("Expected 'n_var' dimension to split platforms")

    n_var_values = [str(v) for v in da.coords["n_var"].values.tolist()]

    has_legacy_prefix = all(name.startswith(("F1", "F2")) for name in n_var_values)
    if has_legacy_prefix:
        plat1 = da.sel(n_var=da.n_var.str.startswith("F1"))
        plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip("F1"))
        plat2 = da.sel(n_var=da.n_var.str.startswith("F2"))
        plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip("F2"))
        return xr.concat([plat1, plat2], dim="plate").assign_coords(plate=[1, 2])

    labels = [_normalize_header_name(v) for v in n_var_values]
    labels = [re.sub(r"_\d+$", "", label) for label in labels]
    unique_labels = _ordered_unique(labels)
    n_time = da.sizes["time"]

    index_map: dict[str, list[int]] = {name: [] for name in unique_labels}
    for i, name in enumerate(labels):
        index_map[name].append(i)

    n_plates = max(len(indices) for indices in index_map.values())
    out = np.full((n_plates, len(unique_labels), n_time), np.nan, dtype=float)

    source = np.asarray(da.values, dtype=float)
    for v_pos, name in enumerate(unique_labels):
        for p_pos, src_idx in enumerate(index_map[name]):
            out[p_pos, v_pos, :] = source[src_idx, :]

    return xr.DataArray(
        out,
        dims=("plate", "n_var", "time"),
        coords={
            "plate": np.arange(1, n_plates + 1),
            "n_var": unique_labels,
            "time": da.time.values,
        },
        name=da.name,
        attrs=da.attrs,
    )


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

    if "plate" not in da.coords and "platform" not in da.coords:
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

    # Add source directory to import biomdp package
    # import sys
    # sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))
    DATASETS_DIR = Path(__file__).resolve().parents[0] / "tests" / "datasets"
    # DATASETS_DIR = Path(r"F:\Programacion\Python\Mios\biomdp\src\biomdp\tests\datasets")

    file = DATASETS_DIR / "kistler_DJ_2plates.txt"
    daForce = read_kistler_txt(file)
    daForce.plot.line(x="time", col="plate")

    plForce = read_kistler_txt(file, raw=False, engine="polars", to_dataarray=False)
    plt.plot(plForce)

    plForce = read_kistler_txt(file, raw=True, engine="polars", to_dataarray=False)
    plt.plot(plForce)

    pdForce = read_kistler_txt(file, raw=True, engine="pandas", to_dataarray=False)
    pdForce.plot(x="abs time")

    arrForce = read_kistler_txt(file, raw=True, engine="arrow", to_dataarray=False)
    arrForce.plot(x="abs time")

    file = DATASETS_DIR / "kistler_CMJ_1plate.txt"
    daForce = read_kistler_txt(file)
    daForce.plot.line(x="time", col="plate")

    file = DATASETS_DIR / "kistler_DJ_2plates.txt"
    daForce = read_kistler_txt(file)
    daForce.plot.line(x="time", col="plate")
    daForce.sum(dim="plate").plot.line(x="time")

    daForce = read_kistler_txt(file, engine="pandas")
    daForce.plot.line(x="time", col="plate")

    daForce = read_kistler_txt(file, engine="arrow")
    daForce.plot.line(x="time", col="plate")

    dfForce = read_kistler_txt(file, engine="pandas", to_dataarray=False)
    dfForce[["Fx", "Fy", "Fz"]].plot()

    file = DATASETS_DIR / "kistler_COP.txt"
    daForce = read_kistler_txt(file, n_vars_load=["Ax", "Ay"], engine="arrow")
    daForce.to_dataset("n_var").plot.scatter(x="Ax", y="Ay", col="plate")

    daForce = read_kistler_txt(file, engine="arrow")
    daForce.to_dataset("n_var").plot.scatter(x="Ax", y="Ay", col="plate")

    pdForce = read_kistler_txt(
        file, n_vars_load=["Ax", "Ay"], engine="arrow", to_dataarray=False
    )
    pdForce.plot.scatter(x="Ax", y="Ay")

    # -------------------------------
    import timeit

    file = DATASETS_DIR / "kistler_DJ_2plates.txt"

    def test_performance():
        result = read_kistler_txt(file, engine="polars")
        return result

    print(
        f"{timeit.timeit('test_performance()', setup='from __main__ import test_performance', number=50):.4f} s"
    )

    def test_performance():
        result = read_kistler_txt(file, engine="pandas")
        return result

    print(
        f"{timeit.timeit('test_performance()', setup='from __main__ import test_performance', number=50):.4f} s"
    )

    def test_performance():
        result = read_kistler_txt(file, engine="arrow")
        return result

    print(
        f"{timeit.timeit('test_performance()', setup='from __main__ import test_performance', number=50):.4f} s"
    )

    def test_performance():
        result = read_kistler_txt(file, engine="polars", raw=True)
        return result

    print(
        f"{timeit.timeit('test_performance()', setup='from __main__ import test_performance', number=50):.4f} s"
    )

    # %% --------------------------------------------------------------------------
    DATASETS_DIR = Path(__file__).resolve().parents[0] / "tests" / "datasets"
    # DATASETS_DIR = Path(r"F:\Programacion\Python\Mios\biomdp\src\biomdp\tests\datasets")

    engines = ["polars", "pandas", "arrow"]
    demo_files = [
        "kistler_DJ_2plates.txt",
        "kistler_CMJ_1plate.txt",
        "kistler_COP.txt",
    ]

    print(f"Dataset directory: {DATASETS_DIR}")
    print("=== DEMO: read_kistler_txt magnitudes ===")

    for filename in demo_files:
        file = DATASETS_DIR / filename
        print(f"\nFile: {filename}")

        for engine in engines:
            try:
                da = read_kistler_txt(
                    file=file,
                    engine=engine,
                    raw=False,
                )
                print(
                    f"  OK  engine={engine:6s}"
                    f"plates={da.sizes['plate']} axis={list(da.axis.values)}"
                )
            except Exception as err:
                print(f"  --  engine={engine:6s} not available ({err})")

    # Keep explicit demonstration of raw=True native return type per engine.
    sample_file = DATASETS_DIR / "kistler_DJ_2plates.txt"
    print("\n=== DEMO: raw=True native return type ===")
    for engine in engines:
        try:
            raw_data = read_kistler_txt(sample_file, engine=engine, raw=True)
            print(f"  engine={engine:6s} raw_type={type(raw_data).__name__}")
        except Exception as err:
            print(f"  engine={engine:6s} raw_type=ERROR ({err})")
