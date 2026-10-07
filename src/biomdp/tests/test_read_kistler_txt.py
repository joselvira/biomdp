import re
import sys
import warnings
from pathlib import Path

import pytest

# Add source directory to import biomdp package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from biomdp.io.read_kistler_txt import split_plataforms, read_kistler_txt


DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
# DATASETS_DIR = Path(r"F:\Programacion\Python\Mios\biomdp\src\biomdp\tests\datasets")

SENSOR_VARS = {"Fx12", "Fx34", "Fy14", "Fy23", "Fz1", "Fz2", "Fz3", "Fz4"}


def _normalize(name):
    clean = str(name).strip()
    clean = re.sub(r"_duplicated_\d+$", "", clean)
    clean = re.sub(r"\.\d+$", "", clean)
    return clean


def test_to_dataarray_true_returns_dataarray_pandas():
    da = read_kistler_txt(DATASETS_DIR / "kistler_DJ_2plates.txt", engine="pandas")

    assert da.__class__.__name__ == "DataArray"
    assert da.dims == ("plate", "n_var", "time")
    assert da.sizes["plate"] == 2
    assert all(
        "time" not in str(var).lower() for var in da.coords["n_var"].values.tolist()
    )


def test_split_plataforms_is_idempotent_for_plate_dataarray():
    da = read_kistler_txt(DATASETS_DIR / "kistler_DJ_2plates.txt", engine="pandas")
    da_split = split_plataforms(da)
    assert da_split.equals(da)


def test_raw_true_returns_native_pandas_dataframe():
    raw_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="pandas",
        raw=True,
        to_dataarray=False,
    )
    assert raw_df.__class__.__name__ == "DataFrame"
    assert "time" in _normalize(raw_df.columns[0]).lower()
    norm_cols = [_normalize(col) for col in raw_df.columns.tolist()[1:]]
    assert norm_cols
    assert all(col in SENSOR_VARS for col in norm_cols)


def test_raw_true_returns_native_polars_dataframe():
    pytest.importorskip("polars")
    raw_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="polars",
        raw=True,
        to_dataarray=False,
    )
    assert raw_df.__class__.__name__ == "DataFrame"
    assert "time" in _normalize(raw_df.columns[0]).lower()
    norm_cols = [_normalize(col) for col in raw_df.columns[1:]]
    assert norm_cols
    assert all(col in SENSOR_VARS for col in norm_cols)


def test_raw_true_returns_native_arrow_dataframe():
    pytest.importorskip("pyarrow")
    raw_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="arrow",
        raw=True,
        to_dataarray=False,
    )
    assert raw_df.__class__.__name__ == "DataFrame"
    assert "time" in _normalize(raw_df.columns[0]).lower()
    norm_cols = [_normalize(col) for col in raw_df.columns.tolist()[1:]]
    assert norm_cols
    assert all(col in SENSOR_VARS for col in norm_cols)


def test_raw_false_returns_all_variables_pandas_dataframe():
    full_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="pandas",
        raw=False,
        to_dataarray=False,
    )
    norm_cols = [_normalize(col) for col in full_df.columns.tolist()]
    assert "time" in norm_cols[0].lower()
    data_cols = [col for col in norm_cols[1:]]
    assert data_cols
    assert not any(col in SENSOR_VARS for col in data_cols)


def test_arrow_dataframe_columns_match_pandas_for_two_platform_file():
    pytest.importorskip("pyarrow")
    pd_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="pandas",
        raw=False,
        to_dataarray=False,
    )
    arrow_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="arrow",
        raw=False,
        to_dataarray=False,
    )

    assert pd_df.columns.tolist() == arrow_df.columns.tolist()


def test_arrow_raw_dataframe_columns_match_pandas_for_two_platform_file():
    pytest.importorskip("pyarrow")
    pd_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="pandas",
        raw=True,
        to_dataarray=False,
    )
    arrow_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="arrow",
        raw=True,
        to_dataarray=False,
    )

    assert pd_df.columns.tolist() == arrow_df.columns.tolist()


def test_arrow_n_vars_load_matches_pandas_primary_platform():
    pytest.importorskip("pyarrow")
    selected = ["Fx", "Fy", "Fz"]

    pd_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="pandas",
        raw=False,
        to_dataarray=False,
        n_vars_load=selected,
    )
    arrow_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="arrow",
        raw=False,
        to_dataarray=False,
        n_vars_load=selected,
    )

    assert pd_df.columns.tolist() == arrow_df.columns.tolist()


def test_arrow_n_vars_load_matches_pandas_secondary_platform():
    pytest.importorskip("pyarrow")
    selected = ["Fx.1", "Fy.1", "Fz.1"]

    pd_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="pandas",
        raw=False,
        to_dataarray=False,
        n_vars_load=selected,
    )
    arrow_df = read_kistler_txt(
        DATASETS_DIR / "kistler_DJ_2plates.txt",
        engine="arrow",
        raw=False,
        to_dataarray=False,
        n_vars_load=selected,
    )

    assert pd_df.columns.tolist() == arrow_df.columns.tolist()


def test_time_var_parameter_accepts_alternate_header_name():
    da = read_kistler_txt(
        DATASETS_DIR / "kistler_COP.txt",
        engine="pandas",
        to_dataarray=True,
        time_var="abs time (s)",
    )
    assert da.__class__.__name__ == "DataArray"


def test_time_var_parameter_raises_if_not_present():
    with pytest.raises(RuntimeError, match="Expected time column"):
        read_kistler_txt(
            DATASETS_DIR / "kistler_COP.txt",
            engine="pandas",
            to_dataarray=False,
            time_var="time_seconds",
        )


def test_magnitude_is_ignored_for_backward_compatibility():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_kistler_txt(
            DATASETS_DIR / "kistler_DJ_2plates.txt",
            engine="pandas",
            magnitude="forces",
            to_dataarray=False,
        )

    assert result.__class__.__name__ == "DataFrame"
    assert any("magnitude" in str(w.message).lower() for w in caught)
