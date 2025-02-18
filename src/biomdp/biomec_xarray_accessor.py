"""
Created on Thu Feb 22 16:17:08 2024

@author: josel
"""

from typing import Optional, Union, Any
import xarray as xr


__author__ = "Jose Luis Lopez Elvira"
__version__ = "v.0.2.0"
__date__ = "07/12/2024"


"""
Modificaciones:
    31/10/2024, v.0.2.0
        - Incluidas funcionnes nanargmax_xr y nanargmin_xr.
    
    31/10/2024, v.0.1.0
        - Incluida función trim_window.
    
    15/03/2024, v.0.0.1
        - Permite también Dataset. De momento probado sólo con filtrar_butter.
          Hay que meter en el return apply_ufunc y probar que funciona.
        
    
"""


@xr.register_dataset_accessor("biomxr")
@xr.register_dataarray_accessor("biomxr")
class DataArrayAccessor:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj

    def nanargmax_xr(self, dim: str = None) -> xr.DataArray:
        """
        Necesario porque la versión argmax de xarray no funciona bien con nans intermedios
        """
        from biomdp.general_processing_functions import nanargmax_xr

        return nanargmax_xr(self._obj, dim=dim)

    def nanargmin_xr(self, dim: str = None) -> xr.DataArray:
        """
        Necesario porque la versión argmax de xarray no funciona bien con nans intermedios
        """
        from biomdp.general_processing_functions import nanargmin_xr

        return nanargmin_xr(self._obj, dim=dim)

    def detect_events(
        self,
        freq: Optional[float] = None,
        n_dim_time: Optional[str] = "time",
        reference_var: Optional[Union[str, dict]] = None,
        discard_phases_ini: int = 0,
        n_phases: Optional[int] = None,
        discard_phases_end: int = 0,
        # include_first_next_last: Optional[bool] = False,
        max_phases: Optional[int] = 100,
        func_events: Optional[Any] = None,
        **kwargs_func_events,
    ) -> xr.DataArray:
        import biomdp.slice_time_series_phases as stsp

        return stsp.detect_events(
            data=self._obj,
            freq=freq,
            n_dim_time=n_dim_time,
            reference_var=reference_var,
            discard_phases_ini=discard_phases_ini,
            n_phases=n_phases,
            discard_phases_end=discard_phases_end,
            # include_first_next_last=include_first_next_last,
            max_phases=max_phases,
            func_events=func_events,
            **kwargs_func_events,
        )

    def slice_time_series(
        self,
        events: Optional[xr.DataArray] = None,
        freq: Optional[float] = None,
        n_dim_time: Optional[str] = "time",
        reference_var: Optional[Union[str, dict]] = None,
        discard_phases_ini: Optional[int] = 0,
        n_phases: Optional[int] = None,
        discard_phases_end: Optional[int] = 0,
        include_first_next_last: Optional[bool] = False,
        max_phases: Optional[int] = 100,
        func_events: Optional[Any] = None,
        split_version_function: Optional[str] = "polars",  # "polars" or "numpy"
        **kwargs_func_events,
    ) -> xr.DataArray:
        import biomdp.slice_time_series_phases as stsp

        return stsp.slice_time_series(
            data=self._obj,
            events=events,
            freq=freq,
            n_dim_time=n_dim_time,
            reference_var=reference_var,
            discard_phases_ini=discard_phases_ini,
            n_phases=n_phases,
            discard_phases_end=discard_phases_end,
            include_first_next_last=include_first_next_last,
            max_phases=max_phases,
            func_events=func_events,
            split_version_function=split_version_function,
            **kwargs_func_events,
        )

    def trim_window(
        self,
        daEvents: Optional[xr.DataArray] = None,
        window=None,
    ) -> xr.DataArray:
        from biomdp.slice_time_series_phases import trim_window

        return trim_window(
            daDatos=self._obj,
            daEvents=daEvents,
            window=window,
        )

    def filter_butter(
        self,
        fr: Optional[float] = None,
        fc: Optional[float] = None,
        order: Optional[float] = 2.0,
        kind: Optional[str] = "low",
        returnRMS: Optional[bool] = False,
        show: Optional[bool] = False,
        ax: Optional[object] = None,
    ) -> xr.DataArray:

        import numpy as np
        from biomdp.filtrar_Butter import filtrar_Butter

        if fr is None:
            if "freq" in self._obj.attrs:
                fr = self._obj.attrs["freq"]
            else:
                if not self._obj.isnull().all():
                    fr = (
                        np.round(
                            1 / (self._obj["time"][1] - self._obj["time"][0]),
                            1,
                        )
                    ).data

        return filtrar_Butter(
            dat_orig=self._obj,
            fr=fr,
            fc=fc,
            order=order,
            kind=kind,
            returnRMS=returnRMS,
            show=show,
            ax=ax,
        )

    def integrate_window(
        self,
        daWindow: Optional[xr.DataArray] = None,
        daOffset: Optional[xr.DataArray] = None,
        result_return: Optional[str] = "continuous",
    ) -> xr.DataArray:
        from biomdp.general_processing_functions import integrate_window

        return integrate_window(
            self._obj, daWindow=daWindow, daOffset=daOffset, result_return=result_return
        )

    def RMS(self, daWindow: Optional[xr.DataArray] = None) -> xr.DataArray:
        from biomdp.general_processing_functions import RMS

        return RMS(self._obj, daWindow=daWindow)
