import unittest
import numpy as np
import xarray as xr
from biomdp.slice_time_series_phases import (
    detect_onset_detecta_aux,
    find_peaks_aux,
    detect_events,
    slice_time_series,
    trim_window,
    SliceTimeSeriesPhases,
)
from biomdp.create_time_series import create_time_series_xr


class TestSliceTimeSeriesPhases(unittest.TestCase):

    def setUp(self):
        # Setup code to create sample data
        self.rnd_seed = np.random.seed(12340)
        self.n = 5
        self.duration = 5
        self.freq = 200.0

        self.Pre_a = (
            create_time_series_xr(
                rnd_seed=self.rnd_seed,
                num_subj=self.n,
                fs=self.freq,
                IDini=0,
                range_offset=[25, 29],
                range_amp=[40, 45],
                range_freq=[1.48, 1.52],
                range_af=[0, 30],
                amplific_noise=[0.4, 0.7],
                fc_noise=[3.0, 3.5],
                range_duration=[self.duration, self.duration],
            )
            .expand_dims({"n_var": ["a"], "moment": ["pre"]})
            .transpose("ID", "moment", "n_var", "time")
        )
        self.Post_a = (
            create_time_series_xr(
                rnd_seed=self.rnd_seed,
                num_subj=self.n,
                fs=self.freq,
                IDini=0,
                range_offset=[22, 26],
                range_amp=[36, 40],
                range_freq=[1.48, 1.52],
                range_af=[0, 30],
                amplific_noise=[0.4, 0.7],
                fc_noise=[3.0, 3.5],
                range_duration=[self.duration, self.duration],
            )
            .expand_dims({"n_var": ["a"], "moment": ["post"]})
            .transpose("ID", "moment", "n_var", "time")
        )
        self.var_a = xr.concat([self.Pre_a, self.Post_a], dim="moment")

        self.daTotal = xr.concat([self.var_a], dim="n_var")
        self.daTotal.name = "Angle"
        self.daTotal.attrs["freq"] = 1 / (
            self.daTotal.time[1].values - self.daTotal.time[0].values
        )
        self.daTotal.attrs["units"] = "deg"
        self.daTotal.time.attrs["units"] = "s"

    def test_detect_events(self):
        daEvents = detect_events(
            data=self.daTotal, func_events=detect_onset_detecta_aux
        )
        self.assertIsInstance(daEvents, xr.DataArray)

    def test_slice_time_series(self):
        daEvents = detect_events(
            data=self.daTotal, func_events=detect_onset_detecta_aux
        )
        dacuts = slice_time_series(self.daTotal, daEvents)
        self.assertIsInstance(dacuts, xr.DataArray)

    def test_trim_window(self):
        daEvents = detect_events(
            data=self.daTotal, func_events=detect_onset_detecta_aux
        )
        window = xr.concat(
            [daEvents.min("n_event"), daEvents.max("n_event")], dim="n_event"
        )
        daTrimed = trim_window(self.daTotal, window)
        self.assertIsInstance(daTrimed, xr.DataArray)


if __name__ == "__main__":
    unittest.main()
