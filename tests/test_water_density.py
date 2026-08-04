import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from class_water_density import WaterDensity


class WaterDensityTest(unittest.TestCase):
    def test_synthetic_data_are_reproducible(self):
        first = WaterDensity(M0=200, V0=100, rho=0.997).gen_fake_data(random_state=42)
        second = WaterDensity(M0=200, V0=100, rho=0.997).gen_fake_data(random_state=42)
        pd.testing.assert_frame_equal(first, second)

    def test_exact_linear_data_recover_density(self):
        volume = np.linspace(100.0, 200.0, 11)
        mass = 0.997 * volume + 100.3
        model = WaterDensity()
        model.set_data(pd.DataFrame({"V": volume, "M": mass}))
        summary, results = model.calculate_fit()
        self.assertAlmostEqual(results["density"], 0.997, places=12)
        self.assertAlmostEqual(results["intercept"], 100.3, places=12)
        self.assertAlmostEqual(results["r_squared"], 1.0, places=12)
        self.assertEqual(summary.loc[0, "n_observations"], 11)

    def test_fit_agrees_with_numpy(self):
        model = WaterDensity(M0=200, V0=100, rho=0.997)
        data = model.gen_fake_data(noise=2, random_state=42)
        _, results = model.calculate_fit()
        slope, intercept = np.polyfit(data["V"], data["M"], 1)
        self.assertAlmostEqual(results["density"], slope, places=12)
        self.assertAlmostEqual(results["intercept"], intercept, places=12)

    def test_new_data_clear_previous_fit(self):
        model = WaterDensity()
        model.gen_fake_data(random_state=1)
        model.calculate_fit()
        model.gen_fake_data(random_state=2)
        self.assertIsNone(model.fit_results)

    def test_invalid_inputs_are_rejected(self):
        model = WaterDensity()
        with self.assertRaises(ValueError):
            model.gen_fake_data(n_points=2)
        with self.assertRaises(ValueError):
            model.gen_fake_data(V_min=2, V_max=1)
        with self.assertRaises(ValueError):
            model.set_data(pd.DataFrame({"V": [1, 1, 1], "M": [1, 2, 3]}))

    def test_plot_returns_figure_and_axes(self):
        model = WaterDensity()
        model.gen_fake_data(random_state=42)
        model.calculate_fit()
        figure, axes = model.plot_regression(show=False)
        self.assertEqual(axes.get_xlabel(), "Volume (mL)")
        figure.clear()


if __name__ == "__main__":
    unittest.main()
