"""Tools for the hydrostatic water-density teaching experiment.

The experimental model is

    M(V) = M0 + rho * (V - V0) = rho * V + b,

where the slope ``rho`` estimates the water density.  The regression implemented
here is ordinary least squares (OLS): it assumes that volume is controlled or
known with negligible error and that mass residuals are independent, centered,
and homoscedastic.  Displayed measurement error bars are not used as OLS weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


PathLike = Union[str, Path]


class WaterDensity:
    """Generate or load mass-volume data and estimate water density by OLS.

    Parameters
    ----------
    M0:
        Reference mass in grams.
    V0:
        Reference volume in millilitres.
    rho:
        Reference density in grams per millilitre.  This value is used only to
        generate synthetic data; measured density is obtained from the fitted
        slope.
    """

    REQUIRED_COLUMNS = ("M", "V")

    def __init__(self, M0: float = 300.0, V0: float = 100.0, rho: float = 1.0):
        self.M0 = self._finite_scalar(M0, "M0")
        self.V0 = self._finite_scalar(V0, "V0")
        self.rho = self._finite_scalar(rho, "rho")
        if self.rho <= 0:
            raise ValueError("rho must be positive.")
        self.df: Optional[pd.DataFrame] = None
        self.fit_results: Optional[dict] = None

    @staticmethod
    def _finite_scalar(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    @staticmethod
    def _validate_relative_error(value: float, name: str) -> float:
        value = WaterDensity._finite_scalar(value, name)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")
        return value

    def gen_fake_data(
        self,
        n_points: int = 10,
        V_min: float = 110.0,
        V_max: float = 200.0,
        err_M: float = 0.025,
        err_V: float = 0.10,
        noise: float = 5.0,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate reproducible synthetic observations.

        ``err_M`` and ``err_V`` are relative measurement uncertainties. ``noise``
        is the half-width, in grams, of a uniform mass perturbation.  Pass an
        integer ``random_state`` whenever the dataset must be reproducible.
        """

        if not isinstance(n_points, (int, np.integer)) or n_points < 3:
            raise ValueError("n_points must be an integer greater than or equal to 3.")
        V_min = self._finite_scalar(V_min, "V_min")
        V_max = self._finite_scalar(V_max, "V_max")
        noise = self._finite_scalar(noise, "noise")
        err_M = self._validate_relative_error(err_M, "err_M")
        err_V = self._validate_relative_error(err_V, "err_V")
        if V_max <= V_min:
            raise ValueError("V_max must be greater than V_min.")
        if noise < 0:
            raise ValueError("noise must be nonnegative.")

        volume = np.linspace(V_min, V_max, n_points, dtype=float)
        ideal_mass = self.M0 + self.rho * (volume - self.V0)
        rng = np.random.default_rng(random_state)
        mass = ideal_mass + rng.uniform(-noise, noise, n_points)

        return self.set_data(
            pd.DataFrame(
                {
                    "M": mass,
                    "sigma_M": np.abs(mass) * err_M,
                    "V": volume,
                    "sigma_V": np.abs(volume) * err_V,
                }
            )
        )

    def set_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Load measured or synthetic data after validating its schema."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        missing = [column for column in self.REQUIRED_COLUMNS if column not in data]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}.")
        if len(data) < 3:
            raise ValueError("At least three observations are required for OLS.")

        validated = data.copy()
        optional_columns = ("sigma_M", "sigma_V")
        for column in self.REQUIRED_COLUMNS + optional_columns:
            if column in validated:
                validated[column] = pd.to_numeric(validated[column], errors="raise")
                if not np.isfinite(validated[column].to_numpy(dtype=float)).all():
                    raise ValueError(f"Column {column} contains non-finite values.")
        for column in optional_columns:
            if column in validated and (validated[column] < 0).any():
                raise ValueError(f"Column {column} must be nonnegative.")
        if np.ptp(validated["V"].to_numpy(dtype=float)) == 0:
            raise ValueError("Volume values must not all be identical.")

        self.df = validated.reset_index(drop=True)
        self.fit_results = None
        return self.df.copy()

    def calculate_fit(self):
        """Fit ``M = a V + b`` by OLS and return summary and full results.

        The density estimate is the slope ``a``. Standard errors use the usual
        residual variance estimator with ``N - 2`` degrees of freedom.
        """

        if self.df is None:
            raise ValueError("No data available. Run gen_fake_data() or set_data() first.")

        volume = self.df["V"].to_numpy(dtype=float)
        mass = self.df["M"].to_numpy(dtype=float)
        n_obs = volume.size
        x_bar = float(volume.mean())
        y_bar = float(mass.mean())
        centered_volume = volume - x_bar
        sxx = float(np.dot(centered_volume, centered_volume))
        if sxx <= 0:
            raise ValueError("OLS is undefined when all volume values are identical.")

        slope = float(np.dot(centered_volume, mass - y_bar) / sxx)
        intercept = float(y_bar - slope * x_bar)
        predicted = slope * volume + intercept
        residuals = mass - predicted
        dof = n_obs - 2
        residual_std_error = float(np.sqrt(np.dot(residuals, residuals) / dof))
        slope_std_error = float(residual_std_error / np.sqrt(sxx))
        intercept_std_error = float(
            residual_std_error * np.sqrt(1.0 / n_obs + x_bar**2 / sxx)
        )
        total_sum_squares = float(np.dot(mass - y_bar, mass - y_bar))
        r_squared = (
            float(1.0 - np.dot(residuals, residuals) / total_sum_squares)
            if total_sum_squares > 0
            else float("nan")
        )

        self.fit_results = {
            "M0": self.M0,
            "V0": self.V0,
            "rho_reference": self.rho,
            "density": slope,
            "density_std_error": slope_std_error,
            "intercept": intercept,
            "intercept_std_error": intercept_std_error,
            "residual_std_error": residual_std_error,
            "r_squared": r_squared,
            "n_observations": n_obs,
            "degrees_of_freedom": dof,
            "x_mean": x_bar,
            "sxx": sxx,
            "predicted": predicted,
            "residuals": residuals,
            # Backward-compatible aliases used by the original notebooks.
            "a": slope,
            "sigma_a": slope_std_error,
            "b": intercept,
            "sigma_b": intercept_std_error,
            "sigma_y": residual_std_error,
            "y_pred": predicted,
        }
        summary_columns = (
            "density",
            "density_std_error",
            "intercept",
            "intercept_std_error",
            "residual_std_error",
            "r_squared",
            "n_observations",
        )
        summary = pd.DataFrame(
            [{column: self.fit_results[column] for column in summary_columns}]
        )
        return summary, self.fit_results.copy()

    def plot_regression(self, confidence_level: float = 0.95, show: bool = True):
        """Plot observations, OLS line, and a mean-response confidence band.

        The band uses the exact OLS covariance structure through
        ``s * sqrt(1/N + (V - mean(V))**2 / Sxx)`` and a Student-t critical
        value. It is a confidence band for the fitted mean, not a prediction
        interval for a future observation.
        """

        if self.df is None or self.fit_results is None:
            raise ValueError("Generate or load data and calculate the fit first.")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1.")

        volume = self.df["V"].to_numpy(dtype=float)
        mass = self.df["M"].to_numpy(dtype=float)
        grid = np.linspace(volume.min(), volume.max(), 300)
        slope = self.fit_results["density"]
        intercept = self.fit_results["intercept"]
        fitted_grid = slope * grid + intercept
        standard_error = self.fit_results["residual_std_error"] * np.sqrt(
            1.0 / self.fit_results["n_observations"]
            + (grid - self.fit_results["x_mean"]) ** 2 / self.fit_results["sxx"]
        )
        critical_value = t.ppf(
            0.5 + confidence_level / 2.0,
            self.fit_results["degrees_of_freedom"],
        )
        margin = critical_value * standard_error

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            volume,
            mass,
            xerr=self.df["sigma_V"] if "sigma_V" in self.df else None,
            yerr=self.df["sigma_M"] if "sigma_M" in self.df else None,
            fmt="o",
            color="black",
            capsize=2,
            label="Observations",
        )
        ax.plot(
            grid,
            fitted_grid,
            "--",
            color="tab:blue",
            label=(
                rf"OLS: $M = ({slope:.4f} \pm {self.fit_results['density_std_error']:.4f})V$"
                rf" $+ ({intercept:.2f} \pm {self.fit_results['intercept_std_error']:.2f})$"
            ),
        )
        ax.fill_between(
            grid,
            fitted_grid - margin,
            fitted_grid + margin,
            color="tab:blue",
            alpha=0.2,
            label=f"{confidence_level:.0%} confidence band for the fitted mean",
        )
        ax.set(xlabel="Volume (mL)", ylabel="Mass (g)", title="Mass–volume OLS fit")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def format_table(self) -> pd.DataFrame:
        """Return a publication-oriented table with optional uncertainties."""

        if self.df is None:
            raise ValueError("No data available. Run gen_fake_data() or set_data() first.")

        table = pd.DataFrame({r"$M_i\,(\mathrm{g})$": self.df["M"].map(lambda x: f"{x:.3f}")})
        if "sigma_M" in self.df:
            table[r"$\delta M_i\,(\mathrm{g})$"] = self.df["sigma_M"].map(lambda x: f"{x:.3f}")
        table[r"$V_i\,(\mathrm{mL})$"] = self.df["V"].map(lambda x: f"{x:.3f}")
        if "sigma_V" in self.df:
            table[r"$\delta V_i\,(\mathrm{mL})$"] = self.df["sigma_V"].map(lambda x: f"{x:.3f}")
        return table

    def export_latex_table(self, filename: Optional[PathLike] = None) -> str:
        """Return the LaTeX table and optionally write it as UTF-8 text."""

        latex_code = self.format_table().to_latex(escape=False, index=False)
        if filename is not None:
            output_path = Path(filename)
            output_path.write_text(latex_code, encoding="utf-8")
        return latex_code
