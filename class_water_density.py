import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WaterDensity:
    """
    A class to simulate experimental data for mass vs. volume measurements,
    fit a linear model using least squares regression, and visualize the results.
    """

    def __init__(self, M0=300, V0=100, rho=1):
        """
        Initialize the WaterDensity object with reference values.

        Parameters:
        M0 : float - Reference mass (g)
        V0 : float - Reference volume (mL)
        rho : float - Density (g/mL)
        """
        self.M0 = M0
        self.V0 = V0
        self.rho = rho
        self.df = None  # DataFrame to store generated data
        self.fit_results = None  # Dictionary to store regression results

    def gen_fake_data(self, n_points=10, V_min=110, V_max=200, err_M=0.025, err_V=0.10, noise=5):
        """
        Generate synthetic mass vs. volume data with uncertainties.

        Parameters:
        n_points : int - Number of data points
        V_min : float - Minimum volume value
        V_max : float - Maximum volume value
        err_M : float - Relative uncertainty in mass
        err_V : float - Relative uncertainty in volume
        noise : float - Random noise added to mass values

        Returns:
        pd.DataFrame containing mass, volume, and their uncertainties.
        """
        # Generate volume values evenly spaced between V_min and V_max
        V_values = np.linspace(V_min, V_max, n_points)
        # Compute mass values using a linear relationship + noise
        M_values = self.rho * V_values + (self.M0 - self.rho * self.V0) + np.random.uniform(-noise, noise, n_points)
        # Create dataframe with uncertainties in mass and volume
        self.df = pd.DataFrame({
            'M': M_values,
            'sigma_M': M_values * err_M,
            'V': V_values,
            'sigma_V': V_values * err_V     
        })
        return self.df

    def calculate_fit(self):
        """
        Perform least squares regression (Mass vs. Volume) on generated data.

        Returns:
        pd.DataFrame and dict containing slope, intercept, uncertainties, and predictions.
        """
        if self.df is None:
            raise ValueError("No data available. Please run gen_fake_data() first.")
        
        # Extract mass and volume data
        V_values = self.df['V'].values
        M_values = self.df['M'].values
        N = len(V_values)

        # Compute necessary sums for least squares formulas
        sum_x = np.sum(V_values)
        sum_y = np.sum(M_values)
        sum_x2 = np.sum(V_values ** 2)
        sum_xy = np.sum(V_values * M_values)

        # Calculate slope (a) and intercept (b)
        D = N * sum_x2 - sum_x ** 2
        a = (N * sum_xy - sum_x * sum_y) / D
        b = (sum_y - a * sum_x) / N

        # Predicted mass values from regression
        y_pred = a * V_values + b

        # Calculate residual standard error
        sigma_y = np.sqrt(np.sum((M_values - y_pred) ** 2) / (N - 2))

        # Calculate uncertainty in slope and intercept
        Sxx = sum_x2 - (sum_x ** 2) / N
        sigma_a = sigma_y / np.sqrt(Sxx)
        x_bar = sum_x / N
        sigma_b = sigma_y * np.sqrt(1 / N + (x_bar ** 2) / Sxx)

        # Store results
        self.fit_results = {
            'M0': self.M0,
            'V0': self.V0,
            'rho': self.rho,
            'a': a,
            'sigma_a': sigma_a,
            'b': b,
            'sigma_b': sigma_b,
            'y_pred': y_pred,
            'sigma_y': sigma_y,
        }
        return pd.DataFrame([self.fit_results]), self.fit_results

    def plot_regression(self):
        """
        Plot experimental data with error bars, regression line, and uncertainty band.
        """
        if self.df is None or self.fit_results is None:
            raise ValueError("You must generate data and calculate fit first.")
        
        # Extract variables and errors
        V_values = self.df['V'].values
        M_values = self.df['M'].values
        V_err = self.df['sigma_V'].values
        M_err = self.df['sigma_M'].values

        # Regression results
        y_pred = self.fit_results['y_pred']
        a = self.fit_results['a']
        b = self.fit_results['b']
        sigma_a = self.fit_results['sigma_a']
        sigma_b = self.fit_results['sigma_b']

        # Sort for plotting a smooth regression line
        sort_idx = np.argsort(V_values)
        V_sorted = V_values[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        # Propagate uncertainties to compute the uncertainty band
        uncertainty = np.sqrt((V_sorted * sigma_a) ** 2 + sigma_b ** 2)
        y_upper = y_pred_sorted + uncertainty
        y_lower = y_pred_sorted - uncertainty

        # Plot experimental data with error bars
        plt.figure(figsize=(8, 5))
        plt.errorbar(V_values, M_values, xerr=V_err, yerr=M_err,
                     fmt='o', color='black', capsize=2, label="Experimental Data")
        # Plot regression line
        plt.plot(V_sorted, y_pred_sorted, linestyle='--', color='b',
                 label=f"Best Fit: y = ({a:.4f} \\pm {sigma_a:.4f})x + ({b:.2f} \\pm {sigma_b:.2f})")
        # Plot uncertainty band
        plt.fill_between(V_sorted, y_lower, y_upper, color='blue', alpha=0.2,
                         label='Uncertainty Band')
        plt.xlabel("Volume (mL)")
        plt.ylabel("Mass (g)")
        plt.title("Least Squares Regression: Mass vs. Volume")
        plt.grid(True)
        plt.legend()
        plt.show()

    def format_table(self):
        """
        Format the dataframe into a LaTeX-style table with uncertainties.

        Returns:
        pd.DataFrame formatted with \\pm symbols and fractional uncertainties.
        """
        if self.df is None:
            raise ValueError("No data to format. Please run gen_fake_data() first.")
        
        return pd.DataFrame({
            "$(M_i \\pm \\delta M_i) \\ g$": [f"{M:.3f} \\pm {sigma_M:.3f}" for M, sigma_M in zip(self.df["M"], self.df["sigma_M"])],
            "$\\frac{\\delta M_i}{M_i}$": [f"{(sigma_M / M):.3f}" for M, sigma_M in zip(self.df["M"], self.df["sigma_M"])],
            "$(V_i \\pm \\delta V_i) \\ ml$": [f"{V:.3f} \\pm {sigma_V:.3f}" for V, sigma_V in zip(self.df["V"], self.df["sigma_V"])],
            "$\\frac{\\delta V_i}{V_i}$": [f"{(sigma_V / V):.3f}" for V, sigma_V in zip(self.df["V"], self.df["sigma_V"])]
        })
    
    def export_latex_table(self, filename=None):
        """
        Generate LaTeX code for the formatted table.

        Parameters:
        filename : str or None
            - If None: prints the LaTeX table directly to the screen.
            - If str: saves the LaTeX table to the specified .tex file.

        Returns:
        str : The LaTeX table code.
        """
        table = self.format_table()
        latex_code = table.to_latex(escape=False, index=False)

        if filename:
            with open(filename, "w") as f:
                f.write(latex_code)
            print(f"LaTeX table exported to {filename}")
        else:
            print(latex_code)

        return latex_code
