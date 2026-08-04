"""
gaussian_fitting.py
Gaussian fitting utilities for pump-probe space profile analysis
"""

import numpy as np
from scipy.optimize import curve_fit


class GaussianFitter:
    """Class for fitting Gaussian curves to profile data."""
    
    def __init__(self):
        """Initialize the Gaussian fitter."""
        self.last_params = None
        self.last_fit_curve = None
        self.last_r_squared = None
    
    @staticmethod
    def gaussian(x, amplitude, center, width, offset):
        """
        Gaussian function.
        
        Args:
            x: Independent variable (space coordinates)
            amplitude: Peak height above baseline
            center: Center position of peak
            width: Standard deviation (sigma)
            offset: Baseline offset
            
        Returns:
            Gaussian function values
        """
        return amplitude * np.exp(-(x - center)**2 / (2 * width**2)) + offset
    
    def auto_initial_guess(self, y_profile, y_coords):
        """
        Automatically generate initial guess for Gaussian parameters.
        
        Args:
            y_profile: 1D array of profile intensities
            y_coords: 1D array of coordinates (space)
            
        Returns:
            tuple: (amplitude, center, width, offset)
        """
        offset = np.min(y_profile)
        amplitude = np.max(y_profile) - offset
        center = y_coords[np.argmax(y_profile)]
        width = (y_coords[-1] - y_coords[0]) / 10
        
        return amplitude, center, width, offset
    
    def fit(self, y_profile, y_coords, initial_params=None, use_auto=True):
        """
        Fit Gaussian to profile data.
        
        Args:
            y_profile: 1D array of profile intensities
            y_coords: 1D array of coordinates (space)
            initial_params: Tuple of (amplitude, center, width, offset) or None
            use_auto: If True, use auto-generated initial guess
            
        Returns:
            dict: {
                'success': bool,
                'params': (amplitude, center, width, offset) or None,
                'fit_curve': fitted curve or None,
                'r_squared': goodness of fit or None,
                'fwhm': Full Width Half Maximum or None,
                'error_message': str or None
            }
        """
        try:
            # Determine initial guess
            if use_auto or initial_params is None:
                initial_guess = self.auto_initial_guess(y_profile, y_coords)
            else:
                initial_guess = initial_params
            
            # Perform fit
            params, covariance = curve_fit(
                self.gaussian, y_coords, y_profile,
                p0=initial_guess,
                maxfev=5000
            )
            
            amplitude, center, width, offset = params
            
            # Generate fitted curve
            fit_curve = self.gaussian(y_coords, *params)
            
            # Calculate R-squared
            residuals = y_profile - fit_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_profile - np.mean(y_profile))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate FWHM
            fwhm = 2.355 * abs(width)  # FWHM = 2.355 * sigma
            
            # Store results
            self.last_params = params
            self.last_fit_curve = fit_curve
            self.last_r_squared = r_squared
            
            return {
                'success': True,
                'params': params,
                'fit_curve': fit_curve,
                'r_squared': r_squared,
                'fwhm': fwhm,
                'error_message': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'params': None,
                'fit_curve': None,
                'r_squared': None,
                'fwhm': None,
                'error_message': str(e)
            }
    
    def get_fit_info_text(self, result, unit='pix'):
        """
        Generate formatted text for fit parameters.
        
        Args:
            result: Result dictionary from fit()
            unit: Unit string ('pix', 'µm', 'nm', etc.)
            
        Returns:
            str: Formatted parameter text
        """
        if not result['success']:
            return f"Fit Failed:\n{result['error_message']}"
        
        amplitude, center, width, offset = result['params']
        fwhm = result['fwhm']
        r_squared = result['r_squared']
        
        text = (
            f"Fit Parameters:\n\n"
            f"Amplitude: {amplitude:.4f}\n"
            f"Center: {center:.4f} {unit}\n"
            f"Width (σ): {abs(width):.4f} {unit}\n"
            f"FWHM: {fwhm:.4f} {unit}\n"
            f"Offset: {offset:.4f}\n"
            f"R²: {r_squared:.6f}"
        )
        
        return text

