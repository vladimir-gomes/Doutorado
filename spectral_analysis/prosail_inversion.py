import numpy as np
import pandas as pd
from prosail import run_prosail
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

def generate_prosail_lut(n_simulations, soil_spectrum):
    """
    Gera uma Look-Up Table (LUT) de espectros simulados usando o PROSAIL.
    """
    print(f"Gerando uma LUT com {n_simulations} simulações...")
    lut_params = []
    lut_spectra = []

    prosail_wavelengths = np.linspace(400, 2500, 2101)

    for _ in range(n_simulations):
        params = {
            'n': np.random.uniform(1.0, 2.5),
            'cab': np.random.uniform(10., 80.),
            'car': np.random.uniform(5., 25.),
            'cbrown': np.random.uniform(0.0, 1.0),
            'cw': np.random.uniform(0.005, 0.06),
            'cm': np.random.uniform(0.002, 0.02),
            'lai': np.random.uniform(0.1, 7.0),
            'lidfa': np.random.uniform(30., 80.),
            'hspot': np.random.uniform(0.01, 0.5),
            'tts': 30.,
            'tto': 0.,
            'psi': 0.,
            'soil_spectrum1': soil_spectrum,
            'rsoil': 1.0,
            'psoil': 0.0
        }
        simulated_reflectance = run_prosail(**params)
        lut_params.append(params)
        lut_spectra.append(simulated_reflectance)

    lut_params_df = pd.DataFrame(lut_params)
    lut_spectra = np.array(lut_spectra)
    print("Geração da LUT concluída!")
    return lut_params_df, lut_spectra

def invert_spectrum(target_spectrum, lut_spectra, lut_params_df, target_wavelengths):
    """
    Encontra os parâmetros da LUT que melhor correspondem a um espectro-alvo.
    """
    prosail_wavelengths_lut = np.linspace(400, 2500, lut_spectra.shape[1])
    target_interp_func = interp1d(target_wavelengths, target_spectrum, kind='linear', fill_value="extrapolate")
    interpolated_target_spectrum = target_interp_func(prosail_wavelengths_lut)

    rmse_list = np.sqrt(np.mean((lut_spectra - interpolated_target_spectrum)**2, axis=1))
    best_index = np.argmin(rmse_list)

    best_params = lut_params_df.iloc[best_index]
    best_simulated_spectrum = lut_spectra[best_index]
    min_rmse = rmse_list[best_index]

    return best_params, best_simulated_spectrum, min_rmse
