import numpy as np
import pandas as pd
from prosail import run_prosail
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

# ============================================================================
# PROSAIL PARAMETER DESCRIPTIONS
# ============================================================================
# These parameters are used in the PROSAIL radiative transfer model for
# vegetation canopy simulation and inversion.
#
# Leaf Biochemical Parameters:
#   'n'       : Leaf structure parameter (refraction index) [1.0-2.5]
#   'cab'     : Chlorophyll a+b content (μg/cm²) [10-80]
#   'car'     : Carotenoid content (μg/cm²) [5-25]
#   'cbrown'  : Brown pigment content (arbitrary units) [0.0-1.0]
#   'cw'      : Equivalent water thickness (cm) [0.005-0.06]
#   'cm'      : Dry matter content (g/cm²) [0.002-0.02]
#
# Canopy Structure Parameters:
#   'lai'     : Leaf Area Index (m²/m²) [0.1-7.0]
#   'lidfa'   : Leaf angle distribution function parameter (degrees) [30-80]
#   'hspot'   : Hot spot parameter (ratio) [0.01-0.5]
#
# Observation Geometry Parameters:
#   'tts'     : Solar zenith angle (degrees)
#   'tto'     : Observer zenith angle (degrees)
#   'psi'     : Relative azimuth angle (degrees)
#
# Soil Parameters:
#   'rsoil'   : Soil brightness parameter
#   'psoil'   : Soil moisture parameter
# ============================================================================

def generate_prosail_lut(n_simulations, soil_spectrum, 
                         tts=30.0, tto=0.0, psi=0.0):
    """
    Gera uma Look-Up Table (LUT) de espectros simulados usando o PROSAIL.
    
    Args:
        n_simulations (int): Número de simulações a serem geradas
        soil_spectrum (array): Espectro de reflectância do solo
        tts (float, optional): Solar zenith angle (degrees). Default: 30.0
        tto (float, optional): Observer zenith angle (degrees). Default: 0.0
        psi (float, optional): Relative azimuth angle (degrees). Default: 0.0
        
    Returns:
        tuple: (DataFrame com parâmetros, array com espectros simulados)
        
    Note:
        Os parâmetros biofísicos são amostrados aleatoriamente dentro de faixas
        físicas válidas para vegetação. Os ângulos de observação podem ser
        especificados para simular diferentes condições de iluminação e visada.
        Veja a documentação de parâmetros acima para detalhes sobre cada variável.
    """
    print(f"Gerando uma LUT com {n_simulations} simulações...")
    lut_params = []
    lut_spectra = []

    prosail_wavelengths = np.linspace(400, 2500, 2101)

    for _ in range(n_simulations):
        # Dictionary mapping PROSAIL parameter names to random values
        # within physically valid ranges for vegetation
        params = {
            'n': np.random.uniform(1.0, 2.5),          # Leaf structure parameter
            'cab': np.random.uniform(10., 80.),        # Chlorophyll a+b (μg/cm²)
            'car': np.random.uniform(5., 25.),         # Carotenoids (μg/cm²)
            'cbrown': np.random.uniform(0.0, 1.0),     # Brown pigments
            'cw': np.random.uniform(0.005, 0.06),      # Water thickness (cm)
            'cm': np.random.uniform(0.002, 0.02),      # Dry matter (g/cm²)
            'lai': np.random.uniform(0.1, 7.0),        # Leaf Area Index
            'lidfa': np.random.uniform(30., 80.),      # Leaf angle distribution (°)
            'hspot': np.random.uniform(0.01, 0.5),     # Hot spot parameter
            'tts': tts,                                 # Solar zenith angle (°)
            'tto': tto,                                 # Observer zenith angle (°)
            'psi': psi,                                 # Relative azimuth angle (°)
            'soil_spectrum1': soil_spectrum,            # Soil reflectance spectrum
            'rsoil': 1.0,                               # Soil brightness
            'psoil': 0.0                                # Soil moisture
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
    
    Este método realiza a inversão do modelo PROSAIL comparando o espectro
    observado com todos os espectros simulados na LUT, retornando o conjunto
    de parâmetros que minimiza o erro RMSE.
    
    Args:
        target_spectrum (array): Espectro de reflectância observado
        lut_spectra (array): Array de espectros simulados da LUT
        lut_params_df (DataFrame): DataFrame com parâmetros correspondentes
        target_wavelengths (array): Comprimentos de onda do espectro observado
        
    Returns:
        tuple: (best_params, best_simulated_spectrum, min_rmse)
            - best_params: Série do pandas com os melhores parâmetros
            - best_simulated_spectrum: Espectro simulado correspondente
            - min_rmse: Erro RMSE mínimo alcançado
            
    Note:
        O espectro-alvo é interpolado para a grade espectral do PROSAIL
        (400-2500 nm, 2101 bandas) antes da comparação.
    """
    # Interpolate target spectrum to PROSAIL wavelength grid
    prosail_wavelengths_lut = np.linspace(400, 2500, lut_spectra.shape[1])
    target_interp_func = interp1d(target_wavelengths, target_spectrum, kind='linear', fill_value="extrapolate")
    interpolated_target_spectrum = target_interp_func(prosail_wavelengths_lut)

    # Calculate RMSE for each LUT spectrum
    rmse_list = np.sqrt(np.mean((lut_spectra - interpolated_target_spectrum)**2, axis=1))
    
    # Find best match (minimum RMSE)
    best_index = np.argmin(rmse_list)

    best_params = lut_params_df.iloc[best_index]
    best_simulated_spectrum = lut_spectra[best_index]
    min_rmse = rmse_list[best_index]

    return best_params, best_simulated_spectrum, min_rmse


def get_parameter_descriptions():
    """
    Retorna um dicionário com descrições legíveis dos parâmetros PROSAIL.
    
    Returns:
        dict: Mapeamento de nomes de parâmetros para suas descrições completas
        
    Example:
        >>> descriptions = get_parameter_descriptions()
        >>> print(descriptions['lai'])
        'Leaf Area Index (m²/m²)'
    """
    return {
        'n': 'Leaf structure parameter (refraction index)',
        'cab': 'Chlorophyll a+b content (μg/cm²)',
        'car': 'Carotenoid content (μg/cm²)',
        'cbrown': 'Brown pigment content (arbitrary units)',
        'cw': 'Equivalent water thickness (cm)',
        'cm': 'Dry matter content (g/cm²)',
        'lai': 'Leaf Area Index (m²/m²)',
        'lidfa': 'Leaf angle distribution function parameter (degrees)',
        'hspot': 'Hot spot parameter (ratio)',
        'tts': 'Solar zenith angle (degrees)',
        'tto': 'Observer zenith angle (degrees)',
        'psi': 'Relative azimuth angle (degrees)',
        'rsoil': 'Soil brightness parameter',
        'psoil': 'Soil moisture parameter',
        'soil_spectrum1': 'Soil reflectance spectrum'
    }


def print_inversion_results(params, rmse=None):
    """
    Imprime os resultados da inversão PROSAIL de forma legível.
    
    Args:
        params (pd.Series or dict): Parâmetros recuperados da inversão
        rmse (float, optional): Erro RMSE da inversão
        
    Example:
        >>> best_params, _, min_rmse = invert_spectrum(...)
        >>> print_inversion_results(best_params, min_rmse)
    """
    descriptions = get_parameter_descriptions()
    
    print("\n" + "="*70)
    print("RESULTADOS DA INVERSÃO PROSAIL")
    print("="*70)
    
    if rmse is not None:
        print(f"\nErro RMSE: {rmse:.6f}\n")
    
    print("Parâmetros recuperados:")
    print("-"*70)
    
    for param_name, param_value in params.items():
        # Skip spectrum arrays for cleaner output
        if param_name == 'soil_spectrum1':
            continue
            
        desc = descriptions.get(param_name, param_name)
        
        # Format output based on parameter type
        if isinstance(param_value, (int, float)):
            print(f"  {param_name:12s} = {param_value:8.4f}  ({desc})")
        else:
            print(f"  {param_name:12s} = {param_value}  ({desc})")
    
    print("="*70 + "\n")

