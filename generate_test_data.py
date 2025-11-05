import pandas as pd
import numpy as np
import os

def generate_synthetic_spectrum(base_spectrum, noise_level=0.05, num_samples=10):
    """Gera espectros sintéticos adicionando ruído a um espectro base."""
    num_bands = len(base_spectrum)
    synthetic_data = np.zeros((num_samples, num_bands))
    for i in range(num_samples):
        noise = np.random.normal(0, noise_level, num_bands)
        synthetic_data[i, :] = base_spectrum + noise
    return np.clip(synthetic_data, 0, 1) # Garante que a reflectância fique entre 0 e 1

def main():
    """Gera arquivos de dados de teste sintéticos."""
    output_dir = 'test_data'
    os.makedirs(output_dir, exist_ok=True)

    num_bands = 285
    wavelengths = np.linspace(380, 2500, num_bands)

    # Espectro base para 'caatinga' (simulando vegetação)
    caatinga_base = 0.1 + 0.4 * np.exp(-((wavelengths - 800) / 300)**2)

    # Espectro base para 'solo' (simulando solo)
    solo_base = 0.15 + 0.1 * (wavelengths / 2500)

    # Gerar e salvar dados da caatinga
    caatinga_data = generate_synthetic_spectrum(caatinga_base, num_samples=20)
    caatinga_df = pd.DataFrame(caatinga_data, columns=[f'reflectance_{i}' for i in range(num_bands)])
    caatinga_df.to_csv(os.path.join(output_dir, 'assinaturas_pontos_caatinga.csv'), index=False)
    print("Dados sintéticos para 'caatinga' gerados.")

    # Gerar e salvar dados do solo
    solo_data = generate_synthetic_spectrum(solo_base, num_samples=15)
    solo_df = pd.DataFrame(solo_data, columns=[f'reflectance_{i}' for i in range(num_bands)])
    solo_df.to_csv(os.path.join(output_dir, 'assinaturas_pontos_solo.csv'), index=False)
    print("Dados sintéticos para 'solo' gerados.")

if __name__ == '__main__':
    main()
