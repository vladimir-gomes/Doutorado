import os
import pandas as pd
from spectral_analysis import data_loader, analysis, prosail_inversion

# --- Configurações ---
# ATENÇÃO: Altere este caminho para a pasta que contém seus arquivos .csv
DATA_FOLDER = "/content/drive/MyDrive/1/"
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "resultados_pipeline")

def main():
    """
    Script principal para executar o pipeline de análise espectral.
    """
    print("Iniciando o pipeline de análise espectral...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Resultados serão salvos em: {OUTPUT_FOLDER}")

    # --- 1. Carregamento e Consolidação dos Dados ---
    try:
        df_completo = data_loader.load_and_consolidate_signatures(DATA_FOLDER)
        X = df_completo.drop(columns=['classe'])
        y = df_completo['classe']
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro crítico ao carregar dados: {e}")
        return

    # --- 2. Análise de Separabilidade Inter-Classe ---
    print("\n--- Executando Análise de Separabilidade Inter-Classe ---")
    analysis.plot_mean_spectra(X, y, OUTPUT_FOLDER)
    analysis.plot_global_pca(X, y, OUTPUT_FOLDER)
    analysis.calculate_separability_matrix(X, y, OUTPUT_FOLDER)

    # --- 3. Análise de Importância das Bandas ---
    print("\n--- Executando Análise de Importância das Bandas ---")
    feature_importance = analysis.get_feature_importance(X, y, OUTPUT_FOLDER)

    # --- 4. Análise de Variabilidade Intra-Classe (Loop por classe) ---
    print("\n--- Executando Análise de Variabilidade Intra-Classe ---")
    for class_name in y.unique():
        print(f"\nAnalisando a classe: {class_name}")
        class_data = X[y == class_name]
        if class_data.empty:
            continue

        analysis.plot_intra_class_variability(class_data, class_name, OUTPUT_FOLDER)
        analysis.plot_dendrogram(class_data, class_name, OUTPUT_FOLDER)
        analysis.plot_pca(class_data, class_name, OUTPUT_FOLDER)

    # --- 5. Exemplo de Inversão PROSAIL (para uma classe de vegetação) ---
    # Este é um exemplo. Você precisará adaptar os nomes das classes e a lógica
    # para corresponder aos seus dados.
    veg_class_name = 'caatinga' # ATENÇÃO: Mude para o nome da sua classe de vegetação
    soil_class_name = 'solo'   # ATENÇÃO: Mude para o nome da sua classe de solo

    if veg_class_name in y.unique() and soil_class_name in y.unique():
        print(f"\n--- Executando Exemplo de Inversão PROSAIL para a classe '{veg_class_name}' ---")

        # Prepara os dados de solo e vegetação
        solo_spectrum_mean = X[y == soil_class_name].mean().values
        veg_spectrum_sample = X[y == veg_class_name].iloc[0].values

        # Define os comprimentos de onda (necessário para interpolação)
        # Assumindo que as colunas são nomeadas como 'reflectance_WL'
        try:
            wavelengths = [float(col.split('_')[-1]) for col in X.columns]
        except (ValueError, IndexError):
            # Se a extração falhar, cria um placeholder
            num_bands = len(X.columns)
            # Estimativa de 380 a 2500 nm, comum para sensores hiperespectrais
            wavelengths = np.linspace(380, 2500, num_bands)
            print("AVISO: Não foi possível extrair comprimentos de onda dos nomes das colunas. Usando uma faixa estimada.")

        # Gera a LUT
        lut_params, lut_spectra = prosail_inversion.generate_prosail_lut(n_simulations=1000, soil_spectrum=solo_spectrum_mean)

        # Inverte o espectro
        best_params, _, min_rmse = prosail_inversion.invert_spectrum(
            target_spectrum=veg_spectrum_sample,
            lut_spectra=lut_spectra,
            lut_params_df=lut_params,
            target_wavelengths=wavelengths
        )

        print("\nResultados da Inversão PROSAIL (Amostra):")
        print(f"  - RMSE Mínimo: {min_rmse:.4f}")
        print(f"  - LAI Recuperado: {best_params['lai']:.2f}")
        print(f"  - Clorofila (cab) Recuperada: {best_params['cab']:.2f}")
        print(f"  - Conteúdo de Água (cw) Recuperado: {best_params['cw']:.4f}")

    print("\nPipeline de análise concluído com sucesso!")

if __name__ == "__main__":
    main()
