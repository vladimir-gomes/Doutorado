import pandas as pd
import os

def load_and_consolidate_signatures(folder_path):
    """
    Carrega todos os arquivos de assinatura de pontos de uma pasta e os consolida em um único DataFrame.

    Args:
        folder_path (str): O caminho para a pasta contendo os arquivos CSV.

    Returns:
        pd.DataFrame: Um DataFrame consolidado com uma coluna 'classe' adicionada.
    """
    all_files = os.listdir(folder_path)
    target_files = [f for f in all_files if f.startswith('assinaturas_pontos_') and f.endswith('.csv')]

    if not target_files:
        raise FileNotFoundError(f"Nenhum arquivo 'assinaturas_pontos_*.csv' encontrado em {folder_path}")

    dfs = []
    print("Carregando arquivos...")
    for filename in target_files:
        try:
            class_name = filename.replace('assinaturas_pontos_', '').replace('.csv', '')
            file_path = os.path.join(folder_path, filename)
            temp_df = pd.read_csv(file_path)
            temp_df['classe'] = class_name
            dfs.append(temp_df)
            print(f" - Arquivo '{filename}' (classe: {class_name}) carregado.")
        except Exception as e:
            print(f"Erro ao carregar o arquivo {filename}: {e}")

    if not dfs:
        raise ValueError("Nenhum arquivo de dados pôde ser carregado com sucesso.")

    df_completo = pd.concat(dfs, ignore_index=True)

    print("\n--- Resumo do Conjunto de Dados Consolidado ---")
    print(f"Número total de amostras: {len(df_completo)}")
    print("Distribuição das amostras por classe:")
    print(df_completo['classe'].value_counts().to_markdown())

    return df_completo
