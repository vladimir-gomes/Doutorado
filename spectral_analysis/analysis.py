import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from scipy import stats
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier

def plot_intra_class_variability(reflectance_data, class_name, wavelengths, output_folder):
    """
    Plota a variabilidade espectral média e o desvio padrão para uma única classe.
    """
    mean_signature = reflectance_data.mean(axis=0)
    std_signature = reflectance_data.std(axis=0)

    plt.figure(figsize=(14, 8))
    plt.plot(wavelengths, mean_signature, color='blue', linewidth=2, label='Média')
    plt.fill_between(wavelengths, mean_signature - std_signature, mean_signature + std_signature, color='blue', alpha=0.2, label='Desvio Padrão')
    plt.title(f'Variabilidade Espectral Intra-Classe: {class_name}', fontsize=16)
    plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
    plt.ylabel('Reflectância', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_folder, f"variabilidade_{class_name}.png"))
    plt.show()

def calculate_sam(s1, s2):
    """
    Calcula o Spectral Angle Mapper (SAM) entre duas assinaturas.
    """
    dot_product = np.dot(s1, s2)
    norm_s1 = np.linalg.norm(s1)
    norm_s2 = np.linalg.norm(s2)
    if norm_s1 == 0 or norm_s2 == 0: return 0.0
    cosine_angle = np.clip(dot_product / (norm_s1 * norm_s2), -1.0, 1.0)
    return np.arccos(cosine_angle)

def plot_dendrogram(reflectance_data, class_name, output_folder):
    """
    Calcula e plota um dendrograma de agrupamento hierárquico.
    """
    sam_distances = pdist(reflectance_data.values, metric=calculate_sam)
    if len(sam_distances) > 0:
        linked = linkage(sam_distances, method='average')
        plt.figure(figsize=(16, 9))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title(f"Dendrograma de Agrupamento Hierárquico - {class_name}", fontsize=16)
        plt.ylabel("Distância SAM (radianos)", fontsize=12)
        plt.savefig(os.path.join(output_folder, f"dendrograma_{class_name}.png"))
        plt.show()

def plot_pca(reflectance_data, class_name, output_folder):
    """
    Realiza e plota a Análise de Componentes Principais (PCA).
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(reflectance_data)
    pca_df = pd.DataFrame(data=pca_result, columns=["Componente Principal 1", "Componente Principal 2"])

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="Componente Principal 1", y="Componente Principal 2", data=pca_df, alpha=0.7)
    plt.title(f'PCA das Assinaturas Espectrais - {class_name}', fontsize=16)
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_folder, f"pca_{class_name}.png"))
    plt.show()

def analyze_intra_class_normality(data, class_name, output_folder):
    """
    Plota a distribuição dos dados, realiza o teste de normalidade e gera um Q-Q plot.
    """
    if len(data) < 3:
        print(f"AVISO: Dados insuficientes ({len(data)} pontos) para análise de distribuição da classe {class_name}.")
        return

    shapiro_stat, p_value = stats.shapiro(data)
    alpha = 0.05
    is_normal = "Sim (p > 0.05)" if p_value > alpha else "Não (p <= 0.05)"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Análise de Distribuição das Distâncias SAM - Classe: {class_name}', fontsize=18)

    sns.histplot(data, kde=True, ax=axes[0], stat="density", color="skyblue")
    axes[0].set_title('Histograma + Curva de Densidade (KDE)', fontsize=14)
    axes[0].set_xlabel('Distância SAM (radianos)', fontsize=12)
    axes[0].set_ylabel('Densidade', fontsize=12)

    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot vs. Distribuição Normal', fontsize=14)

    plt.figtext(0.5, 0.92, f'Teste de Normalidade (Shapiro-Wilk): W={shapiro_stat:.4f}, p-valor={p_value:.4f} | Normal? {is_normal}',
                ha='center', va='bottom', fontsize=12, color='darkred')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_folder, f"distribuicao_normalidade_{class_name}.png"))
    plt.show()

def plot_mean_spectra(X, y, wavelengths, output_folder):
    """
    Plota as assinaturas espectrais médias por classe de forma interativa.
    """
    mean_spectra = X.groupby(y).mean()

    fig = go.Figure()
    for class_name, row in mean_spectra.iterrows():
        fig.add_trace(go.Scatter(x=wavelengths, y=row.values, name=class_name, mode='lines'))

    fig.update_layout(
        title='Assinaturas Espectrais Médias por Classe',
        xaxis_title='Comprimento de Onda (nm)',
        yaxis_title='Reflectância',
        legend_title='Classes',
        hovermode='x unified'
    )
    fig.write_html(os.path.join(output_folder, "perfis_medios_interclasse.html"))
    fig.show()

def plot_global_pca(X, y, output_folder):
    """
    Executa e plota uma PCA global com todas as classes.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['classe'] = y.reset_index(drop=True)

    fig_pca = px.scatter(
        pca_df, x='PC1', y='PC2', color='classe',
        title='PCA Global das Assinaturas Espectrais',
        labels={
            "PC1": f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
            "PC2": f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'
        }
    )
    fig_pca.write_html(os.path.join(output_folder, "pca_global_interclasse.html"))
    fig_pca.show()

def calculate_separability_matrix(X, y, output_folder, n_components=15):
    """
    Calcula a matriz de distância Euclidiana entre centroides de classe no espaço PCA.
    """
    from scipy.spatial.distance import pdist, squareform

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples_per_class = y.value_counts()
    if n_samples_per_class.min() <= 1:
        print("AVISO: Pelo menos uma classe tem apenas 1 amostra. A análise de separabilidade não pode ser executada.")
        return

    n_components = min(n_components, n_samples_per_class.min() - 1)
    print(f"\nUsando {n_components} Componentes Principais para a análise de separabilidade.")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components)])
    X_pca_df['classe'] = y

    class_centroids = X_pca_df.groupby('classe').mean()
    euclidean_distances = pdist(class_centroids, metric='euclidean')
    distance_matrix = pd.DataFrame(squareform(euclidean_distances), index=class_centroids.index, columns=class_centroids.index)

    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
    plt.title('Matriz de Separabilidade - Distância Euclidiana entre Centroides (no Espaço PCA)')
    plt.savefig(os.path.join(output_folder, "matriz_euclidiana_pca.png"))
    plt.show()
    return distance_matrix

def get_feature_importance(X, y, wavelengths, output_folder):
    """
    Treina um RandomForest para extrair a importância das bandas espectrais.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)

    feature_importance = pd.DataFrame({
        'wavelength': wavelengths,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("As 15 bandas mais importantes:")
    print(feature_importance.head(15).to_markdown(index=False))

    plt.figure(figsize=(18, 7))
    sns.barplot(x='wavelength', y='importance', data=feature_importance, color='dodgerblue')
    plt.title('Importância de Cada Banda Espectral para a Classificação (Random Forest)', fontsize=16)
    plt.ylabel('Importância (Gini Importance)', fontsize=12)
    plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)

    # Ajusta os ticks do eixo x para evitar sobreposição
    tick_positions = np.arange(0, len(wavelengths), step=max(1, len(wavelengths) // 20))
    tick_labels = [f"{wavelengths[i]:.1f}" for i in tick_positions]
    plt.gca().set_xticks(tick_positions)
    plt.gca().set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "feature_importance.png"))
    plt.show()
    return feature_importance
