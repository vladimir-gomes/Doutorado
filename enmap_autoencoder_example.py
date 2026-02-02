"""
Exemplo de uso do autoencoder para recuperação de assinaturas espectrais EnMAP.

Este script demonstra como:
1. Carregar dados espectrais (simulados ou de CSV)
2. Treinar um autoencoder com restrições físicas
3. Extrair endmembers e abundâncias
4. Visualizar os resultados
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from spectral_analysis.autoencoder import SpectralAutoencoder, train_autoencoder_from_csv

# Configurações
OUTPUT_DIR = "test_data/autoencoder_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Opção 1: Treinar com dados do test_data ---
def train_with_test_data():
    """Treina o autoencoder com dados de teste existentes."""
    print("="*80)
    print("TREINAMENTO COM DADOS DE TESTE")
    print("="*80)
    
    # Arquivos de dados
    csv_files = [
        "test_data/assinaturas_pontos_caatinga.csv",
        "test_data/assinaturas_pontos_solo.csv"
    ]
    
    # Parâmetros
    n_endmembers = 5  # Número de componentes a extrair
    epochs = 50       # Épocas de treinamento
    batch_size = 32
    
    # Treinar autoencoder
    autoencoder, endmembers, abundances, history = train_autoencoder_from_csv(
        csv_files=csv_files,
        n_endmembers=n_endmembers,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return autoencoder, endmembers, abundances, history


# --- Opção 2: Treinar com dados sintéticos (EnMAP-like) ---
def train_with_synthetic_enmap_data():
    """Treina o autoencoder com dados sintéticos simulando EnMAP."""
    print("="*80)
    print("TREINAMENTO COM DADOS SINTÉTICOS (EnMAP-like)")
    print("="*80)
    
    # Simular dados EnMAP: 224 bandas, 1000 pixels
    n_samples = 1000
    n_bands = 224
    n_endmembers = 5
    
    print(f"\nGerando {n_samples} espectros sintéticos com {n_bands} bandas...")
    
    # Criar endmembers sintéticos com características espectrais típicas
    np.random.seed(42)
    synthetic_endmembers = np.random.rand(n_endmembers, n_bands) * 0.6 + 0.2
    
    # Adicionar características espectrais típicas
    band_centers = np.linspace(0, n_bands-1, n_bands)
    for i in range(n_endmembers):
        # Adicionar picos gaussianos em diferentes posições
        peak_pos = np.random.randint(30, n_bands-30)
        peak_width = np.random.randint(20, 50)
        gaussian = np.exp(-((band_centers - peak_pos)**2) / (2 * peak_width**2))
        synthetic_endmembers[i] += gaussian * 0.3
    
    # Gerar abundâncias aleatórias (soma = 1)
    synthetic_abundances = np.random.dirichlet(np.ones(n_endmembers), n_samples)
    
    # Gerar espectros mistos
    X_train = synthetic_abundances @ synthetic_endmembers
    
    # Adicionar ruído
    noise = np.random.normal(0, 0.02, X_train.shape)
    X_train += noise
    X_train = np.clip(X_train, 0, 1)  # Garantir valores válidos
    
    print(f"Dados sintéticos gerados: {X_train.shape}")
    
    # Criar e treinar autoencoder
    autoencoder = SpectralAutoencoder(n_bands=n_bands, n_endmembers=n_endmembers)
    autoencoder.compile(learning_rate=0.001, loss='sad')
    autoencoder.summary()
    
    print("\n--- Iniciando Treinamento ---")
    history = autoencoder.train(
        X_train,
        epochs=100,
        batch_size=32,
        verbose=1,
        normalize=True,
        validation_split=0.1
    )
    
    # Extrair resultados
    endmembers = autoencoder.extract_endmembers()
    abundances = autoencoder.extract_abundances(X_train)
    
    print("\n--- Resultados da Extração ---")
    print(f"Shape das abundâncias: {abundances.shape}")
    print(f"Shape dos endmembers: {endmembers.shape}")
    
    return autoencoder, endmembers, abundances, history, synthetic_endmembers, synthetic_abundances


def visualize_results(endmembers, abundances, history, output_dir=OUTPUT_DIR, 
                     true_endmembers=None, true_abundances=None):
    """
    Visualiza os resultados do autoencoder.
    
    Args:
        endmembers: Array com os endmembers extraídos (n_endmembers, n_bands)
        abundances: Array com as abundâncias estimadas (n_samples, n_endmembers)
        history: History object do treinamento
        output_dir: Diretório para salvar as figuras
        true_endmembers: Endmembers verdadeiros (para comparação, opcional)
        true_abundances: Abundâncias verdadeiras (para comparação, opcional)
    """
    print("\n--- Gerando Visualizações ---")
    
    n_endmembers, n_bands = endmembers.shape
    
    # 1. Plot dos endmembers extraídos
    plt.figure(figsize=(14, 6))
    for i in range(n_endmembers):
        plt.plot(endmembers[i], label=f'Endmember {i+1}', linewidth=2)
    
    plt.title('Assinaturas Espectrais dos Endmembers Extraídos', fontsize=14, fontweight='bold')
    plt.xlabel('Índice da Banda', fontsize=12)
    plt.ylabel('Reflectância', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'endmembers_extracted.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: {os.path.join(output_dir, 'endmembers_extracted.png')}")
    plt.close()
    
    # 2. Histograma das abundâncias
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(n_endmembers, len(axes))):
        axes[i].hist(abundances[:, i], bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
        axes[i].set_title(f'Abundância - Endmember {i+1}', fontweight='bold')
        axes[i].set_xlabel('Fração')
        axes[i].set_ylabel('Frequência')
        axes[i].grid(True, alpha=0.3)
    
    # Remover subplots vazios
    for i in range(n_endmembers, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Distribuição das Abundâncias', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abundances_histogram.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: {os.path.join(output_dir, 'abundances_histogram.png')}")
    plt.close()
    
    # 3. Curva de perda do treinamento
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda de Treinamento', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Perda de Validação', linewidth=2)
    
    plt.title('Curva de Aprendizado do Autoencoder', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Perda (SAD)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: {os.path.join(output_dir, 'training_loss.png')}")
    plt.close()
    
    # 4. Comparação com endmembers verdadeiros (se disponível)
    if true_endmembers is not None:
        plt.figure(figsize=(14, 10))
        
        for i in range(n_endmembers):
            plt.subplot(n_endmembers, 1, i+1)
            plt.plot(true_endmembers[i], label='Verdadeiro', linewidth=2, alpha=0.7)
            plt.plot(endmembers[i], label='Extraído', linewidth=2, linestyle='--', alpha=0.7)
            plt.title(f'Endmember {i+1}', fontweight='bold')
            plt.ylabel('Reflectância')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if i == n_endmembers - 1:
                plt.xlabel('Índice da Banda')
        
        plt.suptitle('Comparação: Endmembers Verdadeiros vs Extraídos', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'endmembers_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Salvo: {os.path.join(output_dir, 'endmembers_comparison.png')}")
        plt.close()
    
    # 5. Scatter plot das abundâncias (primeiros 2 componentes)
    if n_endmembers >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(abundances[:, 0], abundances[:, 1], 
                            c=np.arange(len(abundances)), cmap='viridis', 
                            alpha=0.6, edgecolor='k', linewidth=0.5)
        plt.colorbar(scatter, label='Índice da Amostra')
        plt.xlabel(f'Abundância - Endmember 1', fontsize=12)
        plt.ylabel(f'Abundância - Endmember 2', fontsize=12)
        plt.title('Espaço de Abundâncias (Primeiros 2 Componentes)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'abundance_space.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Salvo: {os.path.join(output_dir, 'abundance_space.png')}")
        plt.close()
    
    print("\n✓ Todas as visualizações foram salvas com sucesso!")


def main():
    """Função principal do exemplo."""
    print("\n" + "="*80)
    print("EXEMPLO: AUTOENCODER PARA RECUPERAÇÃO DE ASSINATURAS ESPECTRAIS ENMAP")
    print("="*80 + "\n")
    
    # Escolher qual método usar
    use_test_data = os.path.exists("test_data/assinaturas_pontos_caatinga.csv")
    
    if use_test_data:
        print("Usando dados de teste existentes...")
        autoencoder, endmembers, abundances, history = train_with_test_data()
        visualize_results(endmembers, abundances, history)
    else:
        print("Dados de teste não encontrados. Usando dados sintéticos...")
        result = train_with_synthetic_enmap_data()
        autoencoder, endmembers, abundances, history = result[:4]
        synthetic_endmembers, synthetic_abundances = result[4:]
        visualize_results(endmembers, abundances, history, 
                        true_endmembers=synthetic_endmembers)
    
    # Estatísticas finais
    print("\n" + "="*80)
    print("ESTATÍSTICAS FINAIS")
    print("="*80)
    print(f"\nEndmembers extraídos: {endmembers.shape[0]}")
    print(f"Bandas espectrais: {endmembers.shape[1]}")
    print(f"Amostras processadas: {abundances.shape[0]}")
    print(f"\nRestrições verificadas:")
    print(f"  • Abundâncias não-negativas: {np.all(abundances >= 0)}")
    print(f"  • Soma das abundâncias = 1: {np.allclose(abundances.sum(axis=1), 1.0)}")
    print(f"  • Endmembers não-negativos: {np.all(endmembers >= 0)}")
    print(f"\nPerda final de treinamento: {history.history['loss'][-1]:.6f}")
    
    # Salvar modelo
    model_path = os.path.join(OUTPUT_DIR, "spectral_autoencoder.keras")
    autoencoder.save(model_path)
    print(f"\nModelo salvo em: {model_path}")
    
    print("\n" + "="*80)
    print("EXEMPLO CONCLUÍDO COM SUCESSO!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
