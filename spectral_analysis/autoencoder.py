"""
Autoencoder module for spectral signature recovery and endmember extraction.

This module implements a physics-constrained autoencoder based on:
- Spectral Angle Divergence (SAD) loss function
- Dense encoder with LeakyReLU activation
- Linear decoder with non-negativity constraints
- Automatic abundance and endmember extraction

The architecture follows the principles from spectral unmixing literature,
where the encoder learns abundance maps (softmax-constrained) and the decoder
learns endmembers (non-negative constrained).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import NonNeg
import warnings


def sad_loss(y_true, y_pred):
    """
    Calcula a perda Spectral Angle Divergence (SAD).
    
    O SAD mede o ângulo espectral entre os espectros verdadeiros e preditos,
    sendo independente da magnitude e focado na forma espectral.
    
    Args:
        y_true: Tensor com os espectros verdadeiros (ground truth)
        y_pred: Tensor com os espectros preditos pelo modelo
        
    Returns:
        Tensor escalar com a média dos ângulos espectrais
    """
    # Normaliza os vetores (calcula a norma L2)
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Calcula o produto escalar (dot product)
    dot_product = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Evita instabilidade numérica com clipping
    dot_product_clipped = tf.clip_by_value(dot_product, -1.0 + 1e-5, 1.0 - 1e-5)
    
    # Calcula o ângulo (arccosine)
    angle = tf.math.acos(dot_product_clipped)
    
    # Retorna a média dos ângulos
    return tf.reduce_mean(angle)


def build_encoder(input_shape_b, latent_dim_n, hidden_units=128, leaky_alpha=0.02):
    """
    Constrói o modelo do Encoder (E).
    
    O encoder aprende a mapear espectros de entrada para abundâncias.
    Usa uma camada oculta com LeakyReLU e uma camada de saída com softmax
    para impor restrições de soma unitária (ASC) e não-negatividade (ANC).
    
    Args:
        input_shape_b: Número de bandas espectrais de entrada
        latent_dim_n: Número de endmembers/abundâncias (dimensão latente)
        hidden_units: Número de neurônios na camada oculta (padrão: 128)
        leaky_alpha: Parâmetro alpha do LeakyReLU (padrão: 0.02)
        
    Returns:
        Modelo Keras do encoder
    """
    input_layer = Input(shape=(input_shape_b,), name="encoder_input")
    
    # Camada oculta com LeakyReLU
    hidden = Dense(hidden_units, activation=LeakyReLU(alpha=leaky_alpha), 
                   name="encoder_hidden_128")(input_layer)
    
    # Camada latente (Abundâncias 'z')
    # Softmax impõe: z >= 0 e sum(z) = 1
    z_abundances = Dense(latent_dim_n, activation='softmax', 
                        name="z_abundances_softmax")(hidden)
    
    return Model(input_layer, z_abundances, name="Encoder_E")


def build_decoder(latent_dim_n, output_shape_b):
    """
    Constrói o modelo do Decoder (D).
    
    O decoder é um modelo linear simples onde os pesos representam os endmembers.
    Usa restrição de não-negatividade nos pesos e não tem bias.
    
    Args:
        latent_dim_n: Número de endmembers/abundâncias (dimensão latente)
        output_shape_b: Número de bandas espectrais de saída
        
    Returns:
        Modelo Keras do decoder
    """
    latent_input = Input(shape=(latent_dim_n,), name="decoder_input")
    
    # Camada do Decoder: pesos (kernel) W representam os Endmembers M
    # - use_bias=False: sem bias (modelo linear puro)
    # - kernel_constraint=NonNeg(): impõe não-negatividade dos endmembers
    reconstruction = Dense(
        output_shape_b,
        activation='linear',
        use_bias=False,
        kernel_constraint=NonNeg(),
        name="Decoder_W_Endmembers"
    )(latent_input)
    
    return Model(latent_input, reconstruction, name="Decoder_D")


class SpectralAutoencoder:
    """
    Autoencoder com restrições físicas para extração de endmembers e abundâncias.
    
    Esta classe encapsula o treinamento e uso de um autoencoder para:
    - Extração de endmembers (componentes espectrais puros)
    - Estimação de abundâncias (frações de cada endmember)
    - Reconstrução de assinaturas espectrais
    
    Atributos:
        n_bands: Número de bandas espectrais
        n_endmembers: Número de endmembers a extrair
        encoder: Modelo do encoder
        decoder: Modelo do decoder
        autoencoder: Modelo completo (encoder + decoder)
        trained: Flag indicando se o modelo foi treinado
    """
    
    def __init__(self, n_bands, n_endmembers, hidden_units=128, leaky_alpha=0.02):
        """
        Inicializa o SpectralAutoencoder.
        
        Args:
            n_bands: Número de bandas espectrais
            n_endmembers: Número de endmembers a extrair
            hidden_units: Número de neurônios na camada oculta do encoder
            leaky_alpha: Parâmetro alpha do LeakyReLU
        """
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.hidden_units = hidden_units
        self.leaky_alpha = leaky_alpha
        self.trained = False
        
        # Construir os modelos
        self.encoder = build_encoder(n_bands, n_endmembers, hidden_units, leaky_alpha)
        self.decoder = build_decoder(n_endmembers, n_bands)
        
        # Construir o autoencoder completo
        ae_input = Input(shape=(n_bands,), name="AE_Input_Spectra_X")
        z = self.encoder(ae_input)
        x_hat = self.decoder(z)
        self.autoencoder = Model(ae_input, x_hat, name="PhysicsConstrained_AE")
        
    def compile(self, learning_rate=0.001, loss='sad'):
        """
        Compila o autoencoder.
        
        Args:
            learning_rate: Taxa de aprendizado para o otimizador Adam
            loss: Função de perda ('sad' para Spectral Angle Divergence, 
                  ou qualquer função de perda do Keras)
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if loss == 'sad':
            loss_fn = sad_loss
        else:
            loss_fn = loss
            
        self.autoencoder.compile(optimizer=optimizer, loss=loss_fn)
        
    def train(self, X_train, epochs=100, batch_size=32, verbose=1, 
              normalize=True, validation_split=0.0, callbacks=None):
        """
        Treina o autoencoder.
        
        Args:
            X_train: Array numpy com os dados de treinamento (n_samples, n_bands)
            epochs: Número de épocas de treinamento
            batch_size: Tamanho do batch
            verbose: Nível de verbosidade (0=silencioso, 1=barra de progresso, 2=uma linha por época)
            normalize: Se True, normaliza os dados antes do treinamento
            validation_split: Fração dos dados para validação (0.0-1.0)
            callbacks: Lista de callbacks do Keras
            
        Returns:
            History object do Keras com métricas de treinamento
        """
        # Converter para float32 se necessário
        X_train = X_train.astype(np.float32)
        
        # Normalização opcional
        if normalize:
            max_vals = np.max(X_train, axis=1, keepdims=True)
            X_train = X_train / np.maximum(max_vals, 1e-7)
        
        # Treinar (entrada e saída são os mesmos dados)
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks
        )
        
        self.trained = True
        return history
        
    def extract_abundances(self, X):
        """
        Extrai as abundâncias para os dados fornecidos.
        
        Args:
            X: Array numpy com os dados (n_samples, n_bands)
            
        Returns:
            Array numpy com as abundâncias (n_samples, n_endmembers)
        """
        if not self.trained:
            warnings.warn("Modelo não treinado. Resultados podem não ser significativos.")
        
        return self.encoder.predict(X.astype(np.float32))
        
    def extract_endmembers(self):
        """
        Extrai os endmembers do modelo treinado.
        
        Os endmembers são os pesos da camada do decoder.
        
        Returns:
            Array numpy com os endmembers (n_endmembers, n_bands)
        """
        if not self.trained:
            warnings.warn("Modelo não treinado. Resultados podem não ser significativos.")
        
        # Os pesos da camada Dense são (input_dim, output_dim)
        # No decoder, input é n_endmembers e output é n_bands
        # Então a shape é (n_endmembers, n_bands) - já está na forma correta
        weights = self.decoder.get_weights()[0]
        endmembers = weights  # Shape: (n_endmembers, n_bands)
        
        return endmembers
        
    def reconstruct(self, X):
        """
        Reconstrói os espectros a partir das abundâncias aprendidas.
        
        Args:
            X: Array numpy com os dados originais (n_samples, n_bands)
            
        Returns:
            Array numpy com os espectros reconstruídos (n_samples, n_bands)
        """
        if not self.trained:
            warnings.warn("Modelo não treinado. Resultados podem não ser significativos.")
        
        return self.autoencoder.predict(X.astype(np.float32))
        
    def summary(self):
        """Imprime um resumo da arquitetura do modelo."""
        print("\n--- Arquitetura do Encoder (E) ---")
        self.encoder.summary()
        print("\n--- Arquitetura do Decoder (D) ---")
        self.decoder.summary()
        print("\n--- Arquitetura do Autoencoder (AE) Completo ---")
        self.autoencoder.summary()
        
    def save(self, filepath):
        """
        Salva o modelo completo.
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        self.autoencoder.save(filepath)
        
    def load_weights(self, filepath):
        """
        Carrega os pesos do modelo.
        
        Args:
            filepath: Caminho do arquivo de pesos
        """
        self.autoencoder.load_weights(filepath)
        self.trained = True


def train_autoencoder_from_csv(csv_files, n_endmembers, wavelength_col_prefix='reflectance_',
                                exclude_pattern='uncertainty', epochs=100, batch_size=32,
                                hidden_units=128, learning_rate=0.001, verbose=1):
    """
    Função de conveniência para treinar um autoencoder a partir de arquivos CSV.
    
    Args:
        csv_files: Lista de caminhos para arquivos CSV ou caminho único
        n_endmembers: Número de endmembers a extrair
        wavelength_col_prefix: Prefixo das colunas de reflectância
        exclude_pattern: Padrão para excluir colunas (ex: 'uncertainty')
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        hidden_units: Número de neurônios na camada oculta
        learning_rate: Taxa de aprendizado
        verbose: Nível de verbosidade
        
    Returns:
        Tupla (autoencoder, endmembers, abundances, history)
    """
    import pandas as pd
    
    # Garantir que csv_files é uma lista
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Carregar dados
    all_dataframes = []
    for filepath in csv_files:
        df = pd.read_csv(filepath, header=0)
        
        # Selecionar apenas colunas de reflectância
        reflectance_cols = [col for col in df.columns 
                           if col.startswith(wavelength_col_prefix) 
                           and exclude_pattern not in col]
        
        if not reflectance_cols:
            print(f"Aviso: Nenhuma coluna de reflectância encontrada em {filepath}")
            continue
        
        spectral_df = df[reflectance_cols]
        spectral_df = spectral_df.apply(pd.to_numeric, errors='coerce')
        spectral_df.dropna(how='all', inplace=True)
        
        if not spectral_df.empty:
            all_dataframes.append(spectral_df)
    
    if not all_dataframes:
        raise ValueError("Nenhum dado válido encontrado nos arquivos CSV fornecidos")
    
    # Concatenar todos os dataframes
    X_train_df = pd.concat(all_dataframes, ignore_index=True)
    X_train = X_train_df.values.astype(np.float32)
    
    n_samples, n_bands = X_train.shape
    print(f"Dados carregados: {n_samples} amostras, {n_bands} bandas")
    
    # Criar e treinar o autoencoder
    autoencoder = SpectralAutoencoder(n_bands, n_endmembers, hidden_units)
    autoencoder.compile(learning_rate=learning_rate, loss='sad')
    
    print("\n--- Treinando Autoencoder ---")
    history = autoencoder.train(X_train, epochs=epochs, batch_size=batch_size, 
                               verbose=verbose, normalize=True)
    
    # Extrair resultados
    endmembers = autoencoder.extract_endmembers()
    abundances = autoencoder.extract_abundances(X_train)
    
    print("\n--- Extração Concluída ---")
    print(f"Shape das abundâncias: {abundances.shape} (n_amostras, n_endmembers)")
    print(f"Shape dos endmembers: {endmembers.shape} (n_endmembers, n_bandas)")
    print(f"\nVerificação das restrições:")
    print(f"  - Soma da primeira abundância: {np.sum(abundances[0]):.6f} (deve ser ~1.0)")
    print(f"  - Valor mínimo do primeiro endmember: {np.min(endmembers[0]):.6f} (deve ser >= 0.0)")
    
    return autoencoder, endmembers, abundances, history
