# Arquitetura do Sistema de Classificação da Caatinga

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SISTEMA INTEGRADO DE CLASSIFICAÇÃO               │
│                      VEGETAÇÃO DA CAATINGA                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        CAMADA DE ENTRADA                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐                 │
│  │  PRISMA  │      │  EnMAP   │      │   EMIT   │                 │
│  │  HDF5/MAT│      │VNIR+SWIR │      │ GeoTIFF  │                 │
│  └─────┬────┘      └─────┬────┘      └─────┬────┘                 │
│        │                 │                  │                      │
│        └─────────────────┼──────────────────┘                      │
│                          │                                         │
│                  ┌───────▼────────┐                                │
│                  │SatelliteDataLoader│                            │
│                  │  - load_prisma() │                             │
│                  │  - load_enmap()  │                             │
│                  │  - load_emit()   │                             │
│                  └───────┬──────────┘                              │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           │ Cubo Hiperespectral
                           │ (bandas, altura, largura)
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                   CAMADA DE PRÉ-PROCESSAMENTO                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         SpectralPreprocessor                                │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  1. Remover bandas ruidosas                                 │   │
│  │  2. Filtro Savitzky-Golay (suavização)                      │   │
│  │  3. Normalização (Min-Max / Z-score / L2)                   │   │
│  │  4. Remoção de pixels inválidos                             │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
└────────────────────────────┼───────────────────────────────────────┘
                             │
                             │ Espectros Processados
                             │ (n_pixels, n_bandas)
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│              CAMADA DE EXTRAÇÃO DE ENDMEMBERS                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         EndmemberExtractor                                   │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  ┌─────────────┐    ┌──────────────────────┐                │ │
│  │  │     VCA     │    │        AEEB          │                │ │
│  │  │  (simples)  │    │  (robusto)           │                │ │
│  │  ├─────────────┤    ├──────────────────────┤                │ │
│  │  │ 1. PCA      │    │ 1. N subconjuntos    │                │ │
│  │  │ 2. Extremos │    │ 2. VCA em cada       │                │ │
│  │  │             │    │ 3. K-means clusters  │                │ │
│  │  │             │    │ 4. DBSCAN grouping   │                │ │
│  │  └─────┬───────┘    └──────────┬───────────┘                │ │
│  │        └──────────────┬─────────┘                            │ │
│  │                       │                                      │ │
│  └───────────────────────┼──────────────────────────────────────┘ │
└─────────────────────────┼────────────────────────────────────────┘
                          │
                          │ Endmembers
                          │ (n_endmembers, n_bandas)
                          │
┌─────────────────────────▼──────────────────────────────────────────┐
│                 CAMADA DE DESMISTURA ESPECTRAL                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         SpectralUnmixing                                     │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │  FCLS (Fully Constrained Least Squares)                     │ │
│  │                                                              │ │
│  │  Para cada pixel:                                            │ │
│  │    minimizar ||E·a - y||²                                    │ │
│  │    sujeito a:                                                │ │
│  │      - a ≥ 0 (não-negatividade)                              │ │
│  │      - Σ(a) = 1 (conservação)                                │ │
│  │                                                              │ │
│  └────────────────────────┬─────────────────────────────────────┘ │
└──────────────────────────┼────────────────────────────────────────┘
                           │
                           │ Abundâncias
                           │ (n_pixels, n_endmembers)
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                   CAMADA DE ANÁLISE ESPECTRAL                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         CaatingaClassifier                                   │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  Cálculo de Índices Espectrais:                             │ │
│  │                                                              │ │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐│ │
│  │  │  NDVI  │  │  EVI   │  │  SAVI  │  │  NDWI  │  │  BSI   ││ │
│  │  ├────────┤  ├────────┤  ├────────┤  ├────────┤  ├────────┤│ │
│  │  │NIR-RED │  │Enhanced│  │ Soil   │  │ Water  │  │  Soil  ││ │
│  │  │────────│  │Veg.    │  │Adjusted│  │Content │  │ Index  ││ │
│  │  │NIR+RED │  │Index   │  │Veg.    │  │        │  │        ││ │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘│ │
│  │                                                              │ │
│  └────────────────────────┬─────────────────────────────────────┘ │
└──────────────────────────┼────────────────────────────────────────┘
                           │
                           │ Índices Espectrais
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                    CAMADA DE CLASSIFICAÇÃO                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         Classificação por Limiares                           │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │                                                              │ │
│  │  BSI > 0.3 ──────────────────────────► Solo Exposto (5)     │ │
│  │  │                                                           │ │
│  │  NDVI < 0.2 ─────────────────────────► Herbácea (4)         │ │
│  │  │                                                           │ │
│  │  0.2 ≤ NDVI < 0.4 ──┬─ SAVI < 0.3 ──► Arbustiva Aberta (3) │ │
│  │                     └─ SAVI ≥ 0.3 ──► Arbustiva Densa (2)  │ │
│  │  │                                                           │ │
│  │  NDVI ≥ 0.4 ────────┬─ SAVI < 0.5 ──► Arbórea Aberta (1)   │ │
│  │                     └─ SAVI ≥ 0.5 ──► Arbórea Densa (0)    │ │
│  │                                                              │ │
│  └────────────────────────┬─────────────────────────────────────┘ │
└──────────────────────────┼────────────────────────────────────────┘
                           │
                           │ Classes
                           │ (n_pixels,)
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                      CAMADA DE SAÍDA                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Visualizações   │  │   Dados CSV      │  │  Estatísticas   │ │
│  ├──────────────────┤  ├──────────────────┤  ├─────────────────┤ │
│  │ • Endmembers     │  │ • Abundâncias    │  │ • Distribuição  │ │
│  │ • Índices        │  │ • Índices        │  │   de classes    │ │
│  │ • Distribuição   │  │ • Classes        │  │ • Métricas      │ │
│  │ • Mapas          │  │ • Coordenadas    │  │ • Resumos       │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    FLUXO DE DADOS RESUMIDO                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Dados Brutos → Pré-processamento → Extração Endmembers →         │
│  → Desmistura → Índices Espectrais → Classificação → Resultados   │
│                                                                    │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ PRISMA  │   │          │   │          │   │          │        │
│  │ EnMAP   │──►│  Filter  │──►│ AEEB/VCA │──►│   FCLS   │──►     │
│  │  EMIT   │   │ Normalize│   │ Extract  │   │ Unmixing │        │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘        │
│                                                                    │
│       ▼                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                      │
│  │  NDVI    │   │          │   │  Mapas   │                      │
│  │  SAVI    │──►│ Classify │──►│   CSV    │                      │
│  │   BSI    │   │ Threshld │   │  Stats   │                      │
│  └──────────┘   └──────────┘   └──────────┘                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

LEGENDA:
━━━━━  Fluxo principal de dados
┌────┐  Componente do sistema
  │     Conexão entre componentes
  ▼     Direção do fluxo
```

## Classes Principais

```
CaatingaPipeline
├── SatelliteDataLoader
│   ├── load_enmap_scene()
│   ├── load_emit_scene()
│   └── load_prisma_scene()
│
├── SpectralPreprocessor
│   ├── remove_bad_bands()
│   ├── apply_savgol_filter()
│   ├── normalize_spectra()
│   └── preprocess_cube()
│
├── EndmemberExtractor
│   ├── extract_vca()
│   ├── extract_bundles_aeeb()
│   └── cluster_endmembers()
│
├── SpectralAutoencoder (NEW)
│   ├── build_encoder()
│   ├── build_decoder()
│   ├── train()
│   ├── extract_endmembers()
│   ├── extract_abundances()
│   └── reconstruct()
│
├── SpectralUnmixing
│   ├── fcls_unmix()
│   └── unmix_image()
│
└── CaatingaClassifier
    ├── extract_spectral_indices()
    └── classify_vegetation_types()
```

## Tipos de Dados em Cada Etapa

```
Input:        (bandas, altura, largura)  [float32]
              Ex: (224, 1000, 1000)

Preprocessed: (n_pixels, n_bandas)       [float32]
              Ex: (950000, 224)

Endmembers:   (n_endmembers, n_bandas)   [float32]
              Ex: (5, 224)

Abundances:   (n_pixels, n_endmembers)   [float32]
              Ex: (950000, 5)

Indices:      Dict[str, ndarray]         [float32]
              Ex: {'NDVI': (950000,), ...}

Classes:      (n_pixels,)                [int]
              Ex: (950000,) valores 0-5
```

## Configurações

```python
SpectralConfig(
    savgol_window=11,       # Janela do filtro (ímpar)
    savgol_polyorder=3,     # Ordem do polinômio
    n_components_pca=10,    # Componentes principais
    n_endmembers=5,         # Endmembers a extrair
    dbscan_eps=0.3,         # Distância DBSCAN
    dbscan_min_samples=5,   # Amostras mínimas
    min_valid_pixels=100,   # Pixels válidos mínimos
    max_cloud_cover=0.3     # Cobertura máxima (30%)
)
```

## Autoencoder para Recuperação de Assinaturas Espectrais

### Visão Geral

O módulo `spectral_analysis/autoencoder.py` implementa um autoencoder com restrições físicas (physics-constrained) para extração não-supervisionada de endmembers e abundâncias de dados hiperespectrais.

### Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTOENCODER COM RESTRIÇÕES FÍSICAS                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: Espectro X                                                  │
│  (n_bandas,)                                                        │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────┐                                                 │
│  │   ENCODER (E)  │                                                 │
│  ├────────────────┤                                                 │
│  │ Dense(128)     │ ← LeakyReLU(α=0.02)                            │
│  │ Dense(n_end)   │ ← Softmax (ASC + ANC)                          │
│  └────────┬───────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│  Abundâncias Z                                                      │
│  (n_endmembers,)                                                    │
│  • z_i ≥ 0        (ANC: Abundance Non-negativity Constraint)       │
│  • Σ(z_i) = 1     (ASC: Abundance Sum-to-one Constraint)           │
│           │                                                         │
│           ▼                                                         │
│  ┌────────────────┐                                                 │
│  │   DECODER (D)  │                                                 │
│  ├────────────────┤                                                 │
│  │ Dense(n_bands) │ ← Linear, no bias                              │
│  │ Weights = M    │ ← NonNeg constraint                            │
│  └────────┬───────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│  Reconstrução X̂                                                     │
│  (n_bandas,)                                                        │
│                                                                     │
│  Loss: SAD(X, X̂) = arccos(<X̂, X> / ||X̂|| ||X||)                  │
│        (Spectral Angle Divergence)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Componentes Principais

#### 1. Encoder
- **Camada Oculta**: Dense(128) com LeakyReLU(α=0.02)
- **Camada Latente**: Dense(n_endmembers) com Softmax
- **Saída**: Abundâncias com restrições ASC e ANC

#### 2. Decoder
- **Camada Linear**: Dense(n_bands) sem bias
- **Pesos W = M**: Representam os endmembers
- **Restrição**: NonNeg() para não-negatividade

#### 3. Função de Perda
- **SAD (Spectral Angle Divergence)**
- Independente de magnitude
- Focada na forma espectral
- Robusta a variações de iluminação

### Restrições Físicas

```python
# Restrições de Abundância (Encoder)
ASC: Σ(z_i) = 1      # Sum-to-one (softmax)
ANC: z_i ≥ 0         # Non-negativity (softmax)

# Restrições de Endmember (Decoder)
M_ij ≥ 0             # Non-negativity constraint
```

### Uso Básico

```python
from spectral_analysis.autoencoder import SpectralAutoencoder

# Criar autoencoder
autoencoder = SpectralAutoencoder(
    n_bands=224,           # Bandas EnMAP
    n_endmembers=5,        # Número de componentes
    hidden_units=128       # Neurônios na camada oculta
)

# Compilar
autoencoder.compile(learning_rate=0.001, loss='sad')

# Treinar
history = autoencoder.train(
    X_train,               # (n_samples, n_bands)
    epochs=100,
    batch_size=32,
    normalize=True
)

# Extrair resultados
endmembers = autoencoder.extract_endmembers()    # (n_endmembers, n_bands)
abundances = autoencoder.extract_abundances(X)   # (n_samples, n_endmembers)
```

### Vantagens

1. **Não-supervisionado**: Não requer labels ou dados de treinamento rotulados
2. **Restrições Físicas**: Garante resultados fisicamente plausíveis
3. **Robusto**: SAD é invariante a iluminação
4. **Flexível**: Número de endmembers configurável
5. **Rápido**: Treinamento eficiente com TensorFlow

### Aplicações

- Extração de endmembers de imagens hiperespectrais
- Desmistura espectral não-supervisionada
- Detecção de componentes puros em cenas mistas
- Análise de assinaturas espectrais EnMAP, PRISMA, EMIT
- Redução de dimensionalidade preservando informação espectral

### Métricas de Validação

```python
# Verificar restrições
assert np.all(abundances >= 0), "Restrição ANC"
assert np.allclose(abundances.sum(axis=1), 1.0), "Restrição ASC"
assert np.all(endmembers >= 0), "Endmembers não-negativos"

# Erro de reconstrução
X_reconstructed = autoencoder.reconstruct(X)
mse = np.mean((X - X_reconstructed) ** 2)
```

### Referências Científicas

- SAD Loss: Spectral Angle Mapper (SAM) para comparação espectral
- Restrições ANC/ASC: Padrão em desmistura espectral
- Arquitetura: Baseada em autoencoders lineares para unmixing
