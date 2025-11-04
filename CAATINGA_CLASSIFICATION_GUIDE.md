# Guia de Classificação de Vegetação da Caatinga

## Visão Geral

Este sistema integrado realiza análise de bundles, extração de endmembers e classificação automática de tipos funcionais de vegetação da Caatinga usando dados hiperespectrais de satélites PRISMA, EnMAP e EMIT.

## Funcionalidades Principais

### 1. Carregamento Unificado de Dados
- **PRISMA**: Suporte para formatos HDF5 e MAT
- **EnMAP**: Carregamento automático de VNIR + SWIR com máscaras
- **EMIT**: Processamento de dados de reflectância

### 2. Pré-processamento Avançado
- Remoção de bandas ruidosas
- Filtro Savitzky-Golay para suavização espectral
- Normalização (Min-Max, Z-score, L2)
- Remoção automática de pixels inválidos

### 3. Extração de Endmembers
- **VCA (Vertex Component Analysis)**: Método baseado em PCA
- **AEEB (Adaptive Endmember Extraction via Bundles)**: 
  - Extração em múltiplos subconjuntos aleatórios
  - Agrupamento por K-means para robustez
  - Clustering adicional por DBSCAN

### 4. Desmistura Espectral
- **FCLS (Fully Constrained Least Squares)**:
  - Restrição de não-negatividade
  - Restrição de soma das abundâncias = 1
  - Estimação robusta de abundâncias

### 5. Índices Espectrais para Vegetação
- **NDVI**: Normalized Difference Vegetation Index
- **EVI**: Enhanced Vegetation Index
- **NDWI**: Normalized Difference Water Index
- **SAVI**: Soil Adjusted Vegetation Index
- **BSI**: Bare Soil Index

### 6. Classificação Automática
Tipos de vegetação da Caatinga identificados:
- **Arbórea Densa**: Alta cobertura vegetal, dossel fechado
- **Arbórea Aberta**: Cobertura arbórea com espaçamento
- **Arbustiva Densa**: Vegetação arbustiva densa
- **Arbustiva Aberta**: Arbustos esparsos
- **Herbácea**: Cobertura herbácea/gramíneas
- **Solo Exposto**: Áreas com solo descoberto

## Instalação

### Dependências

```bash
pip install numpy pandas matplotlib
pip install rasterio scipy scikit-learn
pip install h5py  # Para arquivos PRISMA HDF5
```

### Importar o Módulo

```python
from caatinga_classification import CaatingaPipeline, SpectralConfig
```

## Uso Básico

### 1. Configuração

```python
# Criar configuração personalizada
config = SpectralConfig(
    savgol_window=11,           # Janela do filtro Savitzky-Golay
    savgol_polyorder=3,         # Ordem do polinômio
    n_components_pca=10,        # Componentes PCA
    n_endmembers=5,             # Número de endmembers
    dbscan_eps=0.3,             # Epsilon para DBSCAN
    dbscan_min_samples=5,       # Amostras mínimas DBSCAN
    min_valid_pixels=100,       # Mínimo de pixels válidos
    max_cloud_cover=0.3         # Cobertura máxima de nuvens
)

# Inicializar pipeline
pipeline = CaatingaPipeline(config)
```

### 2. Configurar Diretório de Dados

```python
# Definir onde estão os dados dos satélites
pipeline.set_data_directory('/caminho/para/dados')
```

### 3. Processar Uma Cena

#### EnMAP

```python
results = pipeline.process_scene(
    scene_path='/caminho/para/cena_enmap',
    sensor='ENMAP'
)
```

#### EMIT

```python
results = pipeline.process_scene(
    scene_path='/caminho/para/arquivo_emit.tif',
    sensor='EMIT'
)
```

#### PRISMA

```python
results = pipeline.process_scene(
    scene_path='/caminho/para/arquivo_prisma.h5',
    sensor='PRISMA'
)
```

### 4. Visualizar Resultados

```python
# Criar visualizações
pipeline.visualize_results(
    results,
    output_dir='/caminho/para/saida'
)
```

### 5. Exportar Resultados

```python
# Exportar para CSV
pipeline.export_results(
    results,
    output_path='/caminho/para/saida/resultados.csv'
)
```

## Estrutura dos Resultados

O dicionário `results` contém:

```python
{
    'metadata': {
        'sensor': 'ENMAP',
        'count': 224,  # número de bandas
        # ... outros metadados
    },
    'stats': {
        'n_pixels_total': 1024000,
        'n_pixels_valid': 950000,
        'n_bands': 224,
        'valid_ratio': 0.928
    },
    'endmembers': np.ndarray,        # (n_endmembers, n_bands)
    'abundances': np.ndarray,         # (n_pixels, n_endmembers)
    'spectral_indices': {
        'NDVI': np.ndarray,
        'SAVI': np.ndarray,
        'EVI': np.ndarray,
        'NDWI': np.ndarray,
        'BSI': np.ndarray
    },
    'classes': np.ndarray,            # (n_pixels,)
    'processed_data': np.ndarray      # (n_pixels, n_bands)
}
```

## Exemplo Completo

```python
from caatinga_classification import CaatingaPipeline, SpectralConfig

# 1. Configurar
config = SpectralConfig(n_endmembers=6)
pipeline = CaatingaPipeline(config)

# 2. Definir diretório
pipeline.set_data_directory('/dados/satelites')

# 3. Processar cena EnMAP
results_enmap = pipeline.process_scene(
    '/dados/satelites/ENMAP/cena001',
    sensor='ENMAP'
)

# 4. Processar cena EMIT
results_emit = pipeline.process_scene(
    '/dados/satelites/EMIT/cena002.tif',
    sensor='EMIT'
)

# 5. Visualizar
pipeline.visualize_results(results_enmap, output_dir='/saida/enmap')
pipeline.visualize_results(results_emit, output_dir='/saida/emit')

# 6. Exportar
pipeline.export_results(results_enmap, '/saida/enmap_classificacao.csv')
pipeline.export_results(results_emit, '/saida/emit_classificacao.csv')

# 7. Analisar resultados
print(f"Classes EnMAP: {set(results_enmap['classes'])}")
print(f"Classes EMIT: {set(results_emit['classes'])}")

# 8. Comparar índices espectrais
import numpy as np
print(f"NDVI médio EnMAP: {np.mean(results_enmap['spectral_indices']['NDVI']):.3f}")
print(f"NDVI médio EMIT: {np.mean(results_emit['spectral_indices']['NDVI']):.3f}")
```

## Formato de Dados Esperado

### EnMAP
```
cena_enmap/
├── *_SPECTRAL_IMAGE_VNIR.TIF
├── *_SPECTRAL_IMAGE_SWIR.TIF
├── *_QL_PIXELMASK_VNIR.TIF (opcional)
└── *_QL_PIXELMASK_SWIR.TIF (opcional)
```

### EMIT
```
arquivo_emit.tif  # Arquivo GeoTIFF com todas as bandas
```

### PRISMA
```
arquivo_prisma.h5   # HDF5 com estrutura HDFEOS
# ou
arquivo_prisma.mat  # MATLAB com variáveis VNIR e SWIR
```

## Personalização

### Alterar Limiares de Classificação

```python
from caatinga_classification import CaatingaClassifier

# Subclassificar e modificar
class CustomClassifier(CaatingaClassifier):
    def classify_vegetation_types(self, abundances, spectral_indices):
        ndvi = spectral_indices['NDVI']
        # Implementar lógica customizada
        classes = np.zeros(len(ndvi), dtype=int)
        # ... sua lógica aqui ...
        return classes

# Usar no pipeline
pipeline.classifier = CustomClassifier()
```

### Adicionar Novos Índices Espectrais

```python
# Modificar método extract_spectral_indices
def extract_custom_indices(self, spectra, wavelengths):
    indices = self.extract_spectral_indices(spectra, wavelengths)
    
    # Adicionar novo índice
    def find_band(target_wl):
        return np.argmin(np.abs(wavelengths - target_wl))
    
    # Exemplo: Carotenoid Reflectance Index
    b510 = find_band(510)
    b550 = find_band(550)
    indices['CRI'] = (1/spectra[:, b510]) - (1/spectra[:, b550])
    
    return indices
```

## Solução de Problemas

### Erro: "Arquivos VNIR/SWIR não encontrados"
- Verifique a estrutura de diretórios
- Confirme que os arquivos terminam com `*_SPECTRAL_IMAGE_VNIR.TIF` e `*_SPECTRAL_IMAGE_SWIR.TIF`

### Erro: "Não há pixels válidos suficientes"
- Aumente `subset_fraction` na extração de bundles
- Reduza `min_valid_pixels` na configuração
- Verifique máscaras de qualidade

### Resultados Inesperados
- Ajuste os limiares de classificação
- Modifique `n_endmembers` para capturar mais variabilidade
- Verifique a calibração radiométrica dos dados

## Referências

### Métodos Implementados

1. **VCA**: Nascimento, J. M., & Dias, J. M. (2005). Vertex component analysis.
2. **FCLS**: Heinz, D. C. (2001). Fully constrained least squares linear spectral mixture analysis.
3. **AEEB**: Conceito de bundle extraction adaptado de múltiplas referências.
4. **Índices Espectrais**: Tucker, C. J. (1979). Red and photographic infrared linear combinations.

## Contribuindo

Para reportar bugs ou sugerir melhorias:
1. Abra uma issue no repositório
2. Descreva o problema ou sugestão detalhadamente
3. Inclua exemplos de código quando possível

## Licença

Este código é parte da tese de doutorado em Sensoriamento Remoto.

## Contato

Vladimir Gomes - Doutorado em Sensoriamento Remoto

---

**Última atualização**: 2025
