# Doutorado em Sensoriamento Remoto

Scripts e ferramentas para análise hiperespectral e classificação de vegetação da Caatinga usando dados de satélites PRISMA, EnMAP e EMIT.

## 🌳 Sistema de Classificação de Vegetação da Caatinga

Este repositório contém um sistema integrado para:
- **Carregamento unificado** de dados de múltiplos sensores hiperespectrais
- **Extração de endmembers** usando técnicas avançadas (VCA, AEEB)
- **Análise de bundles** espectrais e agrupamento
- **Desmistura espectral** (FCLS - Fully Constrained Least Squares)
- **Cálculo de índices espectrais** (NDVI, EVI, SAVI, NDWI, BSI)
- **Classificação automática** de tipos funcionais de vegetação

### 🚀 Início Rápido

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar exemplo
python caatinga_classification.py
```

### 📓 Notebooks Jupyter

- **`example_classification.ipynb`**: Exemplo completo de uso do sistema
- **`Bundles_ENMAP.ipynb`**: Análise de bundles EnMAP
- **`ETL_EMIT (1).ipynb`**: Processamento de dados EMIT
- **`pipeline_anal_espec.ipynb`**: Pipeline de análise espectral

### 📖 Documentação

- **`CAATINGA_CLASSIFICATION_GUIDE.md`**: Guia completo de uso
- **`caatinga_classification.py`**: Módulo principal com todas as classes

### 🔬 Sensores Suportados

| Sensor | Formato | Bandas | Resolução Espectral |
|--------|---------|--------|---------------------|
| **PRISMA** | HDF5, MAT | VNIR + SWIR | ~10nm |
| **EnMAP** | GeoTIFF | VNIR + SWIR | ~6-10nm |
| **EMIT** | GeoTIFF | VSWIR | ~7.4nm |

### 🌱 Classes de Vegetação Identificadas

1. **Arbórea Densa** - Alta cobertura vegetal, dossel fechado
2. **Arbórea Aberta** - Cobertura arbórea com espaçamento
3. **Arbustiva Densa** - Vegetação arbustiva densa
4. **Arbustiva Aberta** - Arbustos esparsos
5. **Herbácea** - Cobertura herbácea/gramíneas
6. **Solo Exposto** - Áreas com solo descoberto

### 💡 Exemplo de Uso

```python
from caatinga_classification import CaatingaPipeline, SpectralConfig

# Configurar
config = SpectralConfig(n_endmembers=5)
pipeline = CaatingaPipeline(config)

# Processar
pipeline.set_data_directory('/dados/satelites')
results = pipeline.process_scene('/dados/enmap/cena001', sensor='ENMAP')

# Visualizar e exportar
pipeline.visualize_results(results, output_dir='/saida')
pipeline.export_results(results, '/saida/classificacao.csv')
```

### 📊 Funcionalidades Principais

#### 1. Pré-processamento
- Filtro Savitzky-Golay para suavização
- Normalização (Min-Max, Z-score, L2)
- Remoção automática de pixels inválidos
- Aplicação de máscaras de qualidade

#### 2. Extração de Endmembers
- **VCA** (Vertex Component Analysis)
- **AEEB** (Adaptive Endmember Extraction via Bundles)
- Clustering por DBSCAN para agrupamento

#### 3. Análise Espectral
- Desmistura FCLS com restrições
- Cálculo de abundâncias
- Índices espectrais para vegetação

#### 4. Classificação
- Classificação automática baseada em limiares
- Mapeamento de tipos funcionais
- Exportação de resultados em CSV

### 🛠️ Requisitos

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
rasterio>=1.2.0
spectral>=0.22.0
matplotlib>=3.4.0
```

Ver `requirements.txt` para lista completa.

### 📁 Estrutura do Repositório

```
.
├── caatinga_classification.py      # Módulo principal
├── CAATINGA_CLASSIFICATION_GUIDE.md # Guia de uso
├── example_classification.ipynb     # Notebook de exemplo
├── requirements.txt                 # Dependências
├── Bundles_ENMAP.ipynb             # Análise de bundles
├── ETL_EMIT (1).ipynb              # Pipeline EMIT
├── pipeline_anal_espec.ipynb       # Pipeline espectral
└── README.md                        # Este arquivo
```

### 📝 Citação

Se você usar este código em sua pesquisa, por favor cite:

```
Gomes, V. (2025). Sistema Integrado de Classificação de Vegetação da Caatinga 
Usando Dados Hiperespectrais de Satélites PRISMA, EnMAP e EMIT.
Tese de Doutorado em Sensoriamento Remoto.
```

### 👤 Autor

Vladimir Gomes - Doutorado em Sensoriamento Remoto

### 📄 Licença

Este projeto faz parte de uma tese de doutorado.
