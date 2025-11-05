# AGENTS.md: Guia para Agentes de IA

Este documento fornece instruções para agentes de IA sobre como trabalhar com este repositório.

## Visão Geral do Projeto

Este repositório contém uma biblioteca Python para análise de dados de sensoriamento remoto hiperespectral. O objetivo é integrar vários scripts de análise de assinaturas espectrais, modelagem de transferência radiativa e classificação. O código foi refatorado de uma série de notebooks Jupyter para uma estrutura de biblioteca modular para melhorar a reutilização e a manutenibilidade.

## Estrutura do Código

O núcleo do projeto está no diretório `spectral_analysis/`, que está estruturado como um pacote Python.

-   `spectral_analysis/`: O pacote principal da biblioteca.
    -   `data_loader.py`: (Legado) Funções para carregar dados de assinaturas espectrais de arquivos CSV.
    -   `satellite_io.py`: A abordagem preferida para a E/S de dados. Inclui classes e funções para ler dados diretamente de imagens de satélite (ex: GeoTIFFs) e extrair assinaturas espectrais a partir de coordenadas. Ele é projetado para ser estendido para diferentes sensores como EMIT, EnMAP e PRISMA.
    -   `analysis.py`: Contém funções para análises estatísticas, tanto intra-classe (variabilidade, agrupamento) quanto inter-classe (separabilidade, PCA, importância de características).
    -   `prosail_inversion.py`: Funções para realizar a inversão do modelo de transferência radiativa PROSAIL usando uma abordagem de Look-Up Table (LUT).
    -   `pinn_inversion.py`: Contém a estrutura para uma abordagem mais avançada de inversão usando Redes Neurais Informadas pela Física (PINNs).
-   `main.py`: O script principal que serve como ponto de entrada para executar o pipeline de análise completo. Ele demonstra como usar os módulos da biblioteca `spectral_analysis` em sequência.
-   `generate_test_data.py`: Um script utilitário para gerar dados sintéticos para testes. Isso é útil para verificar o pipeline sem a necessidade de dados de satélite reais.
-   `test_data/`: Um diretório contendo os dados de teste gerados pelo `generate_test_data.py`.

## Como Começar

### 1. Instalação de Dependências

O projeto requer várias bibliotecas Python. Você pode instalá-las usando `pip`. É crucial que você garanta que todas as dependências estejam instaladas antes de executar qualquer script.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly rasterio prosail tabulate
```

### 2. Executando o Pipeline

O fluxo de trabalho principal é orquestrado pelo `main.py`. Para executar o pipeline completo usando os dados de teste sintéticos, siga estes passos:

1.  **Gere os dados de teste (se ainda não existirem):**
    ```bash
    python generate_test_data.py
    ```
2.  **Execute o script principal:**
    ```bash
    python main.py
    ```

O script `main.py` irá:
1.  Simular a extração de assinaturas espectrais (usando os dados em `test_data/`).
2.  Executar análises de separabilidade de classe.
3.  Determinar a importância das bandas espectrais.
4.  Analisar a variabilidade dentro de cada classe.
5.  Os resultados, incluindo gráficos e tabelas, serão salvos no diretório `test_data/resultados_pipeline_sat/`.

## Tarefas Comuns

-   **Adicionar Suporte para um Novo Sensor:**
    1.  Vá para `spectral_analysis/satellite_io.py`.
    2.  Crie uma nova função `load_<sensor_name>_image()`.
    3.  Dentro desta função, implemente a lógica para ler os metadados específicos do sensor para extrair os comprimentos de onda.
    4.  A função deve retornar um objeto `SatelliteImage`.
-   **Modificar a Análise:**
    1.  Abra `spectral_analysis/analysis.py`.
    2.  Modifique ou adicione funções de análise conforme necessário.
    3.  Atualize o `main.py` para chamar suas novas funções.
