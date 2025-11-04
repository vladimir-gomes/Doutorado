#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Integrado de Classificação de Vegetação da Caatinga
===========================================================
Análise de bundles, endmembers e assinaturas espectrais de satélites
PRISMA, EnMAP e EMIT para classificação automática de tipos funcionais
de vegetação da Caatinga.

Autor: Vladimir Gomes
Data: 2025
"""

import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import savgol_filter
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

@dataclass
class SpectralConfig:
    """Configuração para processamento espectral."""
    # Filtro Savitzky-Golay
    savgol_window: int = 11
    savgol_polyorder: int = 3
    
    # PCA
    n_components_pca: int = 10
    
    # Clustering
    n_endmembers: int = 5
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 5
    
    # Limites de qualidade
    min_valid_pixels: int = 100
    max_cloud_cover: float = 0.3


class SatelliteDataLoader:
    """Carregador unificado para dados de satélites hiperespectrais."""
    
    SUPPORTED_SENSORS = ['PRISMA', 'ENMAP', 'EMIT']
    
    def __init__(self, data_dir: str):
        """
        Inicializa o carregador de dados.
        
        Args:
            data_dir: Diretório raiz contendo os dados dos satélites
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Diretório não encontrado: {data_dir}")
    
    def load_enmap_scene(self, scene_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Carrega uma cena EnMAP completa (VNIR + SWIR).
        
        Args:
            scene_path: Caminho para o diretório da cena EnMAP
            
        Returns:
            Tuple (dados_radiância, metadados)
        """
        scene_path = Path(scene_path)
        vnir_file = list(scene_path.glob('*SPECTRAL_IMAGE_VNIR.TIF'))
        swir_file = list(scene_path.glob('*SPECTRAL_IMAGE_SWIR.TIF'))
        
        if not vnir_file or not swir_file:
            raise ValueError(f"Arquivos VNIR/SWIR não encontrados em {scene_path}")
        
        # Carregar VNIR
        with rasterio.open(vnir_file[0]) as vnir_src:
            vnir_data = vnir_src.read().astype(np.float32)
            metadata = vnir_src.meta.copy()
        
        # Carregar SWIR
        with rasterio.open(swir_file[0]) as swir_src:
            swir_data = swir_src.read().astype(np.float32)
        
        # Combinar VNIR + SWIR
        full_data = np.vstack((vnir_data, swir_data))
        metadata['count'] = full_data.shape[0]
        metadata['sensor'] = 'ENMAP'
        
        # Aplicar máscaras se disponíveis
        vnir_mask_file = list(scene_path.glob('*QL_PIXELMASK_VNIR.TIF'))
        swir_mask_file = list(scene_path.glob('*QL_PIXELMASK_SWIR.TIF'))
        
        if vnir_mask_file and swir_mask_file:
            with rasterio.open(vnir_mask_file[0]) as vnir_mask_src:
                vnir_mask = vnir_mask_src.read(1)
            with rasterio.open(swir_mask_file[0]) as swir_mask_src:
                swir_mask = swir_mask_src.read(1)
            
            combined_mask = np.logical_or(vnir_mask != 0, swir_mask != 0)
            full_mask = np.broadcast_to(combined_mask, full_data.shape)
            full_data[full_mask] = np.nan
        
        return full_data, metadata
    
    def load_emit_scene(self, scene_file: str) -> Tuple[np.ndarray, Dict]:
        """
        Carrega uma cena EMIT.
        
        Args:
            scene_file: Caminho para o arquivo EMIT
            
        Returns:
            Tuple (dados_reflectância, metadados)
        """
        with rasterio.open(scene_file) as src:
            data = src.read().astype(np.float32)
            metadata = src.meta.copy()
            metadata['sensor'] = 'EMIT'
        
        return data, metadata
    
    def load_prisma_scene(self, scene_file: str) -> Tuple[np.ndarray, Dict]:
        """
        Carrega uma cena PRISMA.
        
        Args:
            scene_file: Caminho para o arquivo PRISMA (HDF5 ou MAT)
            
        Returns:
            Tuple (dados, metadados)
        """
        if scene_file.endswith('.mat'):
            mat_data = loadmat(scene_file)
            # Ajustar chaves conforme estrutura do arquivo PRISMA
            if 'VNIR' in mat_data and 'SWIR' in mat_data:
                vnir = mat_data['VNIR']
                swir = mat_data['SWIR']
                data = np.concatenate([vnir, swir], axis=2)
            else:
                data = mat_data['data']
            
            metadata = {
                'sensor': 'PRISMA',
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
        else:
            # Assumir formato HDF5
            import h5py
            with h5py.File(scene_file, 'r') as f:
                # Ajustar caminhos conforme estrutura do arquivo
                data = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['VNIR_Cube'][:]
                swir = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['SWIR_Cube'][:]
                data = np.concatenate([data, swir], axis=2)
                
                metadata = {
                    'sensor': 'PRISMA',
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                }
        
        return data.astype(np.float32), metadata


class SpectralPreprocessor:
    """Pré-processamento de dados espectrais."""
    
    def __init__(self, config: SpectralConfig):
        self.config = config
    
    def remove_bad_bands(self, data: np.ndarray, 
                        bad_bands: Optional[List[int]] = None) -> np.ndarray:
        """
        Remove bandas ruidosas ou com problemas.
        
        Args:
            data: Cubo de dados (bandas, altura, largura)
            bad_bands: Lista de índices de bandas a remover
            
        Returns:
            Dados filtrados
        """
        if bad_bands is None:
            # Bandas problemáticas típicas (absorção atmosférica, bordas)
            bad_bands = []
        
        if bad_bands:
            good_bands = [i for i in range(data.shape[0]) if i not in bad_bands]
            return data[good_bands, :, :]
        return data
    
    def apply_savgol_filter(self, spectra: np.ndarray) -> np.ndarray:
        """
        Aplica filtro Savitzky-Golay para suavização espectral.
        
        Args:
            spectra: Espectros (n_pixels, n_bandas)
            
        Returns:
            Espectros suavizados
        """
        return savgol_filter(
            spectra,
            window_length=self.config.savgol_window,
            polyorder=self.config.savgol_polyorder,
            axis=1
        )
    
    def normalize_spectra(self, spectra: np.ndarray, 
                         method: str = 'minmax') -> np.ndarray:
        """
        Normaliza espectros.
        
        Args:
            spectra: Espectros (n_pixels, n_bandas)
            method: Método de normalização ('minmax', 'zscore', 'l2')
            
        Returns:
            Espectros normalizados
        """
        if method == 'minmax':
            min_val = np.nanmin(spectra, axis=1, keepdims=True)
            max_val = np.nanmax(spectra, axis=1, keepdims=True)
            return (spectra - min_val) / (max_val - min_val + 1e-10)
        elif method == 'zscore':
            scaler = StandardScaler()
            return scaler.fit_transform(spectra)
        elif method == 'l2':
            norms = np.linalg.norm(spectra, axis=1, keepdims=True)
            return spectra / (norms + 1e-10)
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    def preprocess_cube(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Pipeline completo de pré-processamento.
        
        Args:
            data: Cubo de dados (bandas, altura, largura)
            
        Returns:
            Tuple (dados_processados, estatísticas)
        """
        # Reshape para processamento
        n_bands, height, width = data.shape
        data_reshaped = data.reshape(n_bands, -1).T  # (n_pixels, n_bands)
        
        # Remover pixels com NaN
        valid_mask = ~np.isnan(data_reshaped).any(axis=1)
        valid_data = data_reshaped[valid_mask]
        
        # Aplicar filtro Savitzky-Golay
        smoothed_data = self.apply_savgol_filter(valid_data)
        
        # Normalizar
        normalized_data = self.normalize_spectra(smoothed_data, method='minmax')
        
        stats = {
            'n_pixels_total': height * width,
            'n_pixels_valid': len(valid_data),
            'n_bands': n_bands,
            'valid_ratio': len(valid_data) / (height * width)
        }
        
        return normalized_data, stats


class EndmemberExtractor:
    """Extração de endmembers usando múltiplas técnicas."""
    
    def __init__(self, config: SpectralConfig):
        self.config = config
    
    def extract_vca(self, data: np.ndarray, n_endmembers: int) -> np.ndarray:
        """
        Extração de endmembers usando Vertex Component Analysis (VCA).
        Implementação simplificada.
        
        Args:
            data: Espectros (n_pixels, n_bandas)
            n_endmembers: Número de endmembers a extrair
            
        Returns:
            Endmembers extraídos (n_endmembers, n_bandas)
        """
        # Implementação simplificada usando PCA + seleção de extremos
        pca = PCA(n_components=n_endmembers)
        projected = pca.fit_transform(data)
        
        # Selecionar pontos extremos em cada componente principal
        endmember_indices = []
        for i in range(n_endmembers):
            max_idx = np.argmax(np.abs(projected[:, i]))
            endmember_indices.append(max_idx)
        
        return data[endmember_indices]
    
    def extract_bundles_aeeb(self, data: np.ndarray, 
                            n_bundles: int = 10,
                            subset_fraction: float = 0.2) -> np.ndarray:
        """
        Extração de endmember bundles usando AEEB 
        (Adaptive Endmember Extraction via Bundles).
        
        Args:
            data: Espectros (n_pixels, n_bandas)
            n_bundles: Número de subconjuntos aleatórios
            subset_fraction: Fração de pixels em cada subconjunto
            
        Returns:
            Endmembers finais (centros dos bundles)
        """
        n_pixels = data.shape[0]
        subset_size = int(n_pixels * subset_fraction)
        n_endmembers = self.config.n_endmembers
        
        all_endmembers = []
        
        for _ in range(n_bundles):
            # Selecionar subconjunto aleatório
            indices = np.random.choice(n_pixels, subset_size, replace=False)
            subset = data[indices]
            
            # Extrair endmembers do subconjunto
            subset_endmembers = self.extract_vca(subset, n_endmembers)
            all_endmembers.append(subset_endmembers)
        
        # Concatenar todos os endmembers
        all_endmembers = np.vstack(all_endmembers)
        
        # Agrupar usando K-means para encontrar centros dos bundles
        kmeans = KMeans(n_clusters=n_endmembers, random_state=42, n_init=10)
        kmeans.fit(all_endmembers)
        
        return kmeans.cluster_centers_
    
    def cluster_endmembers(self, endmembers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Agrupa endmembers similares usando DBSCAN.
        
        Args:
            endmembers: Matriz de endmembers
            
        Returns:
            Tuple (labels, endmembers_únicos)
        """
        dbscan = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples
        )
        labels = dbscan.fit_predict(endmembers)
        
        # Calcular centróides dos clusters
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remover ruído
        
        representative_endmembers = []
        for label in unique_labels:
            cluster_members = endmembers[labels == label]
            centroid = np.mean(cluster_members, axis=0)
            representative_endmembers.append(centroid)
        
        return labels, np.array(representative_endmembers)


class SpectralUnmixing:
    """Desmistura espectral para estimação de abundâncias."""
    
    @staticmethod
    def fcls_unmix(pixel_spectrum: np.ndarray, 
                   endmembers: np.ndarray) -> np.ndarray:
        """
        Fully Constrained Least Squares (FCLS) unmixing.
        
        Args:
            pixel_spectrum: Espectro do pixel (n_bandas,)
            endmembers: Matriz de endmembers (n_endmembers, n_bandas)
            
        Returns:
            Abundâncias (n_endmembers,)
        """
        from scipy.optimize import lsq_linear
        
        # Adicionar restrição de soma = 1
        E_augmented = np.vstack([endmembers.T, np.ones(endmembers.shape[0])])
        y_augmented = np.append(pixel_spectrum, 1.0)
        
        # Resolver com restrição de não-negatividade
        result = lsq_linear(E_augmented, y_augmented, bounds=(0, np.inf))
        
        return result.x
    
    def unmix_image(self, data: np.ndarray, 
                    endmembers: np.ndarray) -> np.ndarray:
        """
        Desmistura toda a imagem.
        
        Args:
            data: Espectros (n_pixels, n_bandas)
            endmembers: Matriz de endmembers
            
        Returns:
            Abundâncias (n_pixels, n_endmembers)
        """
        n_pixels = data.shape[0]
        n_endmembers = endmembers.shape[0]
        abundances = np.zeros((n_pixels, n_endmembers))
        
        for i in range(n_pixels):
            abundances[i] = self.fcls_unmix(data[i], endmembers)
        
        return abundances


class CaatingaClassifier:
    """Classificador de tipos funcionais de vegetação da Caatinga."""
    
    CAATINGA_TYPES = {
        0: 'Arbórea Densa',
        1: 'Arbórea Aberta',
        2: 'Arbustiva Densa',
        3: 'Arbustiva Aberta',
        4: 'Herbácea',
        5: 'Solo Exposto'
    }
    
    def __init__(self):
        self.classifier = None
    
    def extract_spectral_indices(self, spectra: np.ndarray, 
                                 wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extrai índices espectrais relevantes para vegetação.
        
        Args:
            spectra: Espectros (n_pixels, n_bandas)
            wavelengths: Comprimentos de onda (n_bandas,)
            
        Returns:
            Dicionário de índices espectrais
        """
        def find_band(target_wl):
            """Encontra banda mais próxima do comprimento de onda alvo."""
            return np.argmin(np.abs(wavelengths - target_wl))
        
        # Bandas aproximadas
        red_idx = find_band(650)      # Vermelho
        nir_idx = find_band(800)      # NIR
        swir1_idx = find_band(1600)   # SWIR1
        swir2_idx = find_band(2200)   # SWIR2
        green_idx = find_band(560)    # Verde
        
        # Calcular índices
        red = spectra[:, red_idx]
        nir = spectra[:, nir_idx]
        swir1 = spectra[:, swir1_idx]
        swir2 = spectra[:, swir2_idx]
        green = spectra[:, green_idx]
        
        indices = {}
        
        # NDVI
        indices['NDVI'] = (nir - red) / (nir + red + 1e-10)
        
        # EVI
        indices['EVI'] = 2.5 * (nir - red) / (nir + 6*red - 7.5*green + 1)
        
        # NDWI
        indices['NDWI'] = (nir - swir1) / (nir + swir1 + 1e-10)
        
        # SAVI (Soil Adjusted Vegetation Index)
        L = 0.5  # Fator de ajuste
        indices['SAVI'] = ((nir - red) / (nir + red + L)) * (1 + L)
        
        # Bare Soil Index
        indices['BSI'] = ((swir1 + red) - (nir + green)) / ((swir1 + red) + (nir + green) + 1e-10)
        
        return indices
    
    def classify_vegetation_types(self, abundances: np.ndarray,
                                  spectral_indices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Classifica tipos de vegetação baseado em abundâncias e índices espectrais.
        
        Args:
            abundances: Abundâncias (n_pixels, n_endmembers)
            spectral_indices: Índices espectrais
            
        Returns:
            Classes (n_pixels,)
        """
        n_pixels = abundances.shape[0]
        classes = np.zeros(n_pixels, dtype=int)
        
        ndvi = spectral_indices['NDVI']
        savi = spectral_indices['SAVI']
        bsi = spectral_indices['BSI']
        
        for i in range(n_pixels):
            # Lógica de classificação baseada em limiares
            if bsi[i] > 0.3:
                classes[i] = 5  # Solo Exposto
            elif ndvi[i] < 0.2:
                classes[i] = 4  # Herbácea
            elif ndvi[i] < 0.4:
                if savi[i] < 0.3:
                    classes[i] = 3  # Arbustiva Aberta
                else:
                    classes[i] = 2  # Arbustiva Densa
            else:
                if savi[i] < 0.5:
                    classes[i] = 1  # Arbórea Aberta
                else:
                    classes[i] = 0  # Arbórea Densa
        
        return classes


class CaatingaPipeline:
    """Pipeline completo para classificação de vegetação da Caatinga."""
    
    def __init__(self, config: Optional[SpectralConfig] = None):
        self.config = config or SpectralConfig()
        self.loader = None
        self.preprocessor = SpectralPreprocessor(self.config)
        self.endmember_extractor = EndmemberExtractor(self.config)
        self.unmixer = SpectralUnmixing()
        self.classifier = CaatingaClassifier()
        
        self.results = {}
    
    def set_data_directory(self, data_dir: str):
        """Define diretório de dados."""
        self.loader = SatelliteDataLoader(data_dir)
    
    def process_scene(self, scene_path: str, sensor: str) -> Dict:
        """
        Processa uma cena completa.
        
        Args:
            scene_path: Caminho para a cena
            sensor: Tipo de sensor ('ENMAP', 'EMIT', 'PRISMA')
            
        Returns:
            Dicionário com resultados
        """
        print(f"\n{'='*60}")
        print(f"Processando cena {sensor}: {Path(scene_path).name}")
        print('='*60)
        
        # 1. Carregar dados
        print("\n[1/6] Carregando dados...")
        if sensor == 'ENMAP':
            data, metadata = self.loader.load_enmap_scene(scene_path)
        elif sensor == 'EMIT':
            data, metadata = self.loader.load_emit_scene(scene_path)
        elif sensor == 'PRISMA':
            data, metadata = self.loader.load_prisma_scene(scene_path)
        else:
            raise ValueError(f"Sensor não suportado: {sensor}")
        
        print(f"   Shape: {data.shape}")
        print(f"   Sensor: {metadata.get('sensor', 'Unknown')}")
        
        # 2. Pré-processamento
        print("\n[2/6] Pré-processamento...")
        processed_data, stats = self.preprocessor.preprocess_cube(data)
        print(f"   Pixels válidos: {stats['n_pixels_valid']:,} ({stats['valid_ratio']:.1%})")
        print(f"   Bandas: {stats['n_bands']}")
        
        # 3. Extração de endmembers
        print("\n[3/6] Extraindo endmembers (AEEB)...")
        endmembers = self.endmember_extractor.extract_bundles_aeeb(processed_data)
        print(f"   Endmembers extraídos: {endmembers.shape[0]}")
        
        # 4. Desmistura espectral
        print("\n[4/6] Desmistura espectral (FCLS)...")
        abundances = self.unmixer.unmix_image(processed_data, endmembers)
        print(f"   Abundâncias shape: {abundances.shape}")
        
        # 5. Calcular índices espectrais
        print("\n[5/6] Calculando índices espectrais...")
        # Gerar comprimentos de onda aproximados
        n_bands = processed_data.shape[1]
        wavelengths = np.linspace(400, 2500, n_bands)  # Aproximação
        spectral_indices = self.classifier.extract_spectral_indices(
            processed_data, wavelengths
        )
        print(f"   Índices calculados: {len(spectral_indices)}")
        
        # 6. Classificação
        print("\n[6/6] Classificando tipos de vegetação...")
        classes = self.classifier.classify_vegetation_types(
            abundances, spectral_indices
        )
        
        # Estatísticas das classes
        unique, counts = np.unique(classes, return_counts=True)
        print("\n   Distribuição de classes:")
        for cls, count in zip(unique, counts):
            cls_name = self.classifier.CAATINGA_TYPES.get(cls, 'Desconhecido')
            percentage = (count / len(classes)) * 100
            print(f"      {cls_name}: {count:,} pixels ({percentage:.1f}%)")
        
        # Armazenar resultados
        results = {
            'metadata': metadata,
            'stats': stats,
            'endmembers': endmembers,
            'abundances': abundances,
            'spectral_indices': spectral_indices,
            'classes': classes,
            'processed_data': processed_data
        }
        
        return results
    
    def visualize_results(self, results: Dict, output_dir: Optional[str] = None):
        """
        Visualiza resultados da classificação.
        
        Args:
            results: Dicionário de resultados
            output_dir: Diretório para salvar figuras (opcional)
        """
        import matplotlib.pyplot as plt
        
        # Criar figura com múltiplos subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Resultados da Classificação de Vegetação da Caatinga', 
                    fontsize=16, fontweight='bold')
        
        # 1. Endmembers
        ax = axes[0, 0]
        endmembers = results['endmembers']
        for i, endmember in enumerate(endmembers):
            ax.plot(endmember, label=f'EM-{i+1}', alpha=0.7)
        ax.set_title('Endmembers Extraídos')
        ax.set_xlabel('Banda')
        ax.set_ylabel('Reflectância Normalizada')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2-4. Índices espectrais
        indices_to_plot = ['NDVI', 'SAVI', 'BSI']
        for idx, (i, index_name) in enumerate(zip([1, 2, 3], indices_to_plot)):
            ax_row = i // 3
            ax_col = i % 3
            ax = axes[ax_row, ax_col]
            
            index_values = results['spectral_indices'][index_name]
            ax.hist(index_values, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{index_name}')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
        
        # 5. Distribuição de abundâncias
        ax = axes[1, 0]
        abundances = results['abundances']
        mean_abundances = np.mean(abundances, axis=0)
        ax.bar(range(len(mean_abundances)), mean_abundances)
        ax.set_title('Abundância Média por Endmember')
        ax.set_xlabel('Endmember')
        ax.set_ylabel('Abundância Média')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Distribuição de classes
        ax = axes[1, 1]
        classes = results['classes']
        unique, counts = np.unique(classes, return_counts=True)
        class_names = [self.classifier.CAATINGA_TYPES[c] for c in unique]
        ax.bar(class_names, counts)
        ax.set_title('Distribuição de Classes')
        ax.set_xlabel('Tipo de Vegetação')
        ax.set_ylabel('Contagem de Pixels')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / 'classification_results.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nFigura salva em: {output_path}")
        
        plt.show()
    
    def export_results(self, results: Dict, output_path: str):
        """
        Exporta resultados para arquivo.
        
        Args:
            results: Dicionário de resultados
            output_path: Caminho do arquivo de saída
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Criar DataFrame com resultados
        df = pd.DataFrame({
            'class': results['classes'],
            'class_name': [self.classifier.CAATINGA_TYPES[c] for c in results['classes']],
            'NDVI': results['spectral_indices']['NDVI'],
            'SAVI': results['spectral_indices']['SAVI'],
            'EVI': results['spectral_indices']['EVI'],
            'NDWI': results['spectral_indices']['NDWI'],
            'BSI': results['spectral_indices']['BSI'],
        })
        
        # Adicionar abundâncias
        for i in range(results['abundances'].shape[1]):
            df[f'abundance_EM{i+1}'] = results['abundances'][:, i]
        
        # Salvar
        df.to_csv(output_path, index=False)
        print(f"\nResultados exportados para: {output_path}")
        
        # Salvar estatísticas resumidas
        stats_path = output_path.parent / f"{output_path.stem}_stats.txt"
        with open(stats_path, 'w') as f:
            f.write("ESTATÍSTICAS DA CLASSIFICAÇÃO\n")
            f.write("="*50 + "\n\n")
            
            f.write("Distribuição de Classes:\n")
            f.write("-"*30 + "\n")
            unique, counts = np.unique(results['classes'], return_counts=True)
            total = len(results['classes'])
            for cls, count in zip(unique, counts):
                cls_name = self.classifier.CAATINGA_TYPES[cls]
                percentage = (count / total) * 100
                f.write(f"{cls_name:20s}: {count:8,} pixels ({percentage:5.1f}%)\n")
            
            f.write(f"\n{'Total':20s}: {total:8,} pixels (100.0%)\n")
        
        print(f"Estatísticas salvas em: {stats_path}")


def main():
    """Função principal de exemplo."""
    # Exemplo de uso
    config = SpectralConfig(
        n_endmembers=5,
        savgol_window=11,
        savgol_polyorder=3
    )
    
    pipeline = CaatingaPipeline(config)
    
    # Definir diretório de dados
    # pipeline.set_data_directory('/path/to/data')
    
    # Processar cena
    # results = pipeline.process_scene('/path/to/scene', sensor='ENMAP')
    
    # Visualizar
    # pipeline.visualize_results(results, output_dir='/path/to/output')
    
    # Exportar
    # pipeline.export_results(results, '/path/to/output/results.csv')
    
    print("\n" + "="*60)
    print("Pipeline de Classificação da Caatinga - Inicializado")
    print("="*60)
    print("\nPara usar:")
    print("1. pipeline = CaatingaPipeline()")
    print("2. pipeline.set_data_directory('/path/to/data')")
    print("3. results = pipeline.process_scene('/path/to/scene', sensor='ENMAP')")
    print("4. pipeline.visualize_results(results)")
    print("5. pipeline.export_results(results, '/path/to/output.csv')")


if __name__ == '__main__':
    main()
