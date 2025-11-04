#!/usr/bin/env python3
"""
Script de teste para validar o sistema de classificação
"""

import numpy as np
import sys
from pathlib import Path

# Adicionar o diretório atual ao path
sys.path.insert(0, str(Path(__file__).parent))

from caatinga_classification import (
    SpectralConfig,
    SpectralPreprocessor,
    EndmemberExtractor,
    SpectralUnmixing,
    CaatingaClassifier,
    CaatingaPipeline
)

def test_config():
    """Testa a criação de configuração."""
    print("Testando SpectralConfig...")
    config = SpectralConfig()
    assert config.n_endmembers == 5
    assert config.savgol_window == 11
    print("✓ SpectralConfig OK")

def test_preprocessor():
    """Testa o pré-processamento."""
    print("\nTestando SpectralPreprocessor...")
    config = SpectralConfig()
    preprocessor = SpectralPreprocessor(config)
    
    # Criar dados sintéticos
    n_bands = 224
    height, width = 100, 100
    data = np.random.rand(n_bands, height, width).astype(np.float32)
    
    # Processar
    processed, stats = preprocessor.preprocess_cube(data)
    
    assert processed.shape[1] == n_bands
    assert stats['n_bands'] == n_bands
    print(f"✓ Pré-processamento OK - {stats['n_pixels_valid']} pixels válidos")

def test_endmember_extraction():
    """Testa extração de endmembers."""
    print("\nTestando EndmemberExtractor...")
    config = SpectralConfig(n_endmembers=3)
    extractor = EndmemberExtractor(config)
    
    # Criar dados sintéticos
    n_pixels = 1000
    n_bands = 224
    data = np.random.rand(n_pixels, n_bands).astype(np.float32)
    
    # Extrair endmembers
    endmembers = extractor.extract_bundles_aeeb(data, n_bundles=5)
    
    assert endmembers.shape == (3, n_bands)
    print(f"✓ Extração de endmembers OK - shape: {endmembers.shape}")

def test_unmixing():
    """Testa desmistura espectral."""
    print("\nTestando SpectralUnmixing...")
    unmixer = SpectralUnmixing()
    
    # Criar dados sintéticos
    n_pixels = 100
    n_bands = 224
    n_endmembers = 3
    
    data = np.random.rand(n_pixels, n_bands).astype(np.float32)
    endmembers = np.random.rand(n_endmembers, n_bands).astype(np.float32)
    
    # Desmisturar
    abundances = unmixer.unmix_image(data, endmembers)
    
    assert abundances.shape == (n_pixels, n_endmembers)
    # Verificar que abundâncias são não-negativas
    assert np.all(abundances >= 0)
    print(f"✓ Desmistura OK - shape: {abundances.shape}")

def test_classifier():
    """Testa classificador."""
    print("\nTestando CaatingaClassifier...")
    classifier = CaatingaClassifier()
    
    # Criar dados sintéticos
    n_pixels = 1000
    n_endmembers = 5
    n_bands = 224
    
    abundances = np.random.rand(n_pixels, n_endmembers).astype(np.float32)
    spectra = np.random.rand(n_pixels, n_bands).astype(np.float32)
    wavelengths = np.linspace(400, 2500, n_bands)
    
    # Calcular índices espectrais
    indices = classifier.extract_spectral_indices(spectra, wavelengths)
    
    assert 'NDVI' in indices
    assert 'SAVI' in indices
    assert len(indices['NDVI']) == n_pixels
    
    # Classificar
    classes = classifier.classify_vegetation_types(abundances, indices)
    
    assert classes.shape == (n_pixels,)
    assert np.all(classes >= 0)
    assert np.all(classes <= 5)
    
    print(f"✓ Classificação OK - {len(np.unique(classes))} classes únicas")

def test_pipeline():
    """Testa pipeline completo."""
    print("\nTestando CaatingaPipeline...")
    config = SpectralConfig(n_endmembers=3)
    pipeline = CaatingaPipeline(config)
    
    assert pipeline.config.n_endmembers == 3
    assert pipeline.preprocessor is not None
    assert pipeline.endmember_extractor is not None
    assert pipeline.unmixer is not None
    assert pipeline.classifier is not None
    
    print("✓ Pipeline OK - todos os componentes inicializados")

def test_caatinga_types():
    """Testa tipos de vegetação da Caatinga."""
    print("\nTestando tipos de vegetação...")
    classifier = CaatingaClassifier()
    
    expected_types = {
        0: 'Arbórea Densa',
        1: 'Arbórea Aberta',
        2: 'Arbustiva Densa',
        3: 'Arbustiva Aberta',
        4: 'Herbácea',
        5: 'Solo Exposto'
    }
    
    for key, value in expected_types.items():
        assert classifier.CAATINGA_TYPES[key] == value
    
    print(f"✓ Tipos de vegetação OK - {len(classifier.CAATINGA_TYPES)} classes definidas")

def run_all_tests():
    """Executa todos os testes."""
    print("="*60)
    print("EXECUTANDO TESTES DO SISTEMA DE CLASSIFICAÇÃO")
    print("="*60)
    
    tests = [
        test_config,
        test_preprocessor,
        test_endmember_extraction,
        test_unmixing,
        test_classifier,
        test_pipeline,
        test_caatinga_types
    ]
    
    failed = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FALHOU: {e}")
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    if not failed:
        print("✓ TODOS OS TESTES PASSARAM!")
    else:
        print(f"✗ {len(failed)} TESTE(S) FALHARAM:")
        for name in failed:
            print(f"  - {name}")
    print("="*60)
    
    return len(failed) == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
