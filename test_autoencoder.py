"""
Testes para o módulo de autoencoder espectral.

Testa as funcionalidades principais:
- Construção de encoder e decoder
- Função de perda SAD
- Treinamento do autoencoder
- Extração de endmembers e abundâncias
- Verificação de restrições físicas
"""

import numpy as np
import sys
import os

# Adicionar o diretório raiz ao path para importações
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from spectral_analysis.autoencoder import (
    SpectralAutoencoder, sad_loss, build_encoder, build_decoder
)


def test_encoder_decoder_construction():
    """Testa a construção do encoder e decoder."""
    print("\nTeste 1: Construção do encoder e decoder")
    print("-" * 60)
    
    n_bands = 100
    n_endmembers = 5
    
    encoder = build_encoder(n_bands, n_endmembers)
    decoder = build_decoder(n_endmembers, n_bands)
    
    assert encoder.input_shape == (None, n_bands), "Forma de entrada do encoder incorreta"
    assert encoder.output_shape == (None, n_endmembers), "Forma de saída do encoder incorreta"
    assert decoder.input_shape == (None, n_endmembers), "Forma de entrada do decoder incorreta"
    assert decoder.output_shape == (None, n_bands), "Forma de saída do decoder incorreta"
    
    print(f"✓ Encoder: {n_bands} bandas → {n_endmembers} abundâncias")
    print(f"✓ Decoder: {n_endmembers} abundâncias → {n_bands} bandas")
    print("✓ Arquitetura validada com sucesso!")


def test_sad_loss_function():
    """Testa a função de perda SAD."""
    print("\nTeste 2: Função de perda SAD")
    print("-" * 60)
    
    import tensorflow as tf
    
    # Criar dois espectros idênticos (SAD deve ser ~0)
    y_true = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y_pred = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]])
    loss_identical = sad_loss(y_true, y_pred).numpy()
    
    print(f"SAD entre espectros idênticos: {loss_identical:.6f}")
    assert loss_identical < 0.01, "SAD entre espectros idênticos deve ser próximo de 0"
    
    # Criar dois espectros ortogonais (SAD deve ser ~π/2)
    y_true = tf.constant([[1.0, 0.0]])
    y_pred = tf.constant([[0.0, 1.0]])
    loss_orthogonal = sad_loss(y_true, y_pred).numpy()
    
    print(f"SAD entre espectros ortogonais: {loss_orthogonal:.6f}")
    assert 1.5 < loss_orthogonal < 1.6, "SAD entre espectros ortogonais deve ser ~π/2"
    
    print("✓ Função de perda SAD validada!")


def test_autoencoder_class():
    """Testa a classe SpectralAutoencoder."""
    print("\nTeste 3: Classe SpectralAutoencoder")
    print("-" * 60)
    
    n_bands = 50
    n_endmembers = 3
    
    autoencoder = SpectralAutoencoder(n_bands, n_endmembers)
    
    assert autoencoder.n_bands == n_bands
    assert autoencoder.n_endmembers == n_endmembers
    assert not autoencoder.trained
    
    print(f"✓ Autoencoder criado: {n_bands} bandas, {n_endmembers} endmembers")
    print("✓ Atributos inicializados corretamente")


def test_autoencoder_training():
    """Testa o treinamento do autoencoder com dados sintéticos."""
    print("\nTeste 4: Treinamento do autoencoder")
    print("-" * 60)
    
    # Criar dados sintéticos
    n_samples = 100
    n_bands = 50
    n_endmembers = 3
    
    np.random.seed(42)
    
    # Criar endmembers sintéticos
    true_endmembers = np.random.rand(n_endmembers, n_bands) * 0.8 + 0.1
    
    # Criar abundâncias sintéticas (soma = 1)
    true_abundances = np.random.dirichlet(np.ones(n_endmembers), n_samples)
    
    # Criar espectros mistos
    X_train = true_abundances @ true_endmembers
    
    # Adicionar ruído pequeno
    noise = np.random.normal(0, 0.01, X_train.shape)
    X_train += noise
    X_train = np.clip(X_train, 0, 1)
    
    print(f"Dados sintéticos: {n_samples} amostras, {n_bands} bandas")
    
    # Criar e treinar autoencoder
    autoencoder = SpectralAutoencoder(n_bands, n_endmembers, hidden_units=64)
    autoencoder.compile(learning_rate=0.01, loss='sad')
    
    print("Treinando autoencoder (10 épocas)...")
    history = autoencoder.train(X_train, epochs=10, batch_size=16, verbose=0)
    
    assert autoencoder.trained, "Autoencoder deve estar marcado como treinado"
    assert len(history.history['loss']) == 10, "Histórico deve ter 10 épocas"
    
    # Verificar que a perda diminuiu
    initial_loss = history.history['loss'][0]
    final_loss = history.history['loss'][-1]
    
    print(f"Perda inicial: {initial_loss:.6f}")
    print(f"Perda final: {final_loss:.6f}")
    print(f"Redução: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
    
    assert final_loss < initial_loss, "A perda deve diminuir durante o treinamento"
    print("✓ Treinamento concluído com sucesso!")


def test_endmember_extraction():
    """Testa a extração de endmembers."""
    print("\nTeste 5: Extração de endmembers")
    print("-" * 60)
    
    # Criar e treinar um autoencoder simples
    n_samples = 50
    n_bands = 30
    n_endmembers = 2
    
    np.random.seed(42)
    X_train = np.random.rand(n_samples, n_bands).astype(np.float32)
    
    autoencoder = SpectralAutoencoder(n_bands, n_endmembers)
    autoencoder.compile()
    autoencoder.train(X_train, epochs=5, verbose=0)
    
    # Extrair endmembers
    endmembers = autoencoder.extract_endmembers()
    
    print(f"Shape dos endmembers: {endmembers.shape}")
    assert endmembers.shape == (n_endmembers, n_bands), \
        f"Shape esperada: ({n_endmembers}, {n_bands}), obtida: {endmembers.shape}"
    
    # Verificar não-negatividade
    assert np.all(endmembers >= -1e-6), "Endmembers devem ser não-negativos"
    
    print(f"Valor mínimo: {np.min(endmembers):.6f}")
    print(f"Valor máximo: {np.max(endmembers):.6f}")
    print("✓ Endmembers extraídos com restrições corretas!")


def test_abundance_extraction():
    """Testa a extração de abundâncias."""
    print("\nTeste 6: Extração de abundâncias")
    print("-" * 60)
    
    # Criar e treinar um autoencoder simples
    n_samples = 50
    n_bands = 30
    n_endmembers = 3
    
    np.random.seed(42)
    X_train = np.random.rand(n_samples, n_bands).astype(np.float32)
    
    autoencoder = SpectralAutoencoder(n_bands, n_endmembers)
    autoencoder.compile()
    autoencoder.train(X_train, epochs=5, verbose=0)
    
    # Extrair abundâncias
    abundances = autoencoder.extract_abundances(X_train)
    
    print(f"Shape das abundâncias: {abundances.shape}")
    assert abundances.shape == (n_samples, n_endmembers), \
        f"Shape esperada: ({n_samples}, {n_endmembers}), obtida: {abundances.shape}"
    
    # Verificar restrições ANC (non-negativity) e ASC (sum-to-one)
    assert np.all(abundances >= 0), "Abundâncias devem ser não-negativas (ANC)"
    assert np.allclose(abundances.sum(axis=1), 1.0), "Abundâncias devem somar 1 (ASC)"
    
    print(f"Soma da primeira abundância: {abundances[0].sum():.6f}")
    print(f"Valor mínimo: {np.min(abundances):.6f}")
    print(f"Valor máximo: {np.max(abundances):.6f}")
    print("✓ Abundâncias extraídas com restrições ANC e ASC!")


def test_reconstruction():
    """Testa a reconstrução de espectros."""
    print("\nTeste 7: Reconstrução de espectros")
    print("-" * 60)
    
    # Criar e treinar um autoencoder
    n_samples = 20
    n_bands = 40
    n_endmembers = 4
    
    np.random.seed(42)
    X_train = np.random.rand(n_samples, n_bands).astype(np.float32)
    
    autoencoder = SpectralAutoencoder(n_bands, n_endmembers)
    autoencoder.compile()
    autoencoder.train(X_train, epochs=20, verbose=0)
    
    # Reconstruir espectros
    X_reconstructed = autoencoder.reconstruct(X_train)
    
    assert X_reconstructed.shape == X_train.shape, "Shape da reconstrução deve ser igual ao original"
    
    # Calcular erro de reconstrução
    reconstruction_error = np.mean((X_train - X_reconstructed) ** 2)
    
    print(f"Erro médio quadrático de reconstrução: {reconstruction_error:.6f}")
    assert reconstruction_error < 0.5, "Erro de reconstrução deve ser razoável"
    
    print("✓ Reconstrução de espectros validada!")


def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "=" * 60)
    print("TESTES DO MÓDULO AUTOENCODER ESPECTRAL")
    print("=" * 60)
    
    tests = [
        test_encoder_decoder_construction,
        test_sad_loss_function,
        test_autoencoder_class,
        test_autoencoder_training,
        test_endmember_extraction,
        test_abundance_extraction,
        test_reconstruction
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERRO: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTADOS: {passed} testes passaram, {failed} falharam")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
