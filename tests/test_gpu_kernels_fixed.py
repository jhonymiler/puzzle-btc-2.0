#!/usr/bin/env python3
"""
Teste de funcionalidades dos kernels GPU
=======================================

Testa as capacidades do módulo gpu_kernels.py para operações criptográficas
aceleradas por GPU quando disponível.
"""

import os
import sys
import time
import random
import pytest

# Adiciona o diretório src ao path para importação
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment_detector import get_environment_detector
from src.gpu_kernels import get_gpu_kernels

@pytest.fixture
def env_detector():
    """Fixture para detector de ambiente"""
    return get_environment_detector()

@pytest.fixture
def gpu_kernels(env_detector):
    """Fixture para kernels GPU"""
    return get_gpu_kernels(env_detector)

@pytest.fixture
def test_keys():
    """Fixture para chaves de teste"""
    return [random.randint(2**30, 2**35) for _ in range(50)]

@pytest.fixture
def test_pubkeys(gpu_kernels, test_keys):
    """Fixture para chaves públicas de teste"""
    return gpu_kernels.batch_generate_pubkeys(test_keys)

def test_environment_detection(env_detector):
    """Testa detecção de ambiente e hardware"""
    print("\n🧪 TESTE DE DETECÇÃO DE AMBIENTE")
    print("=" * 50)
    
    assert env_detector is not None
    assert 'cores' in env_detector.cpu_info
    
    print(f"Ambiente: {env_detector.environment}")
    print(f"CPU Cores: {env_detector.cpu_info['cores']}")
    print(f"GPU CUDA: {'✓' if env_detector.cuda_available else '✗'}")
    print(f"GPU ROCm: {'✓' if env_detector.rocm_available else '✗'}")
    print(f"Apple MPS: {'✓' if env_detector.mps_available else '✗'}")

def test_gpu_kernels_initialization(gpu_kernels):
    """Testa inicialização dos kernels GPU"""
    print("\n🧪 TESTE DE KERNELS GPU")
    print("=" * 50)
    
    assert gpu_kernels is not None
    assert hasattr(gpu_kernels, 'device')
    assert hasattr(gpu_kernels, 'batch_size')
    
    print(f"Dispositivo detectado: {gpu_kernels.device}")
    print(f"Tamanho de lote otimizado: {gpu_kernels.batch_size}")
    print(f"Aceleração disponível: {'Sim' if gpu_kernels.has_gpu else 'Não'}")

def test_key_generation(gpu_kernels, test_keys):
    """Testa geração de chaves públicas"""
    print("\n🧪 TESTE DE GERAÇÃO DE CHAVES")
    print("=" * 50)
    
    num_keys = len(test_keys)
    print(f"Gerando {num_keys} chaves aleatórias...")
    
    # Mede tempo para geração das chaves públicas
    start_time = time.time()
    pubkeys = gpu_kernels.batch_generate_pubkeys(test_keys)
    gen_time = time.time() - start_time
    
    print(f"Tempo de geração: {gen_time:.4f}s ({num_keys/gen_time:.1f} chaves/s)")
    
    assert pubkeys is not None
    assert len(pubkeys) == len(test_keys)
    assert all(isinstance(pk, tuple) and len(pk) == 2 for pk in pubkeys)
    
    if pubkeys and len(pubkeys) > 0:
        print(f"\nExemplo de chave privada: {test_keys[0]}")
        print(f"Exemplo de chave pública X: {pubkeys[0][0]}")
        print(f"Exemplo de chave pública Y: {pubkeys[0][1]}")

def test_fitness_calculation(gpu_kernels, test_pubkeys):
    """Testa cálculo de fitness em lote"""
    print("\n🧪 TESTE DE CÁLCULO DE FITNESS")
    print("=" * 50)
    
    assert test_pubkeys is not None and len(test_pubkeys) > 0
    
    # Usa a primeira chave pública como alvo para teste
    target_point = test_pubkeys[0]
    num_samples = min(30, len(test_pubkeys))
    
    # Seleciona algumas chaves para teste
    sample_pubkeys = test_pubkeys[:num_samples]
    
    # Mede tempo para cálculo de fitness
    start_time = time.time()
    fitness_values = gpu_kernels.batch_calculate_fitness(sample_pubkeys, target_point)
    fitness_time = time.time() - start_time
    
    print(f"Tempo de cálculo: {fitness_time:.4f}s ({num_samples/fitness_time:.1f} cálculos/s)")
    
    assert fitness_values is not None
    assert len(fitness_values) == len(sample_pubkeys)
    assert all(isinstance(f, (int, float)) for f in fitness_values)
    
    if fitness_values and len(fitness_values) > 0:
        print(f"\nExemplos de valores de fitness:")
        for i in range(min(5, len(fitness_values))):
            print(f"  Chave {i+1}: {fitness_values[i]}")

def test_hash_calculation(gpu_kernels):
    """Testa cálculo de hash em lote"""
    print("\n🧪 TESTE DE HASH SHA-256 EM LOTE")
    print("=" * 50)
    
    num_hashes = 100  # Reduzido para testes mais rápidos
    # Gera dados aleatórios para hash
    test_data = [os.urandom(32) for _ in range(num_hashes)]
    
    # Mede tempo para hash
    start_time = time.time()
    hash_results = gpu_kernels.batch_sha256(test_data)
    hash_time = time.time() - start_time
    
    print(f"Tempo para {num_hashes} hashes: {hash_time:.4f}s ({num_hashes/hash_time:.1f} hashes/s)")
    
    assert hash_results is not None
    assert len(hash_results) == len(test_data)
    assert all(isinstance(h, bytes) and len(h) == 32 for h in hash_results)
    
    if hash_results and len(hash_results) > 0:
        print(f"\nExemplo de hash: {hash_results[0].hex()[:16]}...")

def test_monte_carlo_optimization(gpu_kernels):
    """Testa otimização Monte Carlo"""
    print("\n🧪 TESTE DE OTIMIZAÇÃO MONTE CARLO")
    print("=" * 50)
    
    # Define uma função de fitness simples para teste
    def simple_fitness(x):
        return abs(x - 12345)  # Quer encontrar o número 12345
    
    # Espaço de busca
    search_space = list(range(10000, 15000))
    
    # Executa Monte Carlo
    start_time = time.time()
    best_samples = gpu_kernels.monte_carlo_optimization(
        fitness_func=simple_fitness,
        sample_space=search_space,
        num_samples=100,
        temperature=1.0
    )
    mc_time = time.time() - start_time
    
    print(f"Tempo de otimização: {mc_time:.4f}s")
    print(f"Amostras encontradas: {len(best_samples)}")
    
    assert best_samples is not None
    assert len(best_samples) > 0
    assert all(isinstance(s, int) for s in best_samples)
    
    # Verifica se encontrou bons candidatos
    best_fitness_values = [simple_fitness(s) for s in best_samples]
    print(f"Melhor fitness encontrado: {min(best_fitness_values)}")

# Função para executar testes como script independente
def run_all_tests():
    """Executa todos os testes como script independente"""
    print("\n🚀 TESTE COMPLETO DE GPU KERNELS")
    print("=" * 60)
    print(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Executa testes em sequência
    env_detector = get_environment_detector()
    gpu_kernels = get_gpu_kernels(env_detector)
    test_keys = [random.randint(2**30, 2**35) for _ in range(50)]
    test_pubkeys = gpu_kernels.batch_generate_pubkeys(test_keys)
    
    test_environment_detection(env_detector)
    test_gpu_kernels_initialization(gpu_kernels)
    test_key_generation(gpu_kernels, test_keys)
    test_fitness_calculation(gpu_kernels, test_pubkeys)
    test_hash_calculation(gpu_kernels)
    test_monte_carlo_optimization(gpu_kernels)
    
    print("\n✅ TESTES CONCLUÍDOS")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
