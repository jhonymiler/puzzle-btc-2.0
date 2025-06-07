#!/usr/bin/env python3
"""
🚀 TESTE DE INTEGRAÇÃO DOS KERNELS GPU
=====================================

Este script testa a integração entre o solucionador genético e os kernels GPU,
verificando se as otimizações GPU funcionam corretamente quando disponíveis.
"""

import os
import sys
import time
import random
import argparse
import numpy as np

# Adiciona o diretório src ao path para importação
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment_detector import get_environment_detector
from src.gpu_kernels import get_gpu_kernels
from src.genetic_bitcoin_solver import GeneticBitcoinSolver

def test_environment_settings():
    """Testa configurações do ambiente e disponibilidade de GPU"""
    print("\n🧪 TESTE DE CONFIGURAÇÕES DE AMBIENTE")
    print("=" * 50)
    
    # Detecta ambiente
    env_detector = get_environment_detector()
    
    # Imprime informações sobre GPU
    print(f"GPU NVIDIA: {'✓' if env_detector.cuda_available else '✗'}")
    print(f"GPU AMD: {'✓' if env_detector.rocm_available else '✗'}")
    print(f"GPU Apple: {'✓' if env_detector.mps_available else '✗'}")
    
    # Imprime parâmetros otimizados
    genetic_params = env_detector.get_optimal_genetic_params(difficulty_level=71)
    print("\nParâmetros genéticos otimizados:")
    print(f"- População: {genetic_params.get('population_size', 'N/A')}")
    print(f"- Taxa de mutação: {genetic_params.get('mutation_rate', 'N/A')}")
    print(f"- Taxa de crossover: {genetic_params.get('crossover_rate', 'N/A')}")
    
    return env_detector

def test_gpu_kernels_integration(env_detector):
    """Testa inicialização e integração dos kernels GPU"""
    print("\n🧪 TESTE DE INTEGRAÇÃO DOS KERNELS GPU")
    print("=" * 50)
    
    # Obtém instância dos kernels GPU
    start_time = time.time()
    gpu_kernels = get_gpu_kernels(env_detector)
    init_time = time.time() - start_time
    
    print(f"Tempo para inicialização dos kernels: {init_time:.4f}s")
    print(f"Tipo de aceleração: {gpu_kernels.device}")
    print(f"GPU disponível: {'Sim' if gpu_kernels.has_gpu else 'Não'}")
    print(f"Módulos acelerados: {'Carregados' if gpu_kernels.accelerated_modules_loaded else 'Não disponíveis'}")
    
    return gpu_kernels

def test_genetic_solver_with_gpu(env_detector, population_size=100):
    """Testa integração do solucionador genético com kernels GPU"""
    print("\n🧪 TESTE DO SOLUCIONADOR GENÉTICO COM GPU")
    print("=" * 50)
    
    # Inicializa o solucionador genético
    start_time = time.time()
    solver = GeneticBitcoinSolver(population_size=population_size)
    init_time = time.time() - start_time
    
    print(f"Tempo para inicialização do solver: {init_time:.4f}s")
    print(f"GPU disponível para o solver: {'Sim' if solver.gpu_available else 'Não'}")
    
    if solver.gpu_available:
        print(f"Kernels GPU inicializados: {'Sim' if hasattr(solver, 'gpu_kernels') and solver.gpu_kernels is not None else 'Não'}")
    
    # Testa inicialização da população (usa kernels GPU se disponíveis)
    print("\nInicializando população do algoritmo genético...")
    start_time = time.time()
    population = solver.initialize_population()
    init_time = time.time() - start_time
    
    print(f"Tempo para inicializar população: {init_time:.4f}s")
    print(f"Tamanho da população: {len(population)}")
    print(f"Melhor fitness: {population[0].fitness:.2f}")
    
    return solver, population

def test_bayesian_optimization(gpu_kernels):
    """Testa otimização bayesiana usando kernels GPU"""
    print("\n🧪 TESTE DE OTIMIZAÇÃO BAYESIANA")
    print("=" * 50)
    
    # Dados sintéticos para teste
    sample_size = 1000
    samples = [random.randint(2**70, 2**71-1) for _ in range(sample_size)]
    fitness_values = [random.random() * 1000 for _ in range(sample_size)]
    
    print(f"Tamanho das amostras: {sample_size}")
    
    # Testa inferência bayesiana
    start_time = time.time()
    posterior = gpu_kernels.batch_bayesian_inference(samples, fitness_values, exploration_factor=0.2)
    inference_time = time.time() - start_time
    
    print(f"Tempo para inferência bayesiana: {inference_time:.4f}s")
    print(f"Tamanho do posterior: {len(posterior)}")
    
    # Verifica se a soma das probabilidades é aproximadamente 1.0
    posterior_sum = sum(posterior)
    print(f"Soma das probabilidades: {posterior_sum:.8f}")
    
    # Testa otimização Monte Carlo
    def dummy_fitness(x):
        # Função de fitness simples para teste
        return np.sin(x / 10000000000) + random.random() * 0.1
    
    print("\nExecutando otimização Monte Carlo...")
    start_time = time.time()
    optimal_samples = gpu_kernels.monte_carlo_optimization(
        dummy_fitness, 
        samples, 
        num_samples=50, 
        temperature=1.0
    )
    mc_time = time.time() - start_time
    
    print(f"Tempo para otimização Monte Carlo: {mc_time:.4f}s")
    print(f"Número de amostras otimizadas: {len(optimal_samples)}")
    
    return posterior, optimal_samples

def test_end_to_end_evolution(solver, initial_population, generations=3):
    """Testa evolução completa com kernels GPU"""
    print("\n🧪 TESTE DE EVOLUÇÃO COMPLETA")
    print("=" * 50)
    
    population = initial_population
    
    for gen in range(generations):
        print(f"\nGeração {gen+1}/{generations}")
        
        start_time = time.time()
        population = solver.evolve_generation(population)
        gen_time = time.time() - start_time
        
        print(f"Tempo para evolução: {gen_time:.4f}s")
        print(f"Melhor fitness: {population[0].fitness:.2f}")
        print(f"Pior fitness: {population[-1].fitness:.2f}")
        
        # Diversidade populacional
        diversity = solver._calculate_population_diversity(population)
        print(f"Diversidade da população: {diversity:.4f}")
    
    return population

def run_integration_tests(args):
    """Executa todos os testes de integração"""
    print("\n🚀 TESTE DE INTEGRAÇÃO COMPLETO GPU KERNELS + GENETIC SOLVER")
    print("=" * 60)
    print(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configurações de teste
    population_size = args.population_size
    generations = args.generations
    
    # Executa testes em sequência
    env_detector = test_environment_settings()
    gpu_kernels = test_gpu_kernels_integration(env_detector)
    solver, population = test_genetic_solver_with_gpu(env_detector, population_size)
    
    if gpu_kernels.has_gpu and gpu_kernels.accelerated_modules_loaded:
        posterior, optimal_samples = test_bayesian_optimization(gpu_kernels)
    
    if generations > 0:
        final_population = test_end_to_end_evolution(solver, population, generations)
    
    print("\n✅ TESTES DE INTEGRAÇÃO CONCLUÍDOS")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste de integração dos kernels GPU")
    parser.add_argument('--population', dest='population_size', type=int, default=100,
                       help='Tamanho da população para teste (padrão: 100)')
    parser.add_argument('--generations', type=int, default=2,
                       help='Número de gerações para testar (padrão: 2)')
    args = parser.parse_args()
    
    run_integration_tests(args)
