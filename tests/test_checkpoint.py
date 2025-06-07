#!/usr/bin/env python3
"""
🔄 VERIFICAÇÃO DE CHECKPOINTS E RETOMADA
======================================

Script para verificar a funcionalidade de checkpoint e retomada
do Bitcoin Puzzle Solver. Realiza testes para garantir que:

1. Os checkpoints são criados corretamente
2. O estado pode ser salvo e restaurado sem perda de dados
3. A execução pode ser retomada do último estado salvo
"""

import os
import sys
import json
import time
import datetime
import argparse
import random

# Adiciona o diretório src ao path para importação
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment_detector import get_environment_detector
from src.genetic_bitcoin_solver import GeneticBitcoinSolver

def test_checkpoint_creation(generations=5, population_size=100):
    """Testa a criação de checkpoints durante a execução"""
    print("\n🧪 TESTE DE CRIAÇÃO DE CHECKPOINTS")
    print("=" * 50)
    
    # Inicializa o solver com população pequena para teste rápido
    env_detector = get_environment_detector()
    solver = GeneticBitcoinSolver(population_size=population_size)
    
    # Configurações para teste
    solver.checkpoint_interval = 1  # Checkpoint a cada geração para teste
    solver.checkpoint_file = "test_checkpoint.json"
    
    # Inicializa população
    population = solver.initialize_population()
    
    # Executa algumas gerações
    print(f"Executando {generations} gerações para teste...")
    for i in range(generations):
        print(f"\nGeração {i+1}/{generations}")
        population = solver.evolve_generation(population)
        print(f"Melhor fitness: {solver.best_fitness:.2f}")
        time.sleep(1)  # Pequena pausa para diferenciar timestamps
    
    # Salva checkpoint final explicitamente
    solver.save_checkpoint(population)
    
    # Verifica se o checkpoint foi criado
    checkpoint_file = "test_checkpoint.json"
    if os.path.exists(checkpoint_file):
        checkpoint_size = os.path.getsize(checkpoint_file) / 1024
        checkpoint_time = datetime.datetime.fromtimestamp(
            os.path.getmtime(checkpoint_file)
        ).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n✅ Checkpoint criado em: {checkpoint_time}")
        print(f"✅ Tamanho do checkpoint: {checkpoint_size:.1f} KB")
        
        # Verifica conteúdo do checkpoint
        with open(solver.checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        print("\n📊 Conteúdo do checkpoint:")
        print(f"  - Geração: {checkpoint_data.get('generation', 'N/A')}")
        print(f"  - Melhor fitness: {checkpoint_data.get('best_fitness', 'N/A')}")
        print(f"  - População: {len(checkpoint_data.get('population', []))} indivíduos")
        
        return solver, checkpoint_data
    else:
        print("\n❌ Falha: Checkpoint não foi criado")
        return None, None

def test_checkpoint_restore(solver, checkpoint_data):
    """Testa a restauração de um checkpoint para continuar a execução"""
    if not solver or not checkpoint_data:
        print("\n❌ Sem dados de checkpoint para teste de restauração")
        return False
        
    print("\n🧪 TESTE DE RESTAURAÇÃO DE CHECKPOINT")
    print("=" * 50)
    
    # Salva alguns valores importantes do solver original
    original_generation = solver.generation
    original_best_fitness = solver.best_fitness
    original_keys_evaluated = solver.keys_evaluated
    
    # Cria uma nova instância do solver
    print("Criando nova instância do solver...")
    new_solver = GeneticBitcoinSolver(population_size=100)
    
    # Restaura o checkpoint
    print(f"Restaurando checkpoint: {solver.checkpoint_file}")
    restore_success = new_solver.load_checkpoint(solver.checkpoint_file)
    
    if restore_success:
        print("\n✅ Checkpoint restaurado com sucesso")
        
        # Verifica se os dados foram restaurados corretamente
        print("\n📊 Verificando restauração de dados:")
        print(f"  - Geração: {original_generation} → {new_solver.generation}")
        print(f"  - Melhor fitness: {original_best_fitness} → {new_solver.best_fitness}")
        print(f"  - Keys avaliadas: {original_keys_evaluated} → {new_solver.keys_evaluated}")
        
        # Executa uma geração adicional para confirmar continuidade
        print("\nExecutando geração adicional após restauração...")
        new_solver.evolve_generation()
        print(f"Nova geração: {new_solver.generation}")
        print(f"Novo melhor fitness: {new_solver.best_fitness:.2f}")
        
        return True
    else:
        print("\n❌ Falha na restauração do checkpoint")
        return False

def run_checkpoint_tests():
    """Executa todos os testes de checkpoint"""
    print("\n🔄 TESTES DE CHECKPOINT DO BITCOIN PUZZLE SOLVER")
    print("=" * 60)
    print(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Testa criação de checkpoint
    solver, checkpoint_data = test_checkpoint_creation(generations=3, population_size=100)
    
    # Testa restauração se a criação foi bem-sucedida
    if solver and checkpoint_data:
        test_checkpoint_restore(solver, checkpoint_data)
        
        # Limpa arquivo de checkpoint de teste
        if os.path.exists(solver.checkpoint_file):
            os.remove(solver.checkpoint_file)
            print(f"\n🧹 Arquivo de checkpoint de teste removido")
    
    print("\n✅ TESTES DE CHECKPOINT CONCLUÍDOS")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa funcionalidade de checkpoint")
    parser.add_argument('--keep-files', action='store_true', help="Não remove arquivos de checkpoint após o teste")
    args = parser.parse_args()
    
    run_checkpoint_tests()
