#!/usr/bin/env python3
"""
🎯 COORDENADOR MESTRE - BITCOIN PUZZLE 71
========================================

Sistema coordenado que executa múltiplas estratégias não ortodoxas:
1. Ultra Smart Solver (ML + Quantum + Heurísticas)
2. Blockchain Forensics (Análise de padrões)
3. Algoritmo Genético otimizado
4. Força bruta inteligente com GPU (se disponível)

Executa tudo em paralelo para maximizar as chances!
"""

import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import subprocess
import sys
import os
from typing import List, Dict, Optional, Tuple
import hashlib
import ecdsa

# Importa nossos módulos
try:
    from ultra_smart_solver import UltraSmartSolver
    from blockchain_forensics import BlockchainForensics
    from genetic_bitcoin_solver import GeneticBitcoinSolver
    from environment_detector import get_environment_detector, get_environment_config
except ImportError as e:
    print(f"⚠️  Erro ao importar módulos: {e}")
    print("Certifique-se que todos os arquivos estão no diretório src/")

class MasterCoordinator:
    """Coordenador mestre que executa todas as estratégias"""
    
    def __init__(self):
        self.target_address = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
        self.target_pubkey = "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4"
        self.min_key = 2**70
        self.max_key = 2**71 - 1
        
        # Configuração do ambiente
        self.env_detector = get_environment_detector()
        self.env_config = get_environment_config()
        
        # Estatísticas globais
        self.total_keys_tested = 0
        self.start_time = time.time()
        self.found_solution = False
        
        # Estado de checkpoint
        self.has_checkpoint = False
        self.load_coordinator_checkpoint()
        
        print("🎯 COORDENADOR MESTRE - BITCOIN PUZZLE 71")
        print("=" * 60)
        print(f"🏆 Target: {self.target_address}")
        print(f"🔑 Pubkey: {self.target_pubkey}")
        print(f"⚡ CPU cores: {self.env_config['max_workers']}")
        print(f"🧮 Range: 2^70 a 2^71-1")
        
        # Mostra status de checkpoint
        if self.has_checkpoint:
            print(f"🔄 Checkpoint anterior encontrado! Continuando execução anterior...")
        
        # Configura CUDA se disponível
        if self.env_detector.cuda_available:
            self.env_detector.setup_cuda_environment()
    
    def load_coordinator_checkpoint(self):
        """Carrega checkpoint do coordenador se existir"""
        try:
            if os.path.exists('master_progress.json'):
                with open('master_progress.json', 'r') as f:
                    data = json.load(f)
                
                self.total_keys_tested = data.get('total_keys_tested', 0)
                prev_runtime = data.get('runtime_seconds', 0)
                
                # Ajusta o start_time para refletir o tempo total de execução
                self.start_time = time.time() - prev_runtime
                
                self.has_checkpoint = True
                return True
        except Exception as e:
            print(f"⚠️ Erro ao carregar checkpoint do coordenador: {e}")
        
        return False
    
    def save_coordinator_checkpoint(self):
        """Salva checkpoint do coordenador"""
        runtime = time.time() - self.start_time
        
        data = {
            'total_keys_tested': self.total_keys_tested,
            'runtime_seconds': runtime,
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        try:
            with open('master_progress.json', 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar checkpoint do coordenador: {e}")
            return False
    
    def verify_solution(self, private_key: int) -> bool:
        """Verifica se uma chave privada é a solução"""
        try:
            if not (self.min_key <= private_key <= self.max_key):
                return False
            
            # Converte para ponto da curva elíptica
            private_key_hex = f"{private_key:064x}"
            sk = ecdsa.SigningKey.from_string(
                bytes.fromhex(private_key_hex),
                curve=ecdsa.SECP256k1
            )
            vk = sk.get_verifying_key()
            point = vk.pubkey.point
            
            # Converte para chave pública comprimida
            x = int(point.x())
            y = int(point.y())
            
            # Determina prefixo (02 ou 03)
            prefix = "03" if y % 2 == 1 else "02"
            pubkey = prefix + f"{x:064x}"
            
            return pubkey == self.target_pubkey
            
        except Exception as e:
            print(f"❌ Erro ao verificar chave {private_key:016x}: {e}")
            return False
    
    def run_ultra_smart_solver(self) -> Optional[int]:
        """Executa o Ultra Smart Solver"""
        print("\n🧠 Iniciando Ultra Smart Solver...")
        try:
            solver = UltraSmartSolver()
            result = solver.run_ultra_smart_search()
            return result
        except Exception as e:
            print(f"❌ Erro no Ultra Smart Solver: {e}")
            return None
    
    def run_blockchain_forensics(self) -> List[int]:
        """Executa análise forense da blockchain"""
        print("\n🕵️ Iniciando Blockchain Forensics...")
        try:
            forensics = BlockchainForensics()
            candidates = forensics.run_forensic_analysis()
            return candidates
        except Exception as e:
            print(f"❌ Erro no Blockchain Forensics: {e}")
            return []
    
    def run_genetic_algorithm(self) -> Optional[int]:
        """Executa algoritmo genético otimizado"""
        print("\n🧬 Iniciando Algoritmo Genético...")
        try:
            # Usa população otimizada para o ambiente
            population = self.env_config['genetic_population']
            solver = GeneticBitcoinSolver(population_size=population, elite_ratio=0.15)
            result = solver.run_evolution(max_generations=500, save_frequency=25)
            
            if hasattr(result, 'private_key'):
                return result.private_key
            return None
        except Exception as e:
            print(f"❌ Erro no Algoritmo Genético: {e}")
            return None
    
    def run_intelligent_bruteforce(self, start_range: int, end_range: int) -> Optional[int]:
        """Executa força bruta inteligente em um range específico"""
        print(f"\n💪 Força bruta inteligente: {start_range:016x} - {end_range:016x}")
        
        # Estratégias inteligentes para força bruta
        strategies = [
            'sequential',
            'random_jump',
            'fibonacci_step',
            'prime_hop',
            'bit_flip_walk'
        ]
        
        keys_per_strategy = (end_range - start_range) // len(strategies)
        
        for i, strategy in enumerate(strategies):
            strategy_start = start_range + i * keys_per_strategy
            strategy_end = strategy_start + keys_per_strategy
            
            if strategy == 'sequential':
                result = self._bruteforce_sequential(strategy_start, strategy_end)
            elif strategy == 'random_jump':
                result = self._bruteforce_random_jump(strategy_start, strategy_end)
            elif strategy == 'fibonacci_step':
                result = self._bruteforce_fibonacci(strategy_start, strategy_end)
            elif strategy == 'prime_hop':
                result = self._bruteforce_prime_hop(strategy_start, strategy_end)
            else:  # bit_flip_walk
                result = self._bruteforce_bit_flip(strategy_start, strategy_end)
            
            if result:
                return result
        
        return None
    
    def _bruteforce_sequential(self, start: int, end: int) -> Optional[int]:
        """Força bruta sequencial"""
        for key in range(start, min(end, start + 1000000)):  # Limita para não travar
            if self.verify_solution(key):
                return key
            self.total_keys_tested += 1
        return None
    
    def _bruteforce_random_jump(self, start: int, end: int) -> Optional[int]:
        """Força bruta com saltos aleatórios"""
        import random
        
        for _ in range(1000000):  # Número fixo de tentativas
            key = random.randint(start, end)
            if self.verify_solution(key):
                return key
            self.total_keys_tested += 1
        return None
    
    def _bruteforce_fibonacci(self, start: int, end: int) -> Optional[int]:
        """Força bruta usando sequência de Fibonacci"""
        fib_a, fib_b = 1, 1
        key = start
        
        for _ in range(1000000):
            if key > end:
                key = start + (key % (end - start))
            
            if self.verify_solution(key):
                return key
            
            # Próximo Fibonacci
            fib_a, fib_b = fib_b, fib_a + fib_b
            key += fib_b % 10000  # Salto baseado em Fibonacci
            self.total_keys_tested += 1
        
        return None
    
    def _bruteforce_prime_hop(self, start: int, end: int) -> Optional[int]:
        """Força bruta saltando por números primos"""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        
        key = start
        prime_idx = 0
        
        for _ in range(1000000):
            if key > end:
                key = start + (key % (end - start))
            
            if self.verify_solution(key):
                return key
            
            # Salta usando primo
            key += primes[prime_idx % len(primes)] * 1000
            prime_idx += 1
            self.total_keys_tested += 1
        
        return None
    
    def _bruteforce_bit_flip(self, start: int, end: int) -> Optional[int]:
        """Força bruta com bit flipping"""
        import random
        
        base_key = (start + end) // 2
        
        for _ in range(1000000):
            # Faz bit flip aleatório
            bit_pos = random.randint(0, 70)
            key = base_key ^ (1 << bit_pos)
            
            if start <= key <= end and self.verify_solution(key):
                return key
            self.total_keys_tested += 1
        
        return None
    
    def test_candidates_parallel(self, candidates: List[int]) -> Optional[int]:
        """Testa candidatos em paralelo"""
        print(f"\n🧪 Testando {len(candidates)} candidatos em paralelo...")
        
        # Usa configuração otimizada do ambiente
        max_workers = self.env_config['max_workers']
        chunk_size = max(1, len(candidates) // max_workers)
        chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self._test_chunk, chunk)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                if result:
                    return result
        
        return None
    
    def _test_chunk(self, chunk: List[int]) -> Optional[int]:
        """Testa um chunk de candidatos"""
        for candidate in chunk:
            if self.verify_solution(candidate):
                return candidate
            self.total_keys_tested += 1
        return None
    
    def save_progress(self, method: str, result: Optional[int] = None):
        """Salva progresso atual"""
        progress = {
            'timestamp': time.time(),
            'runtime': time.time() - self.start_time,
            'total_keys_tested': self.total_keys_tested,
            'keys_per_second': self.total_keys_tested / (time.time() - self.start_time),
            'last_method': method,
            'solution_found': result is not None,
            'solution': f"0x{result:016x}" if result else None
        }
        
        with open('master_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)
        
        if result:
            # Salva solução separadamente
            with open('SOLUCAO_PUZZLE_71.txt', 'w') as f:
                f.write(f"🎉 BITCOIN PUZZLE 71 RESOLVIDO! 🎉\n")
                f.write(f"=" * 40 + "\n")
                f.write(f"Chave privada (hex): 0x{result:016x}\n")
                f.write(f"Chave privada (decimal): {result}\n")
                f.write(f"Método: {method}\n")
                f.write(f"Timestamp: {time.time()}\n")
                f.write(f"Runtime: {time.time() - self.start_time:.2f} segundos\n")
                f.write(f"Total de chaves testadas: {self.total_keys_tested:,}\n")
    
    def run_coordinated_attack(self):
        """Executa ataque coordenado com todas as estratégias"""
        print("\n🚀 INICIANDO ATAQUE COORDENADO")
        print("=" * 60)
        print("⚡ Executando múltiplas estratégias em paralelo...")
        
        # Se temos checkpoint, avisa que está continuando de onde parou
        if self.has_checkpoint:
            print(f"🔄 Continuando execução a partir de checkpoint salvo")
            print(f"🕒 Tempo já executado: {(time.time() - self.start_time):.1f} segundos")
            print(f"🔑 Chaves já testadas: {self.total_keys_tested:,}")
        
        # 1. Executa todas as estratégias em paralelo
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Algoritmo Genético - usa checkpoint se disponível
            print("🧬 Iniciando Algoritmo Genético Otimizado...")
            solver = GeneticBitcoinSolver(population_size=1000, elite_ratio=0.15)
            future_genetic = executor.submit(solver.run_evolution, max_generations=500, save_frequency=5)
            futures.append(("Genetic", future_genetic))
            
            # Análise forense de blockchain
            print("🔍 Iniciando Blockchain Forensics...")
            forensics = BlockchainForensics()
            future_forensic = executor.submit(forensics.analyze_puzzle_71)
            futures.append(("Forensics", future_forensic))
            
            # Ultra Smart Solver
            print("🧠 Iniciando Ultra Smart Solver...")
            future_ultra = executor.submit(self.run_ultra_smart_solver)
            futures.append(("UltraSmart", future_ultra))
            
            # Força bruta como último recurso (paralela)
            cpu_count = self.env_config.get('max_workers', mp.cpu_count())
            
            # Calcula tamanho estimado do espaço
            range_size = self.max_key - self.min_key + 1
            num_workers = max(1, min(cpu_count // 2, 4))  # Limitado a 4 workers
            
            print(f"💪 Iniciando Força Bruta Inteligente (×{num_workers})...")
            
            # Divide range em chunks para força bruta
            chunk_size = range_size // (num_workers * 100)  # 100 chunks por worker
            futures_brute = []
            
            for i in range(num_workers):
                start = self.min_key + (i * chunk_size * 100)
                end = min(start + (chunk_size * 100), self.max_key)
                future_brute = executor.submit(self.run_intelligent_bruteforce, start, end)
                futures.append((f"Bruteforce_{i}", future_brute))
            
            # Timeout baseado na configuração do ambiente
            timeout = None
            if self.env_config.get('timeout_hours'):
                timeout = self.env_config['timeout_hours'] * 3600
            else:
                timeout = 3600  # 1 hora padrão
            
            # Variável para controlar salvamento de checkpoint periódico
            last_checkpoint_time = time.time()
            checkpoint_interval = 300  # Salvar a cada 5 minutos
            
            # Monitora execução com salvamento periódico de checkpoint
            running_futures = list(futures)
            while running_futures:
                # Verifica se está na hora de salvar checkpoint
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval:
                    self.save_coordinator_checkpoint()
                    last_checkpoint_time = current_time
                    print(f"\n💾 Checkpoint salvo! Chaves testadas: {self.total_keys_tested:,}")
                
                # Verifica cada estratégia
                completed = []
                for method, future in running_futures:
                    if future.done():
                        completed.append((method, future))
                        try:
                            result = future.result()
                            if result:
                                print(f"\n🎉 SOLUÇÃO ENCONTRADA COM {method.upper()}!")
                                print(f"🔑 Chave: 0x{result:016x}")
                                self.save_progress(method, result)
                                return result
                        except Exception as e:
                            print(f"⚠️  {method} falhou: {e}")
                
                # Remove estratégias completadas
                for item in completed:
                    running_futures.remove(item)
                
                # Pequena pausa para não sobrecarregar CPU
                if running_futures:
                    time.sleep(1)
        
        # 4. Se nada funcionou, executa força bruta estendida
        print("\n💪 Iniciando força bruta estendida...")
        
        # Divide range em chunks menores para execução paralela
        num_processes = mp.cpu_count()
        chunk_size = range_size // (num_processes * 10)  # Chunks menores
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            
            for i in range(num_processes * 2):  # 2x mais chunks que processos
                start = self.min_key + i * chunk_size
                end = min(start + chunk_size, self.max_key)
                
                future = executor.submit(self.run_intelligent_bruteforce, start, end)
                futures.append(future)
            
            # Monitora força bruta com salvamento periódico
            last_checkpoint_time = time.time()
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=1800)  # 30 min timeout por chunk
                    if result:
                        print(f"\n🎉 SOLUÇÃO ENCONTRADA COM FORÇA BRUTA!")
                        print(f"🔑 Chave: 0x{result:016x}")
                        self.save_progress("Intelligent Bruteforce", result)
                        return result
                    
                    # Verifica se está na hora de salvar checkpoint
                    current_time = time.time()
                    if current_time - last_checkpoint_time >= checkpoint_interval:
                        self.save_coordinator_checkpoint()
                        last_checkpoint_time = current_time
                        print(f"\n💾 Checkpoint salvo! Chaves testadas: {self.total_keys_tested:,}")
                        
                except Exception as e:
                    print(f"⚠️  Chunk {i} falhou: {e}")
        
        # Salva progresso final
        runtime = time.time() - self.start_time
        print(f"\n⏰ Execução concluída!")
        print(f"🕒 Runtime: {runtime:.2f} segundos")
        print(f"🔑 Chaves testadas: {self.total_keys_tested:,}")
        print(f"⚡ Velocidade: {self.total_keys_tested / runtime:.0f} chaves/segundo")
        
        self.save_coordinator_checkpoint()
        self.save_progress("Execution Complete", None)
        
        return None
    
    def continuous_search(self, hours: int = 24):
        """Executa busca contínua por X horas"""
        print(f"\n🔄 BUSCA CONTÍNUA POR {hours} HORAS")
        print("=" * 50)
        
        end_time = time.time() + (hours * 3600)
        iteration = 0
        
        while time.time() < end_time and not self.found_solution:
            iteration += 1
            print(f"\n🔄 Iteração {iteration}")
            
            result = self.run_coordinated_attack()
            if result:
                self.found_solution = True
                return result
            
            # Pausa entre iterações
            remaining_time = end_time - time.time()
            if remaining_time > 300:  # Se sobrou mais de 5 minutos
                print("⏸️  Pausa de 60 segundos antes da próxima iteração...")
                time.sleep(60)
        
        print(f"\n⏰ Busca contínua finalizada após {iteration} iterações")
        return None
    
    def run(self):
        """Método principal de execução - compatível com main.py"""
        return main()

def main():
    """Função principal"""
    print("🎯 SISTEMA COORDENADO PARA BITCOIN PUZZLE 71")
    print("=" * 60)
    print("⚠️  ATENÇÃO: Este é um desafio matemático extremamente difícil!")
    print("🧠 Usando métodos não ortodoxos e coordenação inteligente...")
    print("🚀 Múltiplas estratégias serão executadas em paralelo!")
    print("")
    
    coordinator = MasterCoordinator()
    
    # Opções de execução
    print("📋 OPÇÕES DE EXECUÇÃO:")
    print("1. Ataque coordenado único")
    print("2. Busca contínua (24 horas)")
    print("3. Busca contínua personalizada")
    
    try:
        choice = input("\nEscolha uma opção (1-3): ").strip()
        
        if choice == "1":
            result = coordinator.run_coordinated_attack()
        elif choice == "2":
            result = coordinator.continuous_search(24)
        elif choice == "3":
            hours = int(input("Quantas horas? "))
            result = coordinator.continuous_search(hours)
        else:
            print("❌ Opção inválida, executando ataque único...")
            result = coordinator.run_coordinated_attack()
        
        if result:
            print(f"\n🏆 MISSÃO CUMPRIDA!")
            print(f"💰 Bitcoin Puzzle 71 resolvido!")
            print(f"🔑 Chave privada: 0x{result:016x}")
            print(f"🔢 Decimal: {result}")
        else:
            print(f"\n🔄 Execução finalizada sem encontrar a solução")
            print(f"💡 Execute novamente - cada execução usa estratégias diferentes!")
            print(f"📊 Progresso salvo em 'master_progress.json'")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Execução interrompida pelo usuário")
        coordinator.save_progress("User Interrupted", None)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        coordinator.save_progress("Error", None)

if __name__ == "__main__":
    main()
