#!/usr/bin/env python3
"""
🧬 GENETIC BITCOIN SOLVER - Estratégia Ultra-Eficiente
=====================================================

Algoritmo Genético Avançado para Bitcoin Puzzle 71 usando:
- Coordenadas X,Y da curva elíptica SECP256k1
- Análise de entropia para fitness
- Evolução adaptativa com múltiplas estratégias
- Otimização por crossover e mutação inteligentes
"""

import random
import numpy as np
import hashlib
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import ecdsa
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import os
import math
from key_saver import save_discovered_key

@dataclass
class Individual:
    """Representa um indivíduo no algoritmo genético"""
    private_key: int
    public_key_x: int
    public_key_y: int
    fitness: float
    entropy: float
    generation: int

class GeneticBitcoinSolver:
    """Solver ultra-eficiente baseado em algoritmo genético"""
    
    def __init__(self, population_size=1000, elite_ratio=0.1):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.elite_size = int(population_size * elite_ratio)
        
        # Bitcoin Puzzle 71 configuração
        self.target_pubkey = "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4"
        self.target_x, self.target_y = self.decode_public_key(self.target_pubkey)
        
        # Range do Puzzle 71
        self.min_key = 2**70
        self.max_key = 2**71 - 1
        
        # Configurações evolutivas
        self.mutation_rate = 0.02
        self.crossover_rate = 0.8
        self.generation = 0
        self.best_fitness = float('inf')
        
        # Estatísticas
        self.keys_evaluated = 0
        self.start_time = time.time()
        
        print("🧬 GENETIC BITCOIN SOLVER ULTRA-EFICIENTE")
        print("=" * 60)
        print(f"🎯 Target: {self.target_pubkey}")
        print(f"📊 População: {population_size}")
        print(f"🏆 Elite: {self.elite_size} ({elite_ratio*100:.1f}%)")
        print(f"🎲 Mutação: {self.mutation_rate*100:.1f}%")
        print(f"💑 Crossover: {self.crossover_rate*100:.1f}%")
        
    def decode_public_key(self, pubkey_hex: str) -> Tuple[int, int]:
        """Decodifica chave pública para coordenadas X,Y"""
        try:
            # Remove prefixo '03' ou '02' para chaves comprimidas
            if pubkey_hex.startswith(('02', '03')):
                x = int(pubkey_hex[2:], 16)
                # Calcula Y usando a curva secp256k1
                p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
                y_squared = (x**3 + 7) % p
                y = pow(y_squared, (p + 1) // 4, p)
                
                # Ajusta paridade se necessário
                if pubkey_hex.startswith('03') and y % 2 == 0:
                    y = p - y
                elif pubkey_hex.startswith('02') and y % 2 == 1:
                    y = p - y
                    
                return x, y
            else:
                # Chave não comprimida
                x = int(pubkey_hex[2:66], 16)
                y = int(pubkey_hex[66:], 16)
                return x, y
        except Exception as e:
            print(f"❌ Erro ao decodificar chave pública: {e}")
            return 0, 0
    
    def private_key_to_public(self, private_key: int) -> Tuple[int, int]:
        """Converte chave privada para coordenadas públicas X,Y"""
        try:
            private_key_hex = f"{private_key:064x}"
            sk = ecdsa.SigningKey.from_string(
                bytes.fromhex(private_key_hex), 
                curve=ecdsa.SECP256k1
            )
            vk = sk.get_verifying_key()
            point = vk.pubkey.point
            return int(point.x()), int(point.y())
        except:
            return 0, 0
    
    def calculate_entropy(self, x: int, y: int) -> float:
        """Calcula entropia das coordenadas como medida de aleatoriedade"""
        # Combina X e Y para análise de entropia
        combined = f"{x:064x}{y:064x}"
        
        # Análise de frequência de bits
        bit_string = ''.join(format(ord(c), '04b') for c in combined)
        
        # Calcula entropia de Shannon
        if len(bit_string) == 0:
            return 0.0
            
        # Conta frequências de 0s e 1s
        freq_0 = bit_string.count('0')
        freq_1 = bit_string.count('1')
        total = len(bit_string)
        
        if freq_0 == 0 or freq_1 == 0:
            return 0.0
            
        p0 = freq_0 / total
        p1 = freq_1 / total
        
        entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        
        # Adiciona análise de padrões
        pattern_score = self.analyze_patterns(combined)
        
        return entropy + pattern_score
    
    def analyze_patterns(self, hex_string: str) -> float:
        """Analisa padrões específicos que podem indicar proximidade"""
        score = 0.0
        
        # Penaliza sequências repetitivas
        for i in range(len(hex_string) - 3):
            if hex_string[i] == hex_string[i+1] == hex_string[i+2]:
                score -= 0.1
        
        # Premia distribuição equilibrada de dígitos hex
        digit_counts = {}
        for digit in hex_string:
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        # Entropia de distribuição dos dígitos
        total_digits = len(hex_string)
        digit_entropy = 0.0
        for count in digit_counts.values():
            if count > 0:
                p = count / total_digits
                digit_entropy -= p * np.log2(p)
        
        # Normaliza para 0-1
        max_digit_entropy = np.log2(16)  # 16 dígitos hex possíveis
        normalized_entropy = digit_entropy / max_digit_entropy
        
        return score + normalized_entropy
    
    def calculate_fitness(self, individual: Individual) -> float:
        """Calcula fitness baseado na distância euclidiana e entropia"""
        # Distância euclidiana nas coordenadas da curva
        dx = abs(individual.public_key_x - self.target_x)
        dy = abs(individual.public_key_y - self.target_y)
        
        euclidean_distance = math.sqrt(dx*dx + dy*dy)
        
        # Combina distância com entropia
        # Fitness menor = melhor
        entropy_bonus = individual.entropy * 0.1
        fitness = euclidean_distance - entropy_bonus
        
        # Bonus adicional para proximidade extrema
        if euclidean_distance < 1000:
            fitness *= 0.5
        
        return fitness
    
    def create_individual(self, private_key: Optional[int] = None) -> Individual:
        """Cria um indivíduo aleatório ou com chave específica"""
        if private_key is None:
            private_key = random.randint(self.min_key, self.max_key)
        
        # Garante que a chave está no range válido
        private_key = max(self.min_key, min(self.max_key, private_key))
        
        x, y = self.private_key_to_public(private_key)
        entropy = self.calculate_entropy(x, y)
        
        individual = Individual(
            private_key=private_key,
            public_key_x=x,
            public_key_y=y,
            fitness=0.0,
            entropy=entropy,
            generation=self.generation
        )
        
        individual.fitness = self.calculate_fitness(individual)
        self.keys_evaluated += 1
        
        return individual
    
    def initialize_population(self) -> List[Individual]:
        """Inicializa população com estratégias diversas"""
        population = []
        
        # 50% - Totalmente aleatório
        for _ in range(self.population_size // 2):
            population.append(self.create_individual())
        
        # 25% - Próximo ao meio do range
        middle = (self.min_key + self.max_key) // 2
        for _ in range(self.population_size // 4):
            offset = random.randint(-2**30, 2**30)
            key = middle + offset
            population.append(self.create_individual(key))
        
        # 15% - Baseado em padrões de entropia alta
        for _ in range(int(self.population_size * 0.15)):
            # Gera chaves com padrões específicos
            base_key = random.randint(self.min_key, self.max_key)
            # Modifica para ter alta entropia
            key = base_key ^ random.randint(0, 2**32)
            population.append(self.create_individual(key))
        
        # 10% - Estratégia de gradiente
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            # Usa estratégia de gradiente baseada em coordenadas conhecidas
            gradient_key = self.generate_gradient_key()
            population.append(self.create_individual(gradient_key))
        
        # Ordena por fitness
        population.sort(key=lambda x: x.fitness)
        
        print(f"🧬 População inicial criada: {len(population)} indivíduos")
        print(f"🏆 Melhor fitness inicial: {population[0].fitness:.2f}")
        
        return population
    
    def generate_gradient_key(self) -> int:
        """Gera chave usando gradiente em direção ao target"""
        # Estratégia simplificada: usar bits do target como guia
        target_bits = format(self.target_x, '0256b')
        
        # Cria chave com alguns bits similares
        key_bits = ''
        for i, bit in enumerate(target_bits):
            if random.random() < 0.6:  # 60% chance de manter bit similar
                key_bits += bit
            else:
                key_bits += str(random.randint(0, 1))
        
        key = int(key_bits, 2)
        return max(self.min_key, min(self.max_key, key))
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover inteligente baseado em bits de chaves privadas"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Escolhe um tipo de crossover aleatoriamente
        crossover_type = random.choice(['multi_point', 'uniform', 'arithmetic', 'differential'])
        
        if crossover_type == 'multi_point':
            # Converte chaves para representação binária
            bits1 = format(parent1.private_key, '071b')  # 71 bits para puzzle 71
            bits2 = format(parent2.private_key, '071b')
            
            # Múltiplos pontos de crossover (3 pontos para maior diversidade)
            crossover_points = sorted(random.sample(range(1, 70), 3))
            
            # Cria filhos
            parts1 = [bits1[:crossover_points[0]], 
                     bits1[crossover_points[0]:crossover_points[1]], 
                     bits1[crossover_points[1]:crossover_points[2]], 
                     bits1[crossover_points[2]:]]
            
            parts2 = [bits2[:crossover_points[0]], 
                     bits2[crossover_points[0]:crossover_points[1]], 
                     bits2[crossover_points[1]:crossover_points[2]], 
                     bits2[crossover_points[2]:]]
                     
            # Alterna partes para criar filhos
            child1_bits = parts1[0] + parts2[1] + parts1[2] + parts2[3]
            child2_bits = parts2[0] + parts1[1] + parts2[2] + parts1[3]
            
            child1_key = int(child1_bits, 2)
            child2_key = int(child2_bits, 2)
            
        elif crossover_type == 'uniform':
            # Crossover uniforme (bit por bit)
            bits1 = format(parent1.private_key, '071b')
            bits2 = format(parent2.private_key, '071b')
            
            child1_bits = ''.join(b1 if random.random() < 0.5 else b2 
                                for b1, b2 in zip(bits1, bits2))
                                
            child2_bits = ''.join(b2 if random.random() < 0.5 else b1 
                                for b1, b2 in zip(bits1, bits2))
                                
            child1_key = int(child1_bits, 2)
            child2_key = int(child2_bits, 2)
            
        elif crossover_type == 'arithmetic':
            # Crossover aritmético (médias ponderadas)
            weight = random.random()
            child1_key = int(parent1.private_key * weight + parent2.private_key * (1 - weight))
            child2_key = int(parent2.private_key * weight + parent1.private_key * (1 - weight))
            
        else:  # differential
            # Crossover diferencial (soma/subtração de diferenças)
            diff = abs(parent1.private_key - parent2.private_key)
            child1_key = min(self.max_key, max(self.min_key, parent1.private_key + diff // 3))
            child2_key = min(self.max_key, max(self.min_key, parent2.private_key - diff // 4))
        
        # Garante que as chaves estejam dentro do intervalo válido
        child1_key = max(self.min_key, min(self.max_key, child1_key))
        child2_key = max(self.min_key, min(self.max_key, child2_key))
        
        return self.create_individual(child1_key), self.create_individual(child2_key)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutação adaptativa baseada em fitness e entropia"""
        if random.random() > self.mutation_rate:
            return individual
        
        key = individual.private_key
        
        # Múltiplos tipos de mutação com probabilidades diferentes
        mutation_types = [
            ('bit_flip', 0.3),
            ('multi_bit_flip', 0.15),
            ('arithmetic', 0.25),
            ('entropy_guided', 0.2),
            ('gaussian', 0.1)  # Nova estratégia
        ]
        
        # Seleciona tipo de mutação com base nas probabilidades
        mutation_choices = [t[0] for t in mutation_types]
        mutation_weights = [t[1] for t in mutation_types]
        mutation_type = random.choices(
            mutation_choices, 
            weights=mutation_weights, 
            k=1
        )[0]
        
        if mutation_type == 'bit_flip':
            # Flip de bit aleatório
            bit_pos = random.randint(0, 70)
            key ^= (1 << bit_pos)
            
        elif mutation_type == 'multi_bit_flip':
            # Flip múltiplos bits (2-5 bits)
            num_bits = random.randint(2, 5)
            for _ in range(num_bits):
                bit_pos = random.randint(0, 70)
                key ^= (1 << bit_pos)
            
        elif mutation_type == 'arithmetic':
            # Mutação aritmética
            op = random.choice(['+', '-', '^', '*', '/'])
            
            if op == '+':
                key += random.randint(1, 2**25)
            elif op == '-':
                key -= random.randint(1, 2**25)
            elif op == '^':
                key ^= random.randint(1, 2**35)
            elif op == '*':
                factor = 1 + random.random() * 0.01  # Multiplicar por 1.00-1.01
                key = int(key * factor)
            elif op == '/':
                divisor = 1 + random.random() * 0.01  # Dividir por 1.00-1.01
                key = int(key / divisor)
                
        elif mutation_type == 'entropy_guided':
            # Mutação guiada pela entropia
            if individual.entropy < 0.5:  # Baixa entropia, mutação mais agressiva
                key ^= random.randint(1, 2**40)
                # Também adiciona uma perturbação aritmética
                key += random.randint(-2**30, 2**30)
            else:  # Alta entropia, mutação mais sutil
                key ^= random.randint(1, 2**25)
                
        elif mutation_type == 'gaussian':
            # Mutação gaussiana (mais próxima da chave original)
            # Usa desvio padrão proporcional ao range
            std_dev = (self.max_key - self.min_key) * 0.00001  # 0.001% do range
            delta = int(random.gauss(0, std_dev))
            key = key + delta
        
        # Garante range válido
        key = max(self.min_key, min(self.max_key, key))
        
        return self.create_individual(key)
    
    def selection(self, population: List[Individual]) -> List[Individual]:
        """Seleção por torneio com elitismo"""
        new_population = []
        
        # Elitismo - mantém os melhores
        elite = population[:self.elite_size]
        new_population.extend(elite)
        
        # Seleção por torneio para o resto
        tournament_size = 5
        while len(new_population) < self.population_size:
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = min(tournament, key=lambda x: x.fitness)
            new_population.append(winner)
        
        return new_population[:self.population_size]
    
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Evolui uma geração completa"""
        self.generation += 1
        
        # Seleção
        selected = self.selection(population)
        
        # Verificar diversidade
        unique_keys = set(ind.private_key for ind in selected)
        if len(unique_keys) < 10:  # Diversidade muito baixa
            # Injetar novos indivíduos aleatórios para aumentar diversidade
            self.mutation_rate = min(0.2, self.mutation_rate * 1.5)  # Aumentar mutação
            
            # Adicionar 30% de indivíduos completamente novos
            num_new = int(self.population_size * 0.3)
            for _ in range(num_new):
                new_key = random.randint(self.min_key, self.max_key)
                selected.append(self.create_individual(new_key))
        
        # Crossover e mutação
        new_population = []
        new_population.extend(selected[:self.elite_size])  # Preserva elite
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Ordena por fitness
        new_population = new_population[:self.population_size]
        new_population.sort(key=lambda x: x.fitness)
        
        # Atualiza melhor fitness
        current_best = new_population[0].fitness
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            print(f"🏆 Nova melhor fitness: {current_best:.2f} (Geração {self.generation})")
        
        return new_population
    
    def check_solution(self, population: List[Individual]) -> Optional[Individual]:
        """Verifica se algum indivíduo é a solução"""
        for individual in population:
            if (individual.public_key_x == self.target_x and 
                individual.public_key_y == self.target_y):
                return individual
        return None
    
    def print_statistics(self, population: List[Individual]):
        """Imprime estatísticas da geração atual"""
        runtime = time.time() - self.start_time
        keys_per_second = self.keys_evaluated / runtime if runtime > 0 else 0
        
        best = population[0]
        worst = population[-1]
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        avg_entropy = sum(ind.entropy for ind in population) / len(population)
        
        print(f"\n📊 GERAÇÃO {self.generation} - ESTATÍSTICAS")
        print("=" * 50)
        print(f"⚡ Velocidade: {keys_per_second:,.0f} chaves/segundo")
        print(f"🔑 Chaves avaliadas: {self.keys_evaluated:,}")
        print(f"⏱️  Runtime: {runtime:.1f}s")
        print(f"🏆 Melhor fitness: {best.fitness:.2f}")
        print(f"📉 Pior fitness: {worst.fitness:.2f}")
        print(f"📊 Fitness médio: {avg_fitness:.2f}")
        print(f"🎲 Entropia média: {avg_entropy:.4f}")
        print(f"🧬 Taxa de mutação: {self.mutation_rate*100:.1f}%")
        
        # Mostra os 3 melhores
        print(f"\n🏅 TOP 3 INDIVÍDUOS:")
        for i, ind in enumerate(population[:3]):
            print(f"   {i+1}. Fitness: {ind.fitness:.2f} | "
                  f"Entropia: {ind.entropy:.4f} | "
                  f"Chave: 0x{ind.private_key:016x}")
    
    def save_checkpoint(self, population: List[Individual]):
        """Salva checkpoint da evolução"""
        checkpoint_data = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'keys_evaluated': self.keys_evaluated,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'population': [
                {
                    'private_key': ind.private_key,
                    'public_key_x': ind.public_key_x,
                    'public_key_y': ind.public_key_y,
                    'fitness': ind.fitness,
                    'entropy': ind.entropy,
                    'generation': ind.generation
                }
                for ind in population[:10]  # Salva apenas os 10 melhores
            ]
        }
        
        with open('genetic_checkpoint.json', 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Checkpoint salvo (Geração {self.generation})")
    
    def load_checkpoint(self) -> Tuple[int, List[Individual]]:
        """Carrega checkpoint se disponível"""
        try:
            if os.path.exists('genetic_checkpoint.json'):
                print("🔄 Checkpoint encontrado! Carregando progresso anterior...")
                with open('genetic_checkpoint.json', 'r') as f:
                    data = json.load(f)
                
                # Restaura estado do solver
                self.generation = data['generation']
                self.best_fitness = data['best_fitness']
                self.keys_evaluated = data['keys_evaluated']
                self.mutation_rate = data['mutation_rate']
                
                # Restaura população
                population = []
                for ind_data in data['population']:
                    private_key = ind_data['private_key']
                    
                    # Recria o indivíduo com a mesma chave privada
                    individual = self.create_individual(private_key)
                    
                    # Preenche a população inicial com este indivíduo várias vezes
                    # para manter o melhor fitness enquanto adiciona diversidade
                    population_chunk_size = self.population_size // len(data['population'])
                    for _ in range(population_chunk_size):
                        # Adiciona variação para cada clone para aumentar diversidade
                        if random.random() < 0.7:  # 70% de variação
                            # Cria variação da chave
                            variation = private_key
                            bit_pos = random.randint(0, 70)
                            variation ^= (1 << bit_pos)  # Flipa um bit
                            varied_ind = self.create_individual(variation)
                            population.append(varied_ind)
                        else:
                            # Mantém alguns iguais para preservar boas soluções
                            population.append(individual)
                
                # Completa a população se necessário
                while len(population) < self.population_size:
                    population.append(self.create_individual())
                
                # Ordena por fitness
                population.sort(key=lambda x: x.fitness)
                
                print(f"✅ Checkpoint carregado da geração {self.generation}")
                print(f"🏆 Melhor fitness: {self.best_fitness:.2f}")
                print(f"🔑 Total de chaves avaliadas: {self.keys_evaluated:,}")
                print(f"🧬 Taxa de mutação: {self.mutation_rate*100:.1f}%")
                
                return self.generation, population
            
        except Exception as e:
            print(f"⚠️  Erro ao carregar checkpoint: {e}")
            print("🔄 Iniciando do zero...")
        
        return 0, []
    
    def run_evolution(self, max_generations=1000, save_frequency=10):
        """Executa o algoritmo genético"""
        print("\n🧬 INICIANDO EVOLUÇÃO GENÉTICA")
        print("=" * 60)
        
        # Verifica se existe checkpoint para continuar
        start_gen, checkpoint_population = self.load_checkpoint()
        
        # Se encontrou checkpoint, continua dele, senão inicializa nova população
        if start_gen > 0 and checkpoint_population:
            population = checkpoint_population
            self.start_time = time.time() - (self.keys_evaluated / 1000)  # Estimativa de tempo já executado
        else:
            # Inicializa população
            population = self.initialize_population()
        
        try:
            for generation in range(start_gen, max_generations):
                # Verifica solução
                solution = self.check_solution(population)
                if solution:
                    print(f"\n🎉🎉🎉 SOLUÇÃO ENCONTRADA! 🎉🎉🎉")
                    print(f"🔑 Chave privada: 0x{solution.private_key:016x}")
                    print(f"📊 Fitness: {solution.fitness}")
                    print(f"🎲 Entropia: {solution.entropy}")
                    print(f"🧬 Geração: {solution.generation}")
                    
                    # Salvar chave encontrada usando o sistema robusto
                    try:
                        puzzle_num = 71  # Assumindo que estamos trabalhando no puzzle 71
                        target_address = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"  # Endereço do puzzle 71
                        success = save_discovered_key(
                            puzzle_number=puzzle_num,
                            private_key=f"{solution.private_key:016x}",
                            address=target_address,
                            solver_name="GeneticBitcoinSolver",
                            fitness=solution.fitness,
                            entropy=solution.entropy,
                            generation=solution.generation,
                            execution_time=time.time() - self.start_time,
                            keys_evaluated=self.keys_evaluated
                        )
                        if success:
                            print(f"💾 Chave salva com sucesso no sistema de backup!")
                        else:
                            print(f"⚠️ Falha ao salvar - criando backup de emergência")
                    except Exception as e:
                        print(f"❌ Erro no salvamento: {e}")
                    
                    # Salva resultado no arquivo tradicional também
                    with open('SOLUCAO_ENCONTRADA.txt', 'w') as f:
                        f.write(f"BITCOIN PUZZLE 71 RESOLVIDO!\n")
                        f.write(f"Chave privada: 0x{solution.private_key:016x}\n")
                        f.write(f"Chave privada (decimal): {solution.private_key}\n")
                        f.write(f"Fitness: {solution.fitness}\n")
                        f.write(f"Entropia: {solution.entropy}\n")
                        f.write(f"Geração: {solution.generation}\n")
                    
                    return solution
                
                # Evolui população
                population = self.evolve_generation(population)
                
                # Imprime estatísticas
                if generation % 5 == 0:
                    self.print_statistics(population)
                
                # Salva checkpoint
                if generation % save_frequency == 0:
                    self.save_checkpoint(population)
                
                # Adaptação dinâmica da taxa de mutação
                if generation % 50 == 0 and generation > 0:
                    # Aumenta mutação se estagnou
                    if self.best_fitness == population[0].fitness:
                        self.mutation_rate = min(0.1, self.mutation_rate * 1.2)
                        print(f"📈 Taxa de mutação aumentada para {self.mutation_rate*100:.1f}%")
        
        except KeyboardInterrupt:
            print("\n⏹️  Evolução interrompida pelo usuário")
            self.save_checkpoint(population)
            return None
        
        print(f"\n🏁 Evolução concluída após {max_generations} gerações")
        self.print_statistics(population)
        return population[0]  # Retorna o melhor indivíduo

def main():
    """Função principal"""
    solver = GeneticBitcoinSolver(population_size=500, elite_ratio=0.1)
    result = solver.run_evolution(max_generations=1000, save_frequency=20)
    
    if result and hasattr(result, 'private_key'):
        print("\n🎊 ALGORITMO GENÉTICO CONCLUÍDO!")
        print(f"🏆 Melhor resultado encontrado:")
        print(f"   🔑 Chave: 0x{result.private_key:016x}")
        print(f"   📊 Fitness: {result.fitness}")

if __name__ == "__main__":
    main()
