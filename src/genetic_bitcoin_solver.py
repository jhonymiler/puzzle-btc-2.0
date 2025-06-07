#!/usr/bin/env python3
"""
🧬 GENETIC BITCOIN SOLVER - Estratégia Ultra-Eficiente
=====================================================

Algoritmo Genético Avançado para Bitcoin Puzzle 71 usando:
- Coordenadas X,Y da curva elíptica SECP256k1
- Análise de entropia para fitness
- Evolução adaptativa com múltiplas estratégias
- Otimização por crossover e mutação inteligentes
- Aceleração GPU (CUDA/ROCm/MPS) quando disponível
- Paralelismo massivo adaptado ao ambiente
- Técnicas de exploração adaptativas
- Inferência bayesiana e amostragem Monte Carlo
"""

import random
import numpy as np
import hashlib
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ecdsa
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union
import json
import os
import math
from .key_saver import save_discovered_key

# Importações para estratégias avançadas e detecção de ambiente
from .advanced_genetic_strategies import AdvancedGeneticStrategies, get_advanced_strategies
from .environment_detector import EnvironmentDetector, get_environment_detector

# Tenta importar bibliotecas de aceleração como opcionais
TORCH_AVAILABLE = False
CUPY_AVAILABLE = False

# Verificamos a disponibilidade de bibliotecas sem importá-las diretamente
# para evitar erros de importação em ambientes sem essas bibliotecas
def check_optional_imports():
    global TORCH_AVAILABLE, CUPY_AVAILABLE
    
    try:
        import torch
        TORCH_AVAILABLE = True
        print("✅ PyTorch disponível para aceleração GPU")
    except ImportError:
        TORCH_AVAILABLE = False
    
    try:
        import cupy
        CUPY_AVAILABLE = True
        print("✅ CuPy disponível para aceleração GPU")
    except ImportError:
        CUPY_AVAILABLE = False
        
# Verificar bibliotecas opcionais
check_optional_imports()

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
    
    def __init__(self, population_size=1000, elite_ratio=0.1, config=None):
        # Configuração inicial
        self.config = config or {}
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
        self.mutation_rate = self.config.get('initial_mutation_rate', 0.02)
        self.crossover_rate = self.config.get('initial_crossover_rate', 0.8)
        self.generation = 0
        self.best_fitness = float('inf')
        
        # Estatísticas
        self.keys_evaluated = 0
        self.start_time = time.time()
        
        # Inicializar detector de ambiente
        self.env_detector = get_environment_detector()
        self.env_config = self.env_detector.config
        
        # Inicializa kernels GPU se disponíveis
        self.gpu_available = (self.env_detector.cuda_available or 
                             self.env_detector.rocm_available or 
                             self.env_detector.mps_available)
        
        self.gpu_kernels = None
        if self.gpu_available:
            self._integrate_gpu_kernels()
        
        # Configurar aceleração GPU se disponível
        self.gpu_available = (self.env_detector.cuda_available or 
                             self.env_detector.rocm_available or 
                             self.env_detector.mps_available)
                             
        if self.gpu_available:
            self.env_detector.setup_cuda_environment()
        
        # Inicializar estratégias genéticas avançadas
        self.advanced_strategies = AdvancedGeneticStrategies(
            self.min_key, 
            self.max_key,
            self.config
        )
        
        # Configurar paralelismo baseado no ambiente
        self.parallel_config = self.env_detector.optimize_search_parallelism(
            self.min_key, self.max_key
        )
        
        # Otimizar parâmetros baseados no ambiente
        self._optimize_parameters()
        
        print("🧬 GENETIC BITCOIN SOLVER ULTRA-EFICIENTE")
        print("=" * 60)
        print(f"🎯 Target: {self.target_pubkey}")
        print(f"📊 População: {population_size}")
        print(f"🏆 Elite: {self.elite_size} ({elite_ratio*100:.1f}%)")
        print(f"🎲 Mutação: {self.mutation_rate*100:.1f}%")
        print(f"💑 Crossover: {self.crossover_rate*100:.1f}%")
        
        if self.gpu_available:
            gpu_type = "NVIDIA CUDA" if self.env_detector.cuda_available else \
                       "AMD ROCm" if self.env_detector.rocm_available else \
                       "Apple MPS"
            print(f"🚀 Aceleração GPU: {gpu_type}")
            print(f"📈 Paralelismo: {self.parallel_config['max_workers']} workers")
    
    def _optimize_parameters(self):
        """Otimiza parâmetros baseado no ambiente detectado"""
        # Ajusta tamanho da população com base no hardware disponível
        genetic_params = self.env_detector.get_optimal_genetic_params(difficulty_level=71)
        
        # Atualiza configurações otimizadas
        if 'population_size' in genetic_params:
            self.population_size = genetic_params['population_size']
            self.elite_size = int(self.population_size * self.elite_ratio)
            
        if 'mutation_rate' in genetic_params:
            self.mutation_rate = genetic_params['mutation_rate']
            
        if 'crossover_rate' in genetic_params:
            self.crossover_rate = genetic_params['crossover_rate']
        
        # Configuração para operações em paralelo
        self.batch_size = self.env_config['batch_size']
        self.max_workers = self.env_config['max_workers']
        
        # Configuração para análises avançadas
        self.monte_carlo_samples = self.env_config.get('monte_carlo_samples', 1000)
        self.use_bayesian = self.env_config.get('bayesian_optimization', False)
        
        # Adapta estratégia baseada em recursos de hardware
        self.adaptive_exploration = self.env_config['parallel_strategies']['adaptive_exploration']
        
        # Define métodos a serem usados com base no hardware
        if self.gpu_available:
            self._setup_gpu_methods()
        else:
            self._setup_cpu_methods()
    
    def _setup_gpu_methods(self):
        """Configura métodos otimizados para GPU"""
        self.use_gpu_crypto = True
        
        # Seleciona implementação GPU adequada
        if TORCH_AVAILABLE:
            self._setup_torch_methods()
        elif CUPY_AVAILABLE:
            self._setup_cupy_methods()
        else:
            print("⚠️ GPU detectada, mas nenhuma biblioteca de aceleração encontrada!")
            print("⚠️ Por favor, instale PyTorch ou CuPy para aceleração GPU")
            self.use_gpu_crypto = False
            self._setup_cpu_methods()
    
    def _setup_torch_methods(self):
        """Configura métodos usando PyTorch"""
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch não disponível, usando CPU")
            self.device = "cpu"
            return
            
        # Importação segura, já verificamos que está disponível
        import torch
        
        # Determina o dispositivo apropriado
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            self.device = torch.device("hip")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"🔄 Usando aceleração PyTorch em {self.device}")
    
    def _setup_cupy_methods(self):
        """Configura métodos usando CuPy"""
        if not CUPY_AVAILABLE:
            print("⚠️ CuPy não disponível, usando CPU")
            return
            
        # Importação segura, já verificamos que está disponível
        import cupy as cp
        print(f"🔄 Usando aceleração CuPy")
    
    def _setup_cpu_methods(self):
        """Configura métodos otimizados para CPU"""
        self.use_gpu_crypto = False
        print(f"🔄 Usando processamento CPU com {self.max_workers} workers")
        
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
        # Verifica se podemos usar uma versão GPU-acelerada
        if self.use_gpu_crypto and TORCH_AVAILABLE and hasattr(self, 'device'):
            return self._private_key_to_public_gpu(private_key)
        
        # Fallback para versão CPU
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
    
    def _private_key_to_public_gpu(self, private_key: int) -> Tuple[int, int]:
        """Versão GPU-acelerada da conversão de chave privada para pública"""
        # Esta é uma implementação simplificada. Uma implementação completa
        # exigiria kernels CUDA específicos para operações com curvas elípticas.
        
        try:
            if TORCH_AVAILABLE:
                # Importação segura, já verificamos que está disponível
                import torch
                # Implementação básica usando PyTorch
                # Para operações reais com curvas elípticas, será necessário
                # implementar as operações matemáticas específicas do SECP256k1
            
            # Por enquanto, fazemos fallback para CPU
            return self._private_key_to_public_cpu(private_key)
        except Exception as e:
            print(f"⚠️ Erro em GPU: {e}, usando fallback CPU")
            return self._private_key_to_public_cpu(private_key)
    
    def _private_key_to_public_cpu(self, private_key: int) -> Tuple[int, int]:
        """Versão CPU da conversão de chave privada para pública"""
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
        
        # Métrica de fitness otimizada para curva elíptica
        euclidean_distance = math.sqrt(dx*dx + dy*dy)
        
        # Peso para pontos próximos na curva
        secp256k1_weight = self._calculate_secp256k1_proximity(individual)
        
        # Analisa entropia
        entropy_bonus = individual.entropy * 0.1
        
        # Fitness menor = melhor
        fitness = euclidean_distance - entropy_bonus - secp256k1_weight
        
        # Bonus adicional para proximidade extrema
        if euclidean_distance < 1000:
            fitness *= 0.5
        
        return fitness
    
    def _calculate_secp256k1_proximity(self, individual: Individual) -> float:
        """Calcula proximidade na curva secp256k1 com otimizações específicas"""
        # Esta função implementa análise de proximidade específica para a curva elíptica
        # baseada em propriedades matemáticas da curva secp256k1
        
        # Propriedades da curva
        p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        
        # Cálculos de proximidade baseados em propriedades da curva
        # Distância "Manhattan" nos bits mais significativos (mais importantes)
        x_bin = bin(individual.public_key_x ^ self.target_x)[2:]
        y_bin = bin(individual.public_key_y ^ self.target_y)[2:]
        
        # Conta diferenças nos bits, dando peso maior aos mais significativos
        bit_differences = 0
        
        # Implementação otimizada para calcular diferenças de bits ponderadas
        for i, (bit_x, bit_y) in enumerate(zip(x_bin.zfill(256), y_bin.zfill(256))):
            weight = 1 + (0.01 * (256 - i))  # Bits mais significativos têm maior peso
            if bit_x != '0':
                bit_differences += weight
            if bit_y != '0':
                bit_differences += weight
                
        # Inversão para que um valor maior seja melhor
        proximity_score = 100.0 / (1 + bit_differences)
        
        return proximity_score
    
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
        """Utiliza estratégias avançadas de crossover"""
        # Usa o sistema avançado de estratégias
        parent1_key, parent2_key = parent1.private_key, parent2.private_key
        
        # Aplicar crossover adaptativo das estratégias avançadas
        child1_key, child2_key = self.advanced_strategies.adaptive_crossover(
            parent1_key, parent2_key, self.crossover_rate
        )
        
        # Cria novos indivíduos
        child1 = self.create_individual(child1_key)
        child2 = self.create_individual(child2_key)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Aplica estratégias avançadas de mutação"""
        # Usa o sistema avançado de estratégias
        mutated_key = self.advanced_strategies.adaptive_mutation(
            individual.private_key, 
            self.mutation_rate,
            self.generation,
            self.best_fitness
        )
        
        # Cria novo indivíduo com a chave mutada
        return self.create_individual(mutated_key)
    
    def selection(self, population: List[Individual]) -> List[Individual]:
        """Seleção avançada com múltiplas estratégias"""
        # Converter população para formato compatível com estratégias avançadas
        pop_dict = [{'private_key': ind.private_key,
                   'public_key_x': ind.public_key_x,
                   'public_key_y': ind.public_key_y,
                   'fitness': ind.fitness,
                   'entropy': ind.entropy,
                   'generation': ind.generation} 
                  for ind in population]
        
        # Aplicar seleção adaptativa avançada
        selected_dicts = self.advanced_strategies.adaptive_selection(
            pop_dict,
            self.population_size,
            'fitness'
        )
        
        # Converter de volta para indivíduos
        selected = []
        for ind_dict in selected_dicts:
            selected.append(self.create_individual(ind_dict['private_key']))
        
        return selected
    
    def apply_bayesian_optimization(self, population: List[Individual]) -> List[Individual]:
        """Aplica otimização bayesiana para melhorar a população"""
        if not self.use_bayesian:
            return []
            
        # Converter para formato compatível
        pop_dict = [{'private_key': ind.private_key, 'fitness': ind.fitness} 
                  for ind in population]
        
        # Aplicar otimização bayesiana
        optimized_dicts = self.advanced_strategies.bayesian_optimization(
            pop_dict,
            'fitness',
            num_samples=min(50, self.population_size // 10)
        )
        
        # Converter para indivíduos
        optimized = []
        for ind_dict in optimized_dicts:
            optimized.append(self.create_individual(ind_dict['private_key']))
            
        return optimized
    
    def apply_monte_carlo_exploration(self, best_individual: Individual) -> List[Individual]:
        """Aplica exploração Monte Carlo ao redor do melhor indivíduo"""
        # Define raio de exploração baseado na geração
        radius = max(10000, 2**40 // (1 + self.generation))
        
        # Obtém amostragem Monte Carlo
        sampled_keys = self.advanced_strategies.monte_carlo_exploration(
            best_individual.private_key,
            radius=radius,
            samples=min(100, self.population_size // 10)
        )
        
        # Converte para indivíduos
        monte_carlo_individuals = []
        for key in sampled_keys:
            monte_carlo_individuals.append(self.create_individual(key))
            
        return monte_carlo_individuals
    
    def apply_bayesian_exploration(self, population: List[Individual]) -> List[Individual]:
        """
        Aplica exploração bayesiana para refinar a busca e direcioná-la para regiões promissoras
        
        Args:
            population: População atual de indivíduos
            
        Returns:
            Lista de novos indivíduos gerados por exploração bayesiana
        """
        # Verifica se temos suporte a GPU e kernels
        if not self.use_bayesian or not hasattr(self, 'gpu_kernels'):
            # Se não tiver suporte, usa abordagem alternativa mais simples
            return self.apply_monte_carlo_exploration(population[0])
        
        try:
            # 1. Extrai informação da população atual
            private_keys = [ind.private_key for ind in population]
            fitness_values = [ind.fitness for ind in population]
            
            # 2. Toma as melhores chaves como espaço de amostra
            elite_size = min(50, len(population) // 10)
            elite_keys = private_keys[:elite_size]
            
            # 3. Gera novas chaves a explorar baseado nos parâmetros da elite
            min_key = min(elite_keys)
            max_key = max(elite_keys)
            mean_key = sum(elite_keys) // len(elite_keys)
            
            # Determina um raio de exploração baseado na geração atual
            # Quanto mais avançada a geração, menor o raio de exploração
            # Isso foca a busca com o tempo
            radius = 2**36 // (1 + self.generation // 10)
            
            # Gera um conjunto de chaves para exploração
            sample_keys = []
            for _ in range(100):
                # Usa uma distribuição equilibrada entre o melhor valor e valores aleatórios
                if random.random() < 0.7:
                    # 70% das vezes explora ao redor do melhor valor
                    offset = random.randint(-radius, radius)
                    key = private_keys[0] + offset
                else:
                    # 30% das vezes explora ao redor da média
                    offset = random.randint(-radius*10, radius*10) 
                    key = mean_key + offset
                
                # Garante que a chave está dentro dos limites
                key = max(self.min_key, min(self.max_key, key))
                sample_keys.append(key)
            
            # 4. Aplica inferência bayesiana para dirigir exploração
            if len(population) >= 30:
                # Se tivermos informação suficiente, usando inferência bayesiana
                posterior = self.gpu_kernels.batch_bayesian_inference(
                    private_keys[:30],  # Usa os 30 melhores indivíduos
                    fitness_values[:30],
                    exploration_factor=0.3
                )
                
                # Extrapolação bayesiana: usamos os resultados para prever
                # regiões promissoras com base em similaridade estrutural
                predicted_keys = []
                for i in range(len(posterior)):
                    if posterior[i] > 0.02:  # Threshold para regiões promissoras
                        # Explora regiões ao redor das chaves com alta probabilidade
                        for _ in range(3):  # 3 amostras por região promissora
                            # Gera offset com magnitude inversamente proporcional à probabilidade
                            # (maior probabilidade = exploração mais focada)
                            offset_scale = int(2**32 * (1.0 - posterior[i]))
                            offset = random.randint(-offset_scale, offset_scale)
                            new_key = max(self.min_key, min(self.max_key, private_keys[i] + offset))
                            predicted_keys.append(new_key)
                
                # Adiciona as chaves preditas ao conjunto de amostra
                sample_keys.extend(predicted_keys)
            
            # 5. Gera indivíduos a partir das chaves amostradas
            bayesian_individuals = []
            for key in sample_keys:
                bayesian_individuals.append(self.create_individual(key))
                
            # Seleciona os melhores
            bayesian_individuals.sort(key=lambda ind: ind.fitness)
            
            # Retorna os melhores até o limite de 10% da população
            max_to_return = self.population_size // 10
            return bayesian_individuals[:max_to_return]
            
        except Exception as e:
            print(f"⚠️ Erro na exploração bayesiana: {e}")
            # Fallback para método mais simples
            return self.apply_monte_carlo_exploration(population[0])
    
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Evolui uma geração completa com estratégias adaptativas"""
        self.generation += 1
        
        # Atualiza parâmetros adaptativos
        self._update_adaptive_parameters(population)
        
        # Etapa 1: Seleção adaptativa
        selected = self.selection(population)
        
        # Etapa 2: Estratégias avançadas de exploração/explotação
        new_population = []
        
        # Elite sempre passa direto (explotação)
        elite_size = self.elite_size
        new_population.extend(selected[:elite_size])
        
        # Indivíduos de Otimização Bayesiana (exploração guiada)
        if self.use_bayesian:
            # Usa nova implementação baseada em inferência bayesiana
            bayesian_individuals = self.apply_bayesian_exploration(population)
            if bayesian_individuals:
                print(f"   ├─ Adicionados {len(bayesian_individuals)} indivíduos de exploração bayesiana")
                new_population.extend(bayesian_individuals)
        elif self.generation % 5 == 0:
            # Fallback para método anterior se bayesiano não estiver habilitado
            bayesian_individuals = self.apply_bayesian_optimization(population)
            if bayesian_individuals:
                print(f"   ├─ Adicionados {len(bayesian_individuals)} indivíduos de otimização bayesiana")
                new_population.extend(bayesian_individuals)
        
        # Exploração por Monte Carlo (exploração)
        if self.generation % 3 == 0:
            monte_carlo_individuals = self.apply_monte_carlo_exploration(population[0])
            if monte_carlo_individuals:
                print(f"   ├─ Adicionados {len(monte_carlo_individuals)} indivíduos de exploração Monte Carlo")
                new_population.extend(monte_carlo_individuals)
        
        # Decide se foco na exploração ou explotação
        explore_ratio = self.advanced_strategies.stats['exploration_vs_exploitation_ratio']
        
        # Aplica exploração adaptativa em paralelo se disponível
        if self.adaptive_exploration and self.generation % 10 == 0:
            exploration_results = self._run_parallel_explorations(population[0], population)
            if exploration_results:
                print(f"   ├─ Adicionados {len(exploration_results)} indivíduos de exploração paralela")
                new_population.extend(exploration_results)
        
        # Completa população restante com crossover e mutação
        while len(new_population) < self.population_size:
            # Seleciona pais aleatoriamente dos selecionados
            parent1, parent2 = random.sample(selected, 2)
            
            # Aplica crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Aplica mutação com probabilidade ajustada ao ratio de exploração
            mutation_prob = self.mutation_rate * (1 + explore_ratio)
            if random.random() < mutation_prob:
                child1 = self.mutate(child1)
            if random.random() < mutation_prob:
                child2 = self.mutate(child2)
            
            # Adiciona à nova população
            new_population.extend([child1, child2])
        
        # Ajusta para o tamanho exato da população
        new_population = new_population[:self.population_size]
        
        # Ordena por fitness
        new_population.sort(key=lambda x: x.fitness)
        
        # Atualiza melhor fitness
        current_best = new_population[0].fitness
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            print(f"🏆 Nova melhor fitness: {current_best:.2f} (Geração {self.generation})")
            
            # Se tivermos uma melhoria significativa, salva checkpoint
            self.save_checkpoint(new_population)
        
        # A cada 50 gerações, aplica meta-aprendizado para otimizar estratégias
        if self.generation % 50 == 0:
            # Converte para formato compatível
            pop_dict = [{'private_key': ind.private_key, 'fitness': ind.fitness} 
                    for ind in new_population]
            
            # Aplica meta-aprendizado
            self.advanced_strategies.meta_learning_adaptation(pop_dict, new_population[0].fitness)
        
        return new_population
        
    def _update_adaptive_parameters(self, population: List[Individual]) -> None:
        """Atualiza parâmetros adaptativos baseado no estado atual"""
        # Obtém parâmetros otimizados das estratégias avançadas
        if self.generation > 1 and self.generation % 10 == 0:
            opt_params = self.advanced_strategies.get_optimal_parameters()
            
            # Atualiza taxas com base nas recomendações
            if 'mutation_rate' in opt_params:
                self.mutation_rate = opt_params['mutation_rate']
            
            if 'crossover_rate' in opt_params:
                self.crossover_rate = opt_params['crossover_rate']
    
    def _run_parallel_explorations(self, best_individual: Individual, 
                                 population: List[Individual]) -> List[Individual]:
        """Executa exploração em paralelo de diferentes regiões"""
        if not hasattr(self, 'parallel_config'):
            return []
            
        # Número de workers disponíveis
        n_workers = self.parallel_config['max_workers']
        
        if n_workers <= 1:
            return []
            
        # Cria regiões de exploração
        regions = []
        
        # Região 1: Ao redor do melhor indivíduo atual
        regions.append((best_individual.private_key - 2**30, best_individual.private_key + 2**30))
        
        # Região 2: Na metade do intervalo
        middle = (self.min_key + self.max_key) // 2
        regions.append((middle - 2**40, middle + 2**40))
        
        # Outras regiões baseadas nas regiões promissoras encontradas
        for start, end in self.advanced_strategies.promising_regions[:5]:  # Limita a 5 regiões
            regions.append((start, end))
            
        # Limita ao número de workers
        regions = regions[:n_workers]
        
        # Função para explorar uma região
        def explore_region(region_start, region_end, count=10):
            results = []
            for _ in range(count):
                key = random.randint(region_start, region_end)
                ind = self.create_individual(key)
                results.append(ind)
            return results
            
        # Executa em paralelo se tivermos múltiplas regiões
        if len(regions) > 1:
            exploration_results = []
            
            # Usa ProcessPoolExecutor para paralelismo
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                
                # Submete tarefas
                for start, end in regions:
                    futures.append(
                        executor.submit(explore_region, start, end, 5)
                    )
                
                # Coleta resultados
                for future in futures:
                    try:
                        results = future.result()
                        exploration_results.extend(results)
                    except Exception as e:
                        print(f"⚠️ Erro em exploração paralela: {e}")
                
            return exploration_results
        else:
            # Fallback para exploração sequencial
            return explore_region(regions[0][0], regions[0][1], 20)
    
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
        
        # Calculando a diversidade da população
        diversity = self._calculate_population_diversity(population)
        
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
        print(f"🔄 Diversidade: {diversity:.2f}")
        
        # Estatísticas de estratégias avançadas
        if self.generation % 10 == 0:
            try:
                strategy_stats = self.advanced_strategies.get_strategy_stats()
                print(f"\n🧠 ESTATÍSTICAS DE ESTRATÉGIAS:")
                
                # Estratégias de mutação mais usadas
                if 'mutations' in strategy_stats:
                    mutations = strategy_stats['mutations']
                    top_mutations = sorted(mutations.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("   ├─ Mutações mais usadas: " + ", ".join(f"{k.split('_')[0]}:{v}" for k, v in top_mutations))
                
                # Estratégias de crossover mais usadas
                if 'crossovers' in strategy_stats:
                    crossovers = strategy_stats['crossovers']
                    top_crossovers = sorted(crossovers.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("   ├─ Crossovers mais usados: " + ", ".join(f"{k.split('_')[0]}:{v}" for k, v in top_crossovers))
                
                # Modo exploração vs. explotação atual
                explore_ratio = self.advanced_strategies.stats['exploration_vs_exploitation_ratio']
                print(f"   └─ Explore/Exploit: {explore_ratio:.2f}")
            except Exception as e:
                pass  # Ignora erros nas estatísticas
        
        # Mostra os 3 melhores
        print(f"\n🏅 TOP 3 INDIVÍDUOS:")
        for i, ind in enumerate(population[:3]):
            print(f"   {i+1}. Fitness: {ind.fitness:.2f} | "
                  f"Entropia: {ind.entropy:.4f} | "
                  f"Chave: 0x{ind.private_key:016x}")
    
    def _calculate_population_diversity(self, population: List[Individual]) -> float:
        """Calcula diversidade da população baseado em suas chaves privadas"""
        # Amostra um subconjunto para eficiência
        sample_size = min(50, len(population))
        sample = random.sample(population, sample_size)
        
        # Calcula a distância de Hamming média entre os indivíduos
        total_distance = 0
        comparisons = 0
        
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                # XOR entre chaves pega as diferenças bit a bit
                key_diff = sample[i].private_key ^ sample[j].private_key
                
                # Conta bits que diferem
                diff_count = bin(key_diff).count('1')
                
                total_distance += diff_count
                comparisons += 1
        
        # Diversidade normalizada (0-1)
        if comparisons > 0:
            avg_distance = total_distance / comparisons
            normalized_diversity = avg_distance / 71  # Normalizado para 71 bits
            return normalized_diversity
        else:
            return 0.0
    
    def save_checkpoint(self, population: List[Individual]):
        """Salva checkpoint da evolução com suporte a retomada avançado"""
        # Salva mais informações para permitir retomada precisa
        checkpoint_data = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'keys_evaluated': self.keys_evaluated,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'timestamp': time.time(),
            'advanced_strategies': {
                'generation': self.advanced_strategies.generation,
                'stagnation_counter': self.advanced_strategies.stagnation_counter,
                'exploration_rate': self.advanced_strategies.stats['exploration_vs_exploitation_ratio'],
                'promising_regions': self.advanced_strategies.promising_regions
            },
            'population': [
                {
                    'private_key': ind.private_key,
                    'public_key_x': ind.public_key_x,
                    'public_key_y': ind.public_key_y,
                    'fitness': ind.fitness,
                    'entropy': ind.entropy,
                    'generation': ind.generation
                }
                for ind in population[:min(50, self.population_size)]  # Salva mais indivíduos
            ]
        }
        
        # Salva checkpoint principal
        with open('genetic_checkpoint.json', 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # A cada 10 gerações, salva uma cópia de backup com timestamp
        if self.generation % 10 == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f'checkpoint_gen{self.generation}_{timestamp}.json'
            
            try:
                with open(backup_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"💾 Checkpoint backup salvo: {backup_path}")
            except:
                pass  # Ignora erros no backup
        
        print(f"💾 Checkpoint salvo (Geração {self.generation})")
    
    def load_checkpoint(self, checkpoint_file='genetic_checkpoint.json') -> bool:
        """
        Carrega checkpoint com suporte avançado
        
        Args:
            checkpoint_file: Caminho do arquivo de checkpoint a ser carregado
            
        Returns:
            True se o checkpoint foi carregado com sucesso, False caso contrário
        """
        try:
            if os.path.exists(checkpoint_file):
                print(f"🔄 Checkpoint encontrado: {checkpoint_file}! Carregando progresso anterior...")
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                # Restaura estado do solver
                self.generation = data['generation']
                self.best_fitness = data['best_fitness']
                self.keys_evaluated = data['keys_evaluated']
                self.mutation_rate = data['mutation_rate']
                self.crossover_rate = data.get('crossover_rate', 0.8)
                
                # Restaura dados das estratégias avançadas
                if 'advanced_strategies' in data:
                    strategies_data = data['advanced_strategies']
                    self.advanced_strategies.generation = strategies_data.get('generation', 0)
                    self.advanced_strategies.stagnation_counter = strategies_data.get('stagnation_counter', 0)
                    self.advanced_strategies.stats['exploration_vs_exploitation_ratio'] = strategies_data.get('exploration_rate', 0.5)
                    self.advanced_strategies.promising_regions = strategies_data.get('promising_regions', [])
                
                # Restaura população
                population = []
                for ind_data in data['population']:
                    if 'private_key' in ind_data:
                        ind = self.create_individual(ind_data['private_key'])
                        population.append(ind)
                
                # Completa população com variantes e indivíduos aleatórios
                if len(population) > 0:
                    best_individual = population[0]
                    
                    # Adiciona variantes baseadas nos melhores indivíduos
                    while len(population) < self.population_size * 0.7:  # 70% baseado no checkpoint
                        # Seleciona um indivíduo aleatório dos carregados
                        source_ind = random.choice(population[:min(10, len(population))])
                        
                        # Aplica pequena mutação
                        key = source_ind.private_key
                        bit_pos = random.randint(0, 70)
                        key ^= (1 << bit_pos)  # Flip um bit
                        
                        # Cria novo indivíduo com essa variação
                        new_ind = self.create_individual(key)
                        population.append(new_ind)
                
                # Completa o restante da população com indivíduos aleatórios
                while len(population) < self.population_size:
                    new_ind = self.create_individual()
                    population.append(new_ind)
                
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
                        f.write(f"Entropia: {solution.entropia}\n")
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
                
                # Alternância entre exploração ampla e otimização local
                if generation % 20 == 0 and generation > 0:
                    explore_rate = self.advanced_strategies.stats['exploration_vs_exploitation_ratio']
                    print(f"🔍 Modo atual: {'Exploração' if explore_rate > 0.5 else 'Otimização local'}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Evolução interrompida pelo usuário")
            self.save_checkpoint(population)
            return None
        
        print(f"\n🏁 Evolução concluída após {max_generations} gerações")
        self.print_statistics(population)
        return population[0]  # Retorna o melhor indivíduo

    def _integrate_gpu_kernels(self):
        """Integra kernels GPU para aceleração de operações criptográficas"""
        try:
            # Importa kernels GPU
            from .gpu_kernels import get_gpu_kernels
            
            # Inicializa kernels GPU usando detector de ambiente existente
            self.gpu_kernels = get_gpu_kernels(self.env_detector)
            
            if self.gpu_kernels.has_gpu and self.gpu_kernels.accelerated_modules_loaded:
                print("✅ Kernels GPU inicializados com sucesso")
                
                # Substitui métodos principais por versões aceleradas
                self.generate_pubkeys_batch = self._gpu_generate_pubkeys_batch
                self.calculate_fitness_batch = self._gpu_calculate_fitness_batch
                
                # Configure o tamanho de lote para operações em massa
                self.gpu_batch_size = self.gpu_kernels.batch_size
                
                return True
            else:
                print("⚠️ Kernels GPU não disponíveis, usando CPU para operações criptográficas")
                return False
        except ImportError as e:
            print(f"⚠️ Erro ao carregar kernels GPU: {e}")
            return False

    def _gpu_generate_pubkeys_batch(self, private_keys):
        """Versão otimizada para GPU de geração de chaves públicas em lote"""
        # Verifica se os kernels GPU estão disponíveis
        if not hasattr(self, 'gpu_kernels'):
            # Fallback para CPU se kernels não estiverem disponíveis
            return self._generate_pubkeys_batch_cpu(private_keys)
            
        try:
            # Usa kernels GPU para gerar chaves públicas em lote
            pubkeys = self.gpu_kernels.batch_generate_pubkeys(private_keys)
            return pubkeys
        except Exception as e:
            print(f"⚠️ Erro na geração de chaves públicas GPU: {e}")
            # Fallback para CPU em caso de erro
            return self._generate_pubkeys_batch_cpu(private_keys)

    def _gpu_calculate_fitness_batch(self, pubkeys):
        """Versão otimizada para GPU de cálculo de fitness em lote"""
        # Verifica se os kernels GPU estão disponíveis
        if not hasattr(self, 'gpu_kernels'):
            # Fallback para CPU se kernels não estiverem disponíveis
            return self._calculate_fitness_batch_cpu(pubkeys)
            
        try:
            # Usa kernels GPU para calcular fitness em lote
            target_point = (self.target_x, self.target_y)
            fitness_values = self.gpu_kernels.batch_calculate_fitness(pubkeys, target_point)
            return fitness_values
        except Exception as e:
            print(f"⚠️ Erro no cálculo de fitness GPU: {e}")
            # Fallback para CPU em caso de erro
            return self._calculate_fitness_batch_cpu(pubkeys)
            
    def _generate_pubkeys_batch_cpu(self, private_keys):
        """Implementação CPU para geração de chaves públicas em lote (fallback)"""
        import ecdsa
        curve = ecdsa.SECP256k1
        g = curve.generator
        
        result = []
        for priv_key in private_keys:
            point = g * priv_key
            result.append((point.x(), point.y()))
            
        return result
        
    def _calculate_fitness_batch_cpu(self, pubkeys):
        """Implementação CPU para cálculo de fitness em lote (fallback)"""
        result = []
        
        for pub_x, pub_y in pubkeys:
            # Calcular diferença em coordenadas
            dx = abs(pub_x - self.target_x)
            dy = abs(pub_y - self.target_y)
            
            # Normalização por módulo
            p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
            dx = min(dx, p - dx)
            dy = min(dy, p - dy)
            
            # Ponderação: coordenada x é mais importante que y
            fitness = dx * 0.75 + dy * 0.25
            result.append(fitness)
            
        return result
