#!/usr/bin/env python3
"""
🧠 ESTRATÉGIAS AVANÇADAS PARA ALGORITMO GENÉTICO
===============================================

Implementação de estratégias avançadas para aumentar a eficiência
do algoritmo genético na resolução do Bitcoin Puzzle 71:

- Operadores adaptativos baseados em aprendizado
- Estratégias de exploração vs. explotação
- Técnicas de meta-aprendizado
- Inferência bayesiana para guiar a busca
- Amostragem por Monte Carlo
- Detecção e manutenção de diversidade
"""

import random
import math
import numpy as np
from typing import List, Dict, Tuple, Any, Callable, Optional
import itertools
import time
import hashlib

class AdvancedGeneticStrategies:
    """Implementa estratégias avançadas para o algoritmo genético"""
    
    def __init__(self, min_key: int, max_key: int, config: Dict[str, Any] = None):
        self.min_key = min_key
        self.max_key = max_key
        self.range_size = max_key - min_key + 1
        self.config = config or {}
        
        # Inicializa contadores para análise
        self.stats = {
            'mutation_applications': {},
            'crossover_applications': {},
            'selection_applications': {},
            'strategy_success_rate': {},
            'diversity_measurements': [],
            'fitness_improvement': [],
            'exploration_vs_exploitation_ratio': 0.5,  # Inicialmente equilibrado
        }
        
        # Análise de regiões promissoras
        self.promising_regions = []
        
        # Histórico para meta-aprendizado
        self.history = {
            'mutations': [],
            'crossovers': [],
            'fitness_improvements': [],
            'diversity_changes': []
        }
        
        # Inicializa contador de gerações
        self.generation = 0
        
        # Parâmetros adaptativos
        self.adaptation = {
            'mutation_rate': self.config.get('initial_mutation_rate', 0.02),
            'crossover_rate': self.config.get('initial_crossover_rate', 0.8),
            'selection_pressure': self.config.get('initial_selection_pressure', 2.0),
            'diversity_weight': 0.4,  # Peso dado à diversidade vs fitness
        }
        
        # Inicializa rastreadores de progresso
        self.best_fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        
        print("🧠 Estratégias avançadas de algoritmo genético inicializadas")
    
    # =========== MÉTODOS DE MUTAÇÃO AVANÇADOS ===========
    
    def adaptive_mutation(self, key: int, mutation_rate: float, 
                          generation: int, best_fitness: float) -> int:
        """Estratégia de mutação adaptativa baseada no contexto atual"""
        # Escolhe entre várias estratégias baseado no estado atual
        strategies = [
            (self.bit_flip_mutation, 0.3),
            (self.byte_flip_mutation, 0.2),
            (self.gaussian_mutation, 0.15),
            (self.smart_mutation, 0.25),
            (self.differential_mutation, 0.1)
        ]
        
        # Ajusta probabilidades baseado na estagnação
        if self.stagnation_counter > 10:
            # Aumenta probabilidade de mutações mais disruptivas em caso de estagnação
            strategies = [
                (self.bit_flip_mutation, 0.2),
                (self.byte_flip_mutation, 0.2),
                (self.gaussian_mutation, 0.15),
                (self.smart_mutation, 0.3),
                (self.differential_mutation, 0.15)
            ]
        
        # Amostragem de estratégia baseada em pesos
        strategy_funcs, probs = zip(*strategies)
        strategy = np.random.choice(strategy_funcs, p=probs)
        
        # Registra para análise
        strategy_name = strategy.__name__
        if strategy_name not in self.stats['mutation_applications']:
            self.stats['mutation_applications'][strategy_name] = 0
        self.stats['mutation_applications'][strategy_name] += 1
        
        # Executa a estratégia escolhida
        mutated_key = strategy(key, mutation_rate)
        
        # Garante que a chave está dentro dos limites
        mutated_key = max(self.min_key, min(self.max_key, mutated_key))
        
        return mutated_key
    
    def bit_flip_mutation(self, key: int, mutation_rate: float) -> int:
        """Mutação clássica de flipping de bits individuais"""
        mutated = key
        for bit in range(71):  # Para o puzzle 71
            if random.random() < mutation_rate:
                # Flipa o bit
                mask = 1 << bit
                mutated ^= mask
        return mutated
    
    def byte_flip_mutation(self, key: int, mutation_rate: float) -> int:
        """Mutação que flipa bytes inteiros, mais agressiva"""
        mutated = key
        for byte_pos in range(9):  # 71 bits ≈ 9 bytes
            if random.random() < mutation_rate * 0.5:  # Taxa reduzida
                # Flipa um byte inteiro (8 bits)
                byte_mask = 0xFF << (byte_pos * 8)
                flip_value = random.randint(0, 0xFF) << (byte_pos * 8)
                # Remove o byte atual e insere o novo
                mutated = (mutated & ~byte_mask) | flip_value
        return mutated
    
    def gaussian_mutation(self, key: int, mutation_rate: float) -> int:
        """Mutação baseada em distribuição gaussiana"""
        # Escala de mutação depende do tamanho do espaço
        scale = self.range_size * 0.0000001  # Escala pequena para mudanças suaves
        delta = int(np.random.normal(0, scale))
        
        # Aplicar apenas se random < mutation_rate
        if random.random() < mutation_rate:
            return key + delta
        return key
    
    def smart_mutation(self, key: int, mutation_rate: float) -> int:
        """Mutação inteligente baseada em análise de regiões promissoras"""
        # Se tivermos regiões promissoras, tenta usar
        if self.promising_regions and random.random() < 0.4:
            # Escolhe uma região promissora aleatória
            region_start, region_end = random.choice(self.promising_regions)
            # Gera uma chave nessa região
            return random.randint(region_start, region_end)
        
        # Caso contrário, usa mutação adaptada à fase
        if self.generation < 100:
            # Fase inicial: mutações mais exploratórias
            return self.bit_flip_mutation(key, mutation_rate * 1.5)
        elif self.stagnation_counter > 15:
            # Estagnado: tentar algo completamente diferente
            return random.randint(self.min_key, self.max_key)
        else:
            # Fase normal: mutação de bits com intensidade ajustada
            return self.bit_flip_mutation(key, mutation_rate)
    
    def differential_mutation(self, key: int, mutation_rate: float) -> int:
        """Mutação inspirada em evolução diferencial"""
        # Implementação simplificada - normalmente precisaria de mais indivíduos
        # para a mutação diferencial completa
        if not hasattr(self, 'prev_best_keys'):
            self.prev_best_keys = [key]
            return self.bit_flip_mutation(key, mutation_rate)
        
        # Usa histórico de bons indivíduos como base
        if len(self.prev_best_keys) > 3:
            a, b, c = random.sample(self.prev_best_keys, 3)
            # Operação de mutação diferencial: a + F*(b-c)
            F = 0.8  # Fator diferencial
            diff = b - c
            scaled_diff = int(F * diff)
            mutated = a + scaled_diff
            
            # Adiciona alguma variação adicional
            if random.random() < mutation_rate:
                mutated ^= (1 << random.randint(0, 70))
                
            return mutated
        else:
            # Fallback se não tiver histórico suficiente
            self.prev_best_keys.append(key)
            return self.bit_flip_mutation(key, mutation_rate)
    
    # =========== MÉTODOS DE CROSSOVER AVANÇADOS ===========
    
    def adaptive_crossover(self, parent1: int, parent2: int, crossover_rate: float) -> Tuple[int, int]:
        """Aplica estratégia de crossover adaptativa baseada no contexto"""
        if random.random() > crossover_rate:
            return parent1, parent2
            
        # Escolhe entre várias estratégias baseado no contexto atual
        strategies = [
            (self.uniform_crossover, 0.25),
            (self.two_point_crossover, 0.25),
            (self.arithmetic_crossover, 0.2),
            (self.single_point_crossover, 0.2),
            (self.differential_crossover, 0.1)
        ]
        
        # Ajusta probabilidades baseado na estagnação
        if self.stagnation_counter > 10:
            # Favorece crossovers mais disruptivos quando estagnado
            strategies = [
                (self.uniform_crossover, 0.3),
                (self.arithmetic_crossover, 0.3),
                (self.differential_crossover, 0.2),
                (self.two_point_crossover, 0.1),
                (self.single_point_crossover, 0.1)
            ]
        
        # Amostragem de estratégia baseada em pesos
        strategy_funcs, probs = zip(*strategies)
        strategy = np.random.choice(strategy_funcs, p=probs)
        
        # Registra para análise
        strategy_name = strategy.__name__
        if strategy_name not in self.stats['crossover_applications']:
            self.stats['crossover_applications'][strategy_name] = 0
        self.stats['crossover_applications'][strategy_name] += 1
        
        # Executa a estratégia escolhida
        child1, child2 = strategy(parent1, parent2)
        
        # Garantir que as chaves estão dentro dos limites
        child1 = max(self.min_key, min(self.max_key, child1))
        child2 = max(self.min_key, min(self.max_key, child2))
        
        return child1, child2
    
    def single_point_crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        """Crossover de ponto único tradicional"""
        # Escolha um ponto de corte aleatório
        point = random.randint(1, 69)  # Para um puzzle 71, evita os extremos
        
        # Cria máscara para pegar bits antes e depois do ponto
        mask_before = (1 << point) - 1
        mask_after = ~mask_before & ((1 << 71) - 1)
        
        # Combina bits dos pais
        child1 = (parent1 & mask_before) | (parent2 & mask_after)
        child2 = (parent2 & mask_before) | (parent1 & mask_after)
        
        return child1, child2
    
    def two_point_crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        """Crossover com dois pontos de corte"""
        # Escolhe dois pontos distintos
        point1 = random.randint(1, 69)
        point2 = random.randint(1, 69)
        while point2 == point1:
            point2 = random.randint(1, 69)
            
        if point1 > point2:
            point1, point2 = point2, point1
        
        # Cria máscaras para os três segmentos
        mask1 = (1 << point1) - 1
        mask2 = ((1 << point2) - 1) ^ mask1
        mask3 = ~((1 << point2) - 1) & ((1 << 71) - 1)
        
        # Combina segmentos
        child1 = (parent1 & mask1) | (parent2 & mask2) | (parent1 & mask3)
        child2 = (parent2 & mask1) | (parent1 & mask2) | (parent2 & mask3)
        
        return child1, child2
    
    def uniform_crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        """Crossover uniforme - cada bit é escolhido aleatoriamente de um dos pais"""
        child1, child2 = 0, 0
        
        for bit in range(71):  # Para puzzle 71
            mask = 1 << bit
            bit_p1 = parent1 & mask
            bit_p2 = parent2 & mask
            
            # Para cada bit, escolha aleatoriamente de qual pai herdar
            if random.random() < 0.5:
                child1 |= bit_p1
                child2 |= bit_p2
            else:
                child1 |= bit_p2
                child2 |= bit_p1
        
        return child1, child2
    
    def arithmetic_crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        """Crossover aritmético - usa operações matemáticas para criar filhos"""
        # Diferentes operações aritméticas
        alpha = random.random()  # Fator para combinação linear
        
        # Combinação linear ponderada
        child1 = int(parent1 * alpha + parent2 * (1 - alpha))
        child2 = int(parent1 * (1 - alpha) + parent2 * alpha)
        
        # Adiciona alguma variação para evitar perda de bits significativos
        if random.random() < 0.5:
            bit_pos = random.randint(0, 70)
            child1 ^= (1 << bit_pos)
            
        if random.random() < 0.5:
            bit_pos = random.randint(0, 70)
            child2 ^= (1 << bit_pos)
        
        return child1, child2
    
    def differential_crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        """Inspirado em evolução diferencial"""
        # Implementação simplificada
        diff = abs(parent1 - parent2)
        scale = random.random() * 0.4 + 0.1  # Entre 0.1 e 0.5
        
        # Criar filhos movendo na direção da diferença
        if parent1 > parent2:
            child1 = parent1 + int(diff * scale)
            child2 = parent2 - int(diff * scale)
        else:
            child1 = parent1 - int(diff * scale)
            child2 = parent2 + int(diff * scale)
        
        return child1, child2
    
    # =========== MÉTODOS DE SELEÇÃO AVANÇADOS ===========
    
    def adaptive_selection(self, population: List[Dict], 
                          selection_size: int, 
                          fitness_key: str = 'fitness') -> List[Dict]:
        """Seleciona indivíduos combinando várias estratégias adaptativas"""
        # Adapta estratégia com base na geração e estagnação
        strategies = [
            (self.tournament_selection, 0.4),
            (self.roulette_selection, 0.3),
            (self.rank_selection, 0.2),
            (self.diversity_selection, 0.1)
        ]
        
        # Em caso de estagnação, favorece diversidade
        if self.stagnation_counter > 15:
            strategies = [
                (self.tournament_selection, 0.2),
                (self.roulette_selection, 0.2),
                (self.rank_selection, 0.2),
                (self.diversity_selection, 0.4)
            ]
        
        # Amostragem de estratégia baseada em pesos
        strategy_funcs, probs = zip(*strategies)
        strategy = np.random.choice(strategy_funcs, p=probs)
        
        # Registra para análise
        strategy_name = strategy.__name__
        if strategy_name not in self.stats['selection_applications']:
            self.stats['selection_applications'][strategy_name] = 0
        self.stats['selection_applications'][strategy_name] += 1
        
        # Executa a estratégia escolhida
        return strategy(population, selection_size, fitness_key)
    
    def tournament_selection(self, population: List[Dict], 
                            selection_size: int, 
                            fitness_key: str = 'fitness') -> List[Dict]:
        """Seleção por torneio - competição entre subconjuntos aleatórios"""
        selected = []
        tournament_size = min(5, len(population) // 10)  # Tamanho dinâmico do torneio
        
        while len(selected) < selection_size:
            # Selecionar participantes aleatórios para o torneio
            tournament = random.sample(population, tournament_size)
            # O vencedor é o indivíduo com melhor fitness
            winner = min(tournament, key=lambda x: x[fitness_key])
            selected.append(winner)
        
        return selected
    
    def roulette_selection(self, population: List[Dict], 
                          selection_size: int, 
                          fitness_key: str = 'fitness') -> List[Dict]:
        """Seleção por roleta - probabilidade proporcional ao fitness"""
        selected = []
        
        # Como fitness menor é melhor, precisamos inverter
        total_fitness = sum(1.0 / (ind[fitness_key] + 1e-10) for ind in population)
        
        # Calcula probabilidades de seleção (quanto menor o fitness, maior a probabilidade)
        probabilities = [(1.0 / (ind[fitness_key] + 1e-10)) / total_fitness for ind in population]
        
        # Faz amostragem baseada nas probabilidades
        selected_indices = np.random.choice(
            len(population), 
            size=selection_size, 
            p=probabilities,
            replace=True
        )
        
        for idx in selected_indices:
            selected.append(population[idx])
        
        return selected
    
    def rank_selection(self, population: List[Dict], 
                      selection_size: int, 
                      fitness_key: str = 'fitness') -> List[Dict]:
        """Seleção por ranking - probabilidade baseada no rank de fitness"""
        # Ordena população por fitness
        sorted_pop = sorted(population, key=lambda x: x[fitness_key])
        selected = []
        
        # Probabilidades baseadas no rank (posição na lista ordenada)
        total_rank_sum = len(sorted_pop) * (len(sorted_pop) + 1) / 2
        probabilities = [(len(sorted_pop) - i) / total_rank_sum for i in range(len(sorted_pop))]
        
        # Faz amostragem baseada nas probabilidades
        selected_indices = np.random.choice(
            len(sorted_pop), 
            size=selection_size, 
            p=probabilities,
            replace=True
        )
        
        for idx in selected_indices:
            selected.append(sorted_pop[idx])
        
        return selected
    
    def diversity_selection(self, population: List[Dict], 
                           selection_size: int, 
                           fitness_key: str = 'fitness') -> List[Dict]:
        """Seleção favorecendo diversidade genética"""
        selected = []
        
        # Primeiro, escolhe alguns dos melhores indivíduos (elitismo)
        elite_size = int(selection_size * 0.2)  # 20% de elites
        sorted_pop = sorted(population, key=lambda x: x[fitness_key])
        selected.extend(sorted_pop[:elite_size])
        
        # Para o restante, seleciona indivíduos diversos
        remaining = selection_size - elite_size
        
        # Função auxiliar para calcular "distância" entre indivíduos
        def distance(ind1, ind2):
            # Usando distância de Hamming a nível de bit
            xor = ind1['private_key'] ^ ind2['private_key']
            # Conta bits diferentes
            count = 0
            while xor:
                count += xor & 1
                xor >>= 1
            return count
        
        # Seleção por diversidade
        while len(selected) < selection_size:
            most_diverse = None
            max_min_distance = -1
            
            # Para cada indivíduo não selecionado
            for ind in sorted_pop:
                if ind in selected:
                    continue
                
                # Calcula distância mínima para todos os já selecionados
                min_distance = float('inf')
                for sel in selected:
                    d = distance(ind, sel)
                    min_distance = min(min_distance, d)
                
                # Se este é mais diverso, marca para seleção
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    most_diverse = ind
            
            if most_diverse:
                selected.append(most_diverse)
        
        return selected
    
    # =========== MÉTODOS DE META-APRENDIZADO ===========
    
    def meta_learning_adaptation(self, population: List[Dict], 
                               best_fitness: float,
                               fitness_key: str = 'fitness') -> None:
        """Adapta parâmetros baseado no histórico de aprendizado"""
        self.generation += 1
        
        # Registra melhor fitness para análise
        self.best_fitness_history.append(best_fitness)
        
        # Calcula diversidade da população
        diversity = self._calculate_diversity(population)
        self.diversity_history.append(diversity)
        
        # Detecta estagnação (nenhuma melhoria significativa)
        if len(self.best_fitness_history) > 5:
            recent_improvements = [
                abs(self.best_fitness_history[-i-1] - self.best_fitness_history[-i-2]) 
                for i in range(5)
            ]
            avg_improvement = sum(recent_improvements) / 5
            
            # Se a melhoria for muito pequena, considera estagnado
            if avg_improvement < best_fitness * 0.0001:  # Menos de 0.01% de melhoria
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        # Adapta taxas com base no estado atual
        self._adapt_parameters(diversity)
        
        # Atualiza regiões promissoras
        self._update_promising_regions(population, fitness_key)
        
        # Atualiza taxa de exploração vs. explotação
        self._update_exploration_rate()
        
        # Log interno para debug
        if self.generation % 50 == 0:
            self._log_adaptation_status()
    
    def _calculate_diversity(self, population: List[Dict]) -> float:
        """Calcula diversidade genética da população"""
        if not population:
            return 0.0
            
        sample_size = min(50, len(population))  # Limita amostra para eficiência
        samples = random.sample(population, sample_size)
        
        # Abordagem: média de distância de Hamming entre todos os pares
        total_distance = 0
        pair_count = 0
        
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                ind1 = samples[i]['private_key']
                ind2 = samples[j]['private_key']
                
                # Distância de Hamming
                xor = ind1 ^ ind2
                distance = bin(xor).count('1')  # Conta bits diferentes
                
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
            
        avg_diversity = total_distance / pair_count
        # Normaliza pela quantidade máxima possível (71 bits para Puzzle 71)
        normalized_diversity = avg_diversity / 71
        
        return normalized_diversity
    
    def _adapt_parameters(self, diversity: float) -> None:
        """Adapta parâmetros baseado no estado atual"""
        # Taxa de mutação adaptativa
        if self.stagnation_counter > 15:
            # Estagnação severa: aumenta muito a taxa de mutação
            self.adaptation['mutation_rate'] = min(0.2, self.adaptation['mutation_rate'] * 1.5)
        elif self.stagnation_counter > 5:
            # Estagnação moderada: aumenta um pouco a taxa
            self.adaptation['mutation_rate'] = min(0.1, self.adaptation['mutation_rate'] * 1.2)
        elif diversity < 0.3:
            # Baixa diversidade: aumenta taxa para explorar mais
            self.adaptation['mutation_rate'] = min(0.1, self.adaptation['mutation_rate'] * 1.1)
        else:
            # Bom estado: normaliza taxa gradualmente
            target_rate = 0.02
            self.adaptation['mutation_rate'] = self.adaptation['mutation_rate'] * 0.95 + target_rate * 0.05
        
        # Taxa de crossover adaptativa
        if diversity < 0.2:
            # Baixa diversidade: reduz crossover
            self.adaptation['crossover_rate'] = max(0.6, self.adaptation['crossover_rate'] * 0.95)
        else:
            # Alta diversidade: aumenta crossover
            self.adaptation['crossover_rate'] = min(0.95, self.adaptation['crossover_rate'] * 1.05)
        
        # Pressão de seleção adaptativa
        if self.stagnation_counter > 10:
            # Estagnação: reduz pressão de seleção
            self.adaptation['selection_pressure'] = max(1.0, self.adaptation['selection_pressure'] * 0.9)
        elif diversity < 0.25:
            # Baixa diversidade: reduz pressão
            self.adaptation['selection_pressure'] = max(1.0, self.adaptation['selection_pressure'] * 0.95)
        else:
            # Bom estado: normaliza gradualmente
            target_pressure = 2.0
            self.adaptation['selection_pressure'] = (
                self.adaptation['selection_pressure'] * 0.95 + target_pressure * 0.05
            )
    
    def _update_promising_regions(self, population: List[Dict], fitness_key: str) -> None:
        """Atualiza regiões promissoras baseado nos melhores indivíduos"""
        # Pega os 10% melhores indivíduos
        top_percent = 0.1
        top_count = max(5, int(len(population) * top_percent))
        top_individuals = sorted(population, key=lambda x: x[fitness_key])[:top_count]
        
        # Agrupa indivíduos próximos
        regions = []
        
        for ind in top_individuals:
            key = ind['private_key']
            # Define região como vizinhança de bits próximos
            region_size = 2**20  # Tamanho da região ~1 milhão
            region_start = max(self.min_key, key - region_size // 2)
            region_end = min(self.max_key, region_start + region_size)
            
            # Verifica se sobrepõe com regiões existentes
            merged = False
            for i, (start, end) in enumerate(regions):
                # Se sobrepõe, une as regiões
                if (start <= region_end and end >= region_start):
                    new_start = min(start, region_start)
                    new_end = max(end, region_end)
                    regions[i] = (new_start, new_end)
                    merged = True
                    break
            
            if not merged:
                regions.append((region_start, region_end))
        
        # Limita número de regiões para não dispersar demais
        max_regions = 10
        if len(regions) > max_regions:
            # Ordena por tamanho e pega as maiores
            regions.sort(key=lambda r: r[1] - r[0], reverse=True)
            regions = regions[:max_regions]
        
        self.promising_regions = regions
    
    def _update_exploration_rate(self) -> None:
        """Atualiza a taxa de exploração vs. explotação"""
        base_rate = 0.5  # Base equilibrada
        
        # Ajuste baseado na geração
        generation_factor = max(0.0, min(0.3, (self.generation / 1000) * 0.3))
        # No início explora mais, depois explota mais
        exploration_rate = base_rate - generation_factor
        
        # Ajuste baseado na estagnação
        if self.stagnation_counter > 10:
            # Se estagnado, aumenta exploração
            stagnation_boost = min(0.4, self.stagnation_counter * 0.02)
            exploration_rate += stagnation_boost
        
        # Limites finais
        exploration_rate = max(0.1, min(0.9, exploration_rate))
        self.stats['exploration_vs_exploitation_ratio'] = exploration_rate
    
    def _log_adaptation_status(self) -> None:
        """Loga estado atual das adaptações"""
        print(f"\n🧠 ADAPTATAÇÃO META-LEARNING (Gen {self.generation})")
        print(f"   ├─ Taxa de mutação: {self.adaptation['mutation_rate']:.3f}")
        print(f"   ├─ Taxa de crossover: {self.adaptation['crossover_rate']:.3f}")
        print(f"   ├─ Pressão de seleção: {self.adaptation['selection_pressure']:.1f}")
        print(f"   ├─ Índice estagnação: {self.stagnation_counter}")
        
        if self.diversity_history:
            print(f"   ├─ Diversidade: {self.diversity_history[-1]:.3f}")
        
        print(f"   ├─ Exploração/Explotação: {self.stats['exploration_vs_exploitation_ratio']:.2f}")
        
        if self.promising_regions:
            print(f"   └─ Regiões promissoras: {len(self.promising_regions)}")
    
    # =========== MÉTODOS BAYESIANOS E MONTE CARLO ===========
    
    def bayesian_optimization(self, population: List[Dict], 
                             fitness_key: str = 'fitness',
                             num_samples: int = 100) -> List[Dict]:
        """Aplica otimização bayesiana para gerar indivíduos promissores"""
        # Versão simplificada de otimização bayesiana
        if len(population) < 10:
            return []
        
        # Coleta dados de treino (indivíduos existentes)
        X = [ind['private_key'] for ind in population]
        y = [ind[fitness_key] for ind in population]
        
        # Encontra correlações simples - quais bits parecem importar mais?
        bit_importance = np.zeros(71)
        for i in range(len(X)):
            key = X[i]
            for bit in range(71):
                bit_val = (key >> bit) & 1
                # Correlaciona valor do bit com fitness
                if bit_val == 1:
                    bit_importance[bit] += y[i]
                else:
                    bit_importance[bit] -= y[i]
        
        # Normaliza importâncias
        bit_importance = np.abs(bit_importance)
        bit_importance = bit_importance / np.sum(bit_importance)
        
        # Gera novos indivíduos com base na distribuição aprendida
        new_individuals = []
        
        for _ in range(num_samples):
            # Comece com uma chave existente de boa qualidade
            base_key = random.choice(sorted(population, key=lambda x: x[fitness_key])[:10])['private_key']
            
            # Modifica bits com base na importância aprendida
            for bit in range(71):
                # Quanto mais importante o bit, mais chance de preservá-lo
                # Quanto menos importante, mais chance de flip
                if random.random() < bit_importance[bit] * 0.5:  # Ajusta sensibilidade
                    continue  # Mantem o bit
                else:
                    # Flip o bit
                    base_key ^= (1 << bit)
            
            # Garante que está nos limites
            base_key = max(self.min_key, min(self.max_key, base_key))
            
            # Adiciona como template para criar um novo indivíduo
            new_individuals.append({'private_key': base_key})
        
        return new_individuals
    
    def monte_carlo_exploration(self, best_key: int, 
                               radius: int = 1000, 
                               samples: int = 500) -> List[int]:
        """Explora vizinhança de uma chave usando amostragem de Monte Carlo"""
        results = []
        
        # Diferentes estratégias de amostragem
        strategies = [
            # Amostragem gaussiana de vizinhança
            lambda: int(np.random.normal(best_key, radius)),
            # Flips de 1-5 bits aleatórios
            lambda: best_key ^ (1 << random.randint(0, 70)),
            # Amostragem uniforme na vizinhança
            lambda: best_key + random.randint(-radius, radius),
            # Modificação de byte
            lambda: self._modify_random_byte(best_key),
            # Amostragem logarítmica (permite grandes saltos ocasionais)
            lambda: self._log_sample(best_key, radius)
        ]
        
        # Amostragem usando as diferentes estratégias
        samples_per_strategy = samples // len(strategies)
        
        for strategy in strategies:
            for _ in range(samples_per_strategy):
                key = strategy()
                # Garante limites
                key = max(self.min_key, min(self.max_key, key))
                results.append(key)
        
        # Completa até o número desejado caso necessário
        while len(results) < samples:
            key = strategies[0]()  # Usa primeira estratégia
            key = max(self.min_key, min(self.max_key, key))
            results.append(key)
            
        return results
    
    def _modify_random_byte(self, key: int) -> int:
        """Modifica um byte aleatório na chave"""
        byte_pos = random.randint(0, 8)  # 9 bytes para 71 bits
        byte_mask = 0xFF << (byte_pos * 8)
        new_byte = random.randint(0, 0xFF) << (byte_pos * 8)
        
        # Remove byte antigo e adiciona novo
        return (key & ~byte_mask) | new_byte
    
    def _log_sample(self, key: int, radius: int) -> int:
        """Amostragem em escala logarítmica (permite saltos ocasionais maiores)"""
        # Gera um valor em distribuição log-normal
        log_scale = math.log(radius)
        delta = int(np.random.lognormal(0, log_scale))
        
        # Decide direção
        if random.random() < 0.5:
            return key + delta
        else:
            return key - delta
            
    # =========== MÉTODOS DE SUPORTE ===========
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Retorna os parâmetros atuais otimizados pelo aprendizado"""
        return {
            'mutation_rate': self.adaptation['mutation_rate'],
            'crossover_rate': self.adaptation['crossover_rate'],
            'selection_pressure': self.adaptation['selection_pressure'],
            'exploration_rate': self.stats['exploration_vs_exploitation_ratio'],
            'promising_regions': len(self.promising_regions),
            'stagnation_counter': self.stagnation_counter,
            'generation': self.generation
        }
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas sobre uso de estratégias"""
        return {
            'mutations': self.stats['mutation_applications'],
            'crossovers': self.stats['crossover_applications'],
            'selections': self.stats['selection_applications'],
            'diversity': self.diversity_history[-1] if self.diversity_history else 0
        }


# Função de conveniência para criar uma instância
def get_advanced_strategies(min_key: int, max_key: int, config: Dict[str, Any] = None):
    """Cria e retorna uma instância de estratégias avançadas"""
    return AdvancedGeneticStrategies(min_key, max_key, config)
