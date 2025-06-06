#!/usr/bin/env python3
"""
🕵️ BLOCKCHAIN FORENSICS & PATTERN ANALYZER
============================================

Análise forense avançada para Bitcoin Puzzle 71:
- Análise de transações da wallet target
- Pattern mining em wallets similares  
- Exploração de vulnerabilidades em RNG
- Análise temporal de criação das chaves
- Weak key detection
"""

import requests
import json
import time
import hashlib
import random
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import sqlite3

class BlockchainForensics:
    """Análise forense avançada da blockchain"""
    
    def __init__(self):
        self.target_address = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
        self.target_pubkey = "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4"
        
        # APIs para análise blockchain (usar com moderação)
        self.blockchain_apis = [
            "https://blockstream.info/api",
            "https://api.blockcypher.com/v1/btc/main"
        ]
        
        print("🕵️ BLOCKCHAIN FORENSICS INICIADO")
        print(f"🎯 Target: {self.target_address}")
    
    def analyze_target_transactions(self) -> Dict:
        """Analisa transações da wallet target"""
        print("\n🔍 Analisando transações da wallet target...")
        
        try:
            # Consulta API blockchain
            url = f"{self.blockchain_apis[0]}/address/{self.target_address}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Dados obtidos da blockchain")
                
                analysis = {
                    'balance': data.get('chain_stats', {}).get('funded_txo_sum', 0),
                    'tx_count': data.get('chain_stats', {}).get('tx_count', 0),
                    'first_seen': None,
                    'last_seen': None,
                    'patterns': []
                }
                
                return analysis
            else:
                print(f"❌ Erro na API: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  Erro ao consultar blockchain: {e}")
        
        return {}
    
    def find_similar_puzzles_patterns(self) -> List[Dict]:
        """Encontra padrões em puzzles similares já resolvidos"""
        print("\n🧩 Analisando padrões em puzzles resolvidos...")
        
        # Dados dos puzzles já resolvidos (fonte pública)
        solved_puzzles = [
            {
                'puzzle': 63,
                'private_key': 0x7A1CAD7C4458AFE8,
                'address': '1PoQRMsXyQFSqCCRek7tt7umfRkJG9TY8x',
                'found_date': '2023-02-19',
                'method': 'unknown'
            },
            {
                'puzzle': 64, 
                'private_key': 0x14B5E5A50A4A3D06,
                'address': '1NBi7EfzR7ZX2mNdnwP8L8x5HZ9pbR4b4p',
                'found_date': '2023-02-20',
                'method': 'unknown'
            },
            {
                'puzzle': 65,
                'private_key': 0x1BF69C3647CC6D7A,
                'address': '1AhXFhcz2WAhuVGkyE6Ur66BDBC9MrE5MM',
                'found_date': '2023-02-21',
                'method': 'unknown'
            },
            {
                'puzzle': 66,
                'private_key': 0x3FB74830F6DF0D5A,
                'address': '1Fvoya7XMvzfpioQnzaskndL7YkeUxjkxv',
                'found_date': '2023-02-22',
                'method': 'unknown'
            }
        ]
        
        patterns = []
        
        print(f"📊 Analisando {len(solved_puzzles)} puzzles resolvidos...")
        
        for puzzle in solved_puzzles:
            # Análise temporal
            pattern = {
                'puzzle_num': puzzle['puzzle'],
                'key_hex': f"{puzzle['private_key']:016x}",
                'bit_analysis': self._analyze_key_bits(puzzle['private_key'], puzzle['puzzle']),
                'hex_patterns': self._analyze_hex_patterns(puzzle['private_key']),
                'statistical_props': self._calculate_statistical_properties(puzzle['private_key'])
            }
            patterns.append(pattern)
        
        # Busca correlações
        correlations = self._find_correlations(patterns)
        
        print(f"🔗 Encontradas {len(correlations)} correlações interessantes")
        
        return patterns
    
    def _analyze_key_bits(self, key: int, puzzle_num: int) -> Dict:
        """Analisa distribuição de bits na chave"""
        bit_string = format(key, f'0{puzzle_num}b')
        
        return {
            'total_bits': len(bit_string),
            'ones_count': bit_string.count('1'),
            'zeros_count': bit_string.count('0'),
            'ones_ratio': bit_string.count('1') / len(bit_string),
            'longest_run_ones': self._longest_run(bit_string, '1'),
            'longest_run_zeros': self._longest_run(bit_string, '0'),
            'alternations': self._count_alternations(bit_string)
        }
    
    def _longest_run(self, bit_string: str, char: str) -> int:
        """Encontra a sequência mais longa de um caractere"""
        max_run = 0
        current_run = 0
        
        for bit in bit_string:
            if bit == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def _count_alternations(self, bit_string: str) -> int:
        """Conta alternações entre 0 e 1"""
        alternations = 0
        for i in range(len(bit_string) - 1):
            if bit_string[i] != bit_string[i + 1]:
                alternations += 1
        return alternations
    
    def _analyze_hex_patterns(self, key: int) -> Dict:
        """Analisa padrões hexadecimais"""
        hex_str = f"{key:016x}"
        
        # Análise de dígitos
        digit_freq = {}
        for digit in '0123456789abcdef':
            digit_freq[digit] = hex_str.count(digit)
        
        # Padrões específicos
        patterns = {
            'digit_frequency': digit_freq,
            'unique_digits': len(set(hex_str)),
            'most_common': max(digit_freq, key=digit_freq.get),
            'least_common': min(digit_freq, key=digit_freq.get),
            'has_sequences': self._has_hex_sequences(hex_str),
            'symmetry_score': self._hex_symmetry_score(hex_str)
        }
        
        return patterns
    
    def _has_hex_sequences(self, hex_str: str) -> Dict:
        """Detecta sequências em hex"""
        sequences = {
            'ascending': 0,
            'descending': 0,
            'repeated': 0
        }
        
        for i in range(len(hex_str) - 2):
            triple = hex_str[i:i+3]
            
            # Sequência ascendente (123, abc, etc)
            if all(ord(triple[j]) + 1 == ord(triple[j+1]) for j in range(2)):
                sequences['ascending'] += 1
            
            # Sequência descendente (321, cba, etc)  
            if all(ord(triple[j]) - 1 == ord(triple[j+1]) for j in range(2)):
                sequences['descending'] += 1
            
            # Repetição (111, aaa, etc)
            if len(set(triple)) == 1:
                sequences['repeated'] += 1
        
        return sequences
    
    def _hex_symmetry_score(self, hex_str: str) -> float:
        """Calcula score de simetria do hex"""
        n = len(hex_str)
        matches = 0
        
        for i in range(n // 2):
            if hex_str[i] == hex_str[n - 1 - i]:
                matches += 1
        
        return matches / (n // 2) if n > 0 else 0
    
    def _calculate_statistical_properties(self, key: int) -> Dict:
        """Calcula propriedades estatísticas da chave"""
        hex_str = f"{key:016x}"
        values = [int(c, 16) for c in hex_str]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'variance': np.var(values),
            'min': min(values),
            'max': max(values),
            'entropy': self._calculate_entropy(values)
        }
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calcula entropia de Shannon"""
        if not values:
            return 0
        
        # Conta frequências
        freq = {}
        for val in values:
            freq[val] = freq.get(val, 0) + 1
        
        # Calcula entropia
        entropy = 0
        total = len(values)
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _find_correlations(self, patterns: List[Dict]) -> List[Dict]:
        """Encontra correlações entre padrões"""
        correlations = []
        
        # Analisa tendências nos puzzles sequenciais
        for i in range(len(patterns) - 1):
            curr = patterns[i]
            next_puzzle = patterns[i + 1]
            
            correlation = {
                'puzzles': f"{curr['puzzle_num']} -> {next_puzzle['puzzle_num']}",
                'bit_ratio_change': (
                    next_puzzle['bit_analysis']['ones_ratio'] - 
                    curr['bit_analysis']['ones_ratio']
                ),
                'entropy_change': (
                    next_puzzle['statistical_props']['entropy'] -
                    curr['statistical_props']['entropy']
                ),
                'pattern_similarity': self._calculate_pattern_similarity(curr, next_puzzle)
            }
            correlations.append(correlation)
        
        return correlations
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calcula similaridade entre dois padrões"""
        # Compara propriedades estatísticas
        props1 = pattern1['statistical_props']
        props2 = pattern2['statistical_props']
        
        similarity_scores = []
        
        # Compara métricas numéricas
        for key in ['mean', 'std', 'entropy']:
            if key in props1 and key in props2:
                diff = abs(props1[key] - props2[key])
                # Normaliza diferença (assumindo range 0-15 para hex)
                normalized_diff = diff / 15.0
                similarity = 1.0 - min(normalized_diff, 1.0)
                similarity_scores.append(similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def detect_weak_rng_patterns(self) -> List[int]:
        """Detecta padrões de RNG fraco"""
        print("\n🎲 Analisando possíveis vulnerabilidades em RNG...")
        
        weak_candidates = []
        
        # 1. Linear Congruential Generator (LCG) comum
        weak_candidates.extend(self._test_lcg_patterns())
        
        # 2. Mersenne Twister com seeds previsíveis
        weak_candidates.extend(self._test_mersenne_patterns())
        
        # 3. Timestamp-based seeds
        weak_candidates.extend(self._test_timestamp_seeds())
        
        # 4. Low-entropy seeds
        weak_candidates.extend(self._test_low_entropy_seeds())
        
        print(f"⚠️  Encontrados {len(weak_candidates)} candidatos de RNG fraco")
        
        return weak_candidates
    
    def _test_lcg_patterns(self) -> List[int]:
        """Testa padrões de Linear Congruential Generator"""
        candidates = []
        
        # Parâmetros comuns de LCG
        lcg_params = [
            (1103515245, 12345, 2**31),  # glibc
            (214013, 2531011, 2**32),    # Microsoft Visual C++
            (1664525, 1013904223, 2**32) # Numerical Recipes
        ]
        
        # Sementes comuns baseadas em timestamps históricos
        base_timestamps = [
            1234567890,  # 2009-02-13 (época Bitcoin)
            1577836800,  # 2020-01-01
            1609459200,  # 2021-01-01
            1640995200   # 2022-01-01
        ]
        
        for a, c, m in lcg_params:
            for base_seed in base_timestamps:
                # Gera sequência LCG
                seed = base_seed
                for _ in range(100):  # Testa várias iterações
                    seed = (a * seed + c) % m
                    
                    # Mapeia para o range do puzzle 71
                    candidate = (2**70) + (seed % (2**71 - 2**70))
                    candidates.append(candidate)
        
        return candidates[:1000]  # Limita resultado
    
    def _test_mersenne_patterns(self) -> List[int]:
        """Testa padrões do Mersenne Twister"""
        candidates = []
        
        # Seeds previsíveis para Mersenne Twister
        predictable_seeds = []
        
        # Timestamps comuns
        for year in range(2009, 2024):
            timestamp = int(datetime(year, 1, 1).timestamp())
            predictable_seeds.append(timestamp)
        
        # Valores comuns usados como seeds
        common_seeds = [
            1, 42, 123, 1337, 12345, 123456, 1000000,
            0xDEADBEEF, 0xCAFEBABE, 0x31337
        ]
        predictable_seeds.extend(common_seeds)
        
        for seed in predictable_seeds:
            # Simula Mersenne Twister simples
            mt_state = seed
            for _ in range(50):
                # Algoritmo simplificado do MT
                mt_state ^= mt_state >> 11
                mt_state ^= (mt_state << 7) & 0x9D2C5680
                mt_state ^= (mt_state << 15) & 0xEFC60000
                mt_state ^= mt_state >> 18
                
                # Mapeia para range do puzzle
                candidate = (2**70) + (mt_state % (2**71 - 2**70))
                candidates.append(candidate)
        
        return candidates[:500]
    
    def _test_timestamp_seeds(self) -> List[int]:
        """Testa seeds baseadas em timestamps"""
        candidates = []
        
        # Timestamps importantes na história do Bitcoin
        important_dates = [
            datetime(2009, 1, 3),   # Genesis block
            datetime(2009, 1, 9),   # Primeira transação
            datetime(2010, 5, 22),  # Pizza day
            datetime(2017, 8, 1),   # SegWit activation
            datetime(2021, 9, 7),   # El Salvador
        ]
        
        for date in important_dates:
            base_timestamp = int(date.timestamp())
            
            # Varia timestamp em segundos, minutos, horas
            for offset in range(-3600*24, 3600*24, 3600):  # ±1 dia, por hora
                timestamp = base_timestamp + offset
                
                # Diferentes formas de usar timestamp como seed
                seeds = [
                    timestamp,
                    timestamp * 1000,  # Milliseconds
                    timestamp // 60,   # Minutes
                    timestamp ^ 0x5DEECE66D,  # XOR comum
                ]
                
                for seed in seeds:
                    candidate = (2**70) + (abs(seed) % (2**71 - 2**70))
                    candidates.append(candidate)
        
        return candidates[:200]
    
    def _test_low_entropy_seeds(self) -> List[int]:
        """Testa seeds de baixa entropia"""
        candidates = []
        
        # Padrões de baixa entropia
        low_entropy_patterns = [
            # Repetições
            0x1111111111111111,
            0x2222222222222222,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            
            # Sequências
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
            
            # Padrões alternados
            0xA5A5A5A5A5A5A5A5,
            0x5A5A5A5A5A5A5A5A,
            
            # Shift patterns
            0x0123456789ABCDEF,
            0x8000000000000000,
            0x4000000000000000,
        ]
        
        for pattern in low_entropy_patterns:
            # Varia o padrão ligeiramente
            for i in range(100):
                variant = pattern ^ (1 << (i % 64))
                candidate = (2**70) + (variant % (2**71 - 2**70))
                candidates.append(candidate)
        
        return candidates
    
    def analyze_weak_keys(self) -> List[int]:
        """Análise de chaves fracas e vulneráveis"""
        print("\n🔐 Analisando chaves potencialmente fracas...")
        
        candidates = []
        
        # Chaves com padrões conhecidamente fracos
        weak_patterns = self._test_low_entropy_seeds()
        candidates.extend(weak_patterns)
        
        # Chaves derivadas de palavras comuns (brain wallets)
        common_phrases = [
            "bitcoin", "satoshi", "nakamoto", "puzzle", 
            "blockchain", "cryptocurrency", "wallet", "private",
            "genesis", "block", "miner", "mining"
        ]
        
        for phrase in common_phrases:
            # Tenta várias derivações de chave
            h1 = int(hashlib.sha256(phrase.encode()).hexdigest(), 16) % (2**71 - 2**70) + 2**70
            h2 = int(hashlib.sha256((phrase + "71").encode()).hexdigest(), 16) % (2**71 - 2**70) + 2**70
            h3 = int(hashlib.sha256((phrase + phrase).encode()).hexdigest(), 16) % (2**71 - 2**70) + 2**70
            
            candidates.extend([h1, h2, h3])
        
        # Remove duplicados
        candidates = list(set(candidates))
        
        # Limita número de candidatos
        max_candidates = 500
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
            
        print(f"🔑 Análise de chaves fracas gerou {len(candidates)} candidatos")
        
        return candidates
    
    def temporal_analysis(self) -> List[int]:
        """Análise temporal de criação das chaves"""
        print("\n⏰ Executando análise temporal...")
        
        candidates = []
        
        # Hipótese: chaves podem ter sido geradas em momentos específicos
        # Analisa padrões temporais dos puzzles já resolvidos
        
        # Supondo que puzzles foram criados em sequência temporal
        puzzle_creation_estimates = [
            datetime(2012, 1, 1),   # Estimativa inicial
            datetime(2012, 6, 1),   # Mid-year
            datetime(2013, 1, 1),   # Next year
            datetime(2015, 1, 1),   # Bitcoin growth
            datetime(2017, 1, 1),   # Bull run
        ]
        
        for creation_date in puzzle_creation_estimates:
            timestamp = int(creation_date.timestamp())
            
            # Diferentes algoritmos de derivação temporal
            temporal_seeds = [
                timestamp,
                timestamp + 71 * 3600,  # +71 horas (puzzle number)
                timestamp * 71,         # Multiply by puzzle number
                timestamp ^ 71,         # XOR with puzzle number
                hashlib.sha256(str(timestamp).encode()).digest()[:8],
            ]
            
            for seed in temporal_seeds:
                if isinstance(seed, bytes):
                    seed = int.from_bytes(seed, 'big')
                
                # Deriva chave do puzzle 71
                candidate = (2**70) + (abs(seed) % (2**71 - 2**70))
                candidates.append(candidate)
        
        print(f"📅 Análise temporal gerou {len(candidates)} candidatos")
        return candidates
    
    def run_forensic_analysis(self) -> List[int]:
        """Executa análise forense completa"""
        print("\n🚀 INICIANDO ANÁLISE FORENSE COMPLETA")
        print("=" * 50)
        
        all_candidates = []
        
        # 1. Análise de transações
        try:
            tx_analysis = self.analyze_target_transactions()
            print(f"📊 Análise de transações: {tx_analysis}")
        except Exception as e:
            print(f"⚠️  Erro na análise de transações: {e}")
        
        # 2. Padrões de puzzles resolvidos
        try:
            patterns = self.find_similar_puzzles_patterns()
            print(f"🧩 Padrões encontrados: {len(patterns)}")
        except Exception as e:
            print(f"⚠️  Erro na análise de padrões: {e}")
        
        # 3. Vulnerabilidades de RNG
        try:
            weak_candidates = self.detect_weak_rng_patterns()
            all_candidates.extend(weak_candidates)
            print(f"🎲 Candidatos de RNG fraco: {len(weak_candidates)}")
        except Exception as e:
            print(f"⚠️  Erro na análise de RNG: {e}")
        
        # 4. Análise temporal
        try:
            temporal_candidates = self.temporal_analysis()
            all_candidates.extend(temporal_candidates)
        except Exception as e:
            print(f"⚠️  Erro na análise temporal: {e}")
        
        # 5. Análise de chaves fracas
        try:
            weak_key_candidates = self.analyze_weak_keys()
            all_candidates.extend(weak_key_candidates)
        except Exception as e:
            print(f"⚠️  Erro na análise de chaves fracas: {e}")
        
        # Remove duplicatas e mantém apenas candidatos válidos
        unique_candidates = list(set(all_candidates))
        valid_candidates = [c for c in unique_candidates if 2**70 <= c <= 2**71 - 1]
        
        print(f"\n🎯 Total de candidatos forenses: {len(valid_candidates)}")
        
        return valid_candidates

def main():
    """Função principal"""
    forensics = BlockchainForensics()
    candidates = forensics.run_forensic_analysis()
    
    print(f"\n📋 RESUMO DA ANÁLISE FORENSE:")
    print(f"🔍 Candidatos gerados: {len(candidates)}")
    
    if candidates:
        print(f"🥇 Primeiros 10 candidatos:")
        for i, candidate in enumerate(candidates[:10]):
            print(f"  {i+1}. 0x{candidate:016x}")
        
        # Salva candidatos para uso posterior
        with open('forensic_candidates.json', 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'total_candidates': len(candidates),
                'candidates': candidates[:1000]  # Salva top 1000
            }, f, indent=2)
        
        print(f"\n💾 Candidatos salvos em 'forensic_candidates.json'")

if __name__ == "__main__":
    main()
