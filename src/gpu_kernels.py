#!/usr/bin/env python3
"""
🔥 KERNELS GPU PARA OPERAÇÕES CRIPTOGRÁFICAS
===========================================

Implementação de kernels otimizados para GPU para acelerar
operações criptográficas do Bitcoin Puzzle Solver.

Suporta:
- NVIDIA CUDA via CuPy/PyTorch
- AMD ROCm via ROCm/PyTorch
- Apple MPS via Metal Performance Shaders

Operações aceleradas:
- Multiplicação em curva elíptica secp256k1
- Geração de chaves públicas em lote
- Cálculo de fitness em massa
- Hash SHA-256/RIPEMD-160 em paralelo
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
import time
import hashlib
import random

class GPUKernels:
    """Implementa kernels GPU para operações criptográficas"""
    
    def __init__(self, env_detector):
        """
        Inicializa os kernels GPU de acordo com o ambiente detectado
        
        Args:
            env_detector: Instância de EnvironmentDetector para detecção de hardware
        """
        self.env_detector = env_detector
        self.device = self._detect_and_initialize_device()
        
        # Flags de capacidades
        self.has_cuda = env_detector.cuda_available
        self.has_rocm = env_detector.rocm_available
        self.has_mps = env_detector.mps_available
        self.has_gpu = self.has_cuda or self.has_rocm or self.has_mps
        self.batch_size = self._get_optimal_batch_size()
        
        # Módulos aceleradores carregados dinamicamente
        self.torch = None
        self.cupy = None
        self.accelerated_modules_loaded = self._load_accelerated_modules()
        
        # Compila kernels específicos para o dispositivo atual (quando possível)
        if self.accelerated_modules_loaded:
            self._compile_device_specific_kernels()
            
        print(f"🚀 Kernels GPU inicializados para {self.device}")
    
    def _detect_and_initialize_device(self) -> str:
        """Detecta e inicializa o dispositivo GPU mais adequado"""
        if self.env_detector.cuda_available:
            return "cuda"
        elif self.env_detector.rocm_available:
            return "rocm"
        elif self.env_detector.mps_available:
            return "mps"
        else:
            return "cpu"
    
    def _get_optimal_batch_size(self) -> int:
        """Determina o tamanho de lote ótimo para o hardware disponível"""
        if self.device == "cpu":
            # Para CPU, usamos um valor menor para evitar sobrecarga de memória
            return min(1024, self.env_detector.cpu_info.get('threads', 4) * 256)
        
        # Para GPU, baseamos no tamanho da memória disponível
        if self.device == "cuda" or self.device == "rocm":
            gpu_mem = self.env_detector.gpu_info.get('total_memory', 4) # GB
            # Heurística: aproximadamente 1M elementos por GB de VRAM
            return min(16384, int(gpu_mem * 1024 * 1024))
        
        # Para Apple MPS, valores mais conservadores
        if self.device == "mps":
            return 8192
        
        return 1024  # Valor padrão seguro
    
    def _load_accelerated_modules(self) -> bool:
        """Carrega módulos de aceleração apropriados para o dispositivo atual"""
        try:
            if self.device == "cuda" or self.device == "rocm":
                # Tenta carregar PyTorch com suporte a CUDA/ROCm
                import torch
                self.torch = torch
                
                if self.device == "cuda":
                    # Tenta carregar CuPy para kernels personalizados CUDA
                    import cupy
                    self.cupy = cupy
                
                return True
                
            elif self.device == "mps":
                # Para Apple Silicon, carregamos apenas PyTorch com suporte a MPS
                import torch
                self.torch = torch
                return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                
            return False
        except ImportError:
            print("⚠️ Não foi possível carregar módulos de aceleração GPU")
            return False
    
    def _compile_device_specific_kernels(self):
        """Compila kernels específicos para o tipo de GPU detectado"""
        if self.device == "cuda" and self.cupy:
            # Define kernels CUDA via CuPy para operações criptográficas
            # Estes serão utilizados quando não for possível usar PyTorch diretamente
            
            # Kernel para multiplicação escalar de pontos em curva elíptica secp256k1
            # Esta é uma versão simplificada para demonstração - a implementação real
            # seria muito mais complexa devido ao funcionamento da curva elíptica
            secp256k1_scalar_mul = self.cupy.ElementwiseKernel(
                'uint64 private_key, uint64 base_x, uint64 base_y, uint64 p',
                'uint64 result_x, uint64 result_y',
                '''
                // Este é um placeholder para a implementação real
                // A multiplicação escalar real em secp256k1 requer implementação complexa
                // de operações em curva elíptica
                result_x = (base_x * private_key) % p;
                result_y = (base_y * private_key) % p;
                ''',
                'secp256k1_scalar_mul'
            )
            
            # Na realidade, este kernel precisaria implementar o algoritmo double-and-add
            # para multiplicação escalar em curva elíptica, que é muito mais complexo
            
            self.cuda_kernels = {
                'secp256k1_scalar_mul': secp256k1_scalar_mul,
                # Outros kernels personalizados seriam definidos aqui
            }
    
    # =========== OPERAÇÕES CRIPTOGRÁFICAS ACELERADAS ===========
    
    def batch_generate_pubkeys(self, private_keys: List[int]) -> List[Tuple[int, int]]:
        """
        Gera chaves públicas em lote usando aceleração GPU quando disponível
        
        Args:
            private_keys: Lista de chaves privadas para gerar chaves públicas
            
        Returns:
            Lista de tuplas (x, y) representando pontos da curva elíptica (chaves públicas)
        """
        start_time = time.time()
        result = []
        
        # Se não temos GPU ou módulos acelerados, usamos a implementação CPU
        if not self.has_gpu or not self.accelerated_modules_loaded:
            # Implementação fallback usando biblioteca ecdsa
            import ecdsa
            curve = ecdsa.SECP256k1
            g = curve.generator
            
            for priv_key in private_keys:
                point = g * priv_key
                result.append((point.x(), point.y()))
                
        else:
            # Implementações específicas por dispositivo
            if self.device == "cuda" and self.torch:
                # PyTorch + CUDA implementação
                result = self._torch_cuda_batch_pubkeys(private_keys)
                
            elif self.device == "rocm" and self.torch:
                # PyTorch + ROCm implementação
                result = self._torch_rocm_batch_pubkeys(private_keys)
                
            elif self.device == "mps" and self.torch:
                # PyTorch + MPS (Apple) implementação
                result = self._torch_mps_batch_pubkeys(private_keys)
        
        duration = time.time() - start_time
        print(f"🔑 Geradas {len(private_keys)} chaves públicas em {duration:.4f}s")
        return result

    def _torch_cuda_batch_pubkeys(self, private_keys: List[int]) -> List[Tuple[int, int]]:
        """Implementação específica para CUDA usando PyTorch"""
        # Esta implementação usaria bibliotecas especializadas como:
        # - torchsecpk256k1 (hipotética)
        # - ou implementação personalizada usando operações em tensor
        
        # Código de placeholder - na implementação real precisaríamos:
        # 1. Converter private_keys para tensor CUDA
        # 2. Fazer o cálculo de multiplicação de ponto
        # 3. Retornar os resultados como pontos (x,y)
        
        # Na ausência de uma biblioteca específica, usamos CPU como fallback
        import ecdsa
        curve = ecdsa.SECP256k1
        g = curve.generator
        
        result = []
        for priv_key in private_keys:
            point = g * priv_key
            result.append((point.x(), point.y()))
            
        return result
    
    def _torch_rocm_batch_pubkeys(self, private_keys: List[int]) -> List[Tuple[int, int]]:
        """Implementação específica para AMD ROCm usando PyTorch"""
        # Implementação similar à CUDA, mas com otimizações ROCm
        return self._torch_cuda_batch_pubkeys(private_keys)  # Mesmo código por enquanto
    
    def _torch_mps_batch_pubkeys(self, private_keys: List[int]) -> List[Tuple[int, int]]:
        """Implementação específica para Apple MPS usando PyTorch"""
        # Implementação para Metal Performance Shaders no Apple Silicon
        return self._torch_cuda_batch_pubkeys(private_keys)  # Mesmo código por enquanto

    def batch_calculate_fitness(self, 
                          pubkeys: List[Tuple[int, int]], 
                          target_point: Tuple[int, int]) -> List[float]:
        """
        Calcula fitness em lote para múltiplas chaves públicas
        
        Args:
            pubkeys: Lista de chaves públicas como pontos (x,y)
            target_point: Ponto alvo como (x,y)
            
        Returns:
            Lista de valores de fitness (menor é melhor)
        """
        if not self.has_gpu or not self.accelerated_modules_loaded:
            # Implementação fallback CPU
            return self._cpu_batch_calculate_fitness(pubkeys, target_point)
            
        # Implementações GPU específicas por plataforma
        if self.device == "cuda" and self.torch:
            return self._cuda_batch_calculate_fitness(pubkeys, target_point)
        elif self.device == "rocm" and self.torch:
            return self._rocm_batch_calculate_fitness(pubkeys, target_point)
        elif self.device == "mps" and self.torch:
            return self._mps_batch_calculate_fitness(pubkeys, target_point)
            
        # Fallback caso nenhuma implementação específica seja disponível
        return self._cpu_batch_calculate_fitness(pubkeys, target_point)
    
    def _cpu_batch_calculate_fitness(self, 
                          pubkeys: List[Tuple[int, int]], 
                          target_point: Tuple[int, int]) -> List[float]:
        """Implementação CPU para cálculo de fitness"""
        # Implementação de exemplo - para casos reais seria usada otimização de curva elíptica
        target_x, target_y = target_point
        result = []
        
        for pub_x, pub_y in pubkeys:
            # Calcular diferença em coordenadas
            dx = abs(pub_x - target_x)
            dy = abs(pub_y - target_y)
            
            # Normalização por módulo
            p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
            dx = min(dx, p - dx)
            dy = min(dy, p - dy)
            
            # Ponderação: coordenada x é mais importante que y
            fitness = dx * 0.75 + dy * 0.25
            result.append(fitness)
            
        return result
    
    def _cuda_batch_calculate_fitness(self, 
                          pubkeys: List[Tuple[int, int]], 
                          target_point: Tuple[int, int]) -> List[float]:
        """Implementação CUDA para cálculo de fitness"""
        # A implementação real usaria tensores CUDA para processamento paralelo
        
        if not self.torch:
            return self._cpu_batch_calculate_fitness(pubkeys, target_point)
            
        try:
            import torch
            
            # Preparar tensores CUDA
            target_x, target_y = target_point
            pub_x = torch.tensor([p[0] for p in pubkeys], dtype=torch.float64, device='cuda')
            pub_y = torch.tensor([p[1] for p in pubkeys], dtype=torch.float64, device='cuda')
            
            # Parâmetro da curva
            p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
            p_tensor = torch.tensor(p, dtype=torch.float64, device='cuda')
            
            # Calcular diferença e normalizar pelo módulo
            dx = torch.abs(pub_x - target_x)
            dy = torch.abs(pub_y - target_y)
            
            dx = torch.min(dx, p_tensor - dx)
            dy = torch.min(dy, p_tensor - dy)
            
            # Calcular fitness ponderado
            fitness = dx * 0.75 + dy * 0.25
            
            return fitness.detach().cpu().numpy().tolist()
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo de fitness CUDA: {e}")
            return self._cpu_batch_calculate_fitness(pubkeys, target_point)

    def _rocm_batch_calculate_fitness(self, 
                          pubkeys: List[Tuple[int, int]], 
                          target_point: Tuple[int, int]) -> List[float]:
        """Implementação ROCm para cálculo de fitness"""
        # Mesmo código do CUDA com torch em dispositivo ROCm
        return self._cuda_batch_calculate_fitness(pubkeys, target_point)
    
    def _mps_batch_calculate_fitness(self, 
                          pubkeys: List[Tuple[int, int]], 
                          target_point: Tuple[int, int]) -> List[float]:
        """Implementação MPS (Apple) para cálculo de fitness"""
        # Implementação específica para Apple Metal, muito similar ao CUDA
        # mas com dispositivo 'mps'
        
        if not self.torch:
            return self._cpu_batch_calculate_fitness(pubkeys, target_point)
            
        try:
            import torch
            
            # Preparar tensores MPS
            target_x, target_y = target_point
            pub_x = torch.tensor([p[0] for p in pubkeys], dtype=torch.float64, device='mps')
            pub_y = torch.tensor([p[1] for p in pubkeys], dtype=torch.float64, device='mps')
            
            # Parâmetro da curva
            p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
            p_tensor = torch.tensor(p, dtype=torch.float64, device='mps')
            
            # Calcular diferença e normalizar pelo módulo
            dx = torch.abs(pub_x - target_x)
            dy = torch.abs(pub_y - target_y)
            
            dx = torch.min(dx, p_tensor - dx)
            dy = torch.min(dy, p_tensor - dy)
            
            # Calcular fitness ponderado
            fitness = dx * 0.75 + dy * 0.25
            
            return fitness.detach().cpu().numpy().tolist()
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo de fitness MPS: {e}")
            return self._cpu_batch_calculate_fitness(pubkeys, target_point)

    # =========== OPERAÇÕES DE HASH ACELERADAS ===========
    
    def batch_sha256(self, data_list: List[bytes]) -> List[bytes]:
        """
        Calcula SHA-256 em lote para múltiplos dados
        
        Args:
            data_list: Lista de dados bytes para calcular hash
            
        Returns:
            Lista de hashes SHA-256 resultantes
        """
        start_time = time.time()
        
        if not self.has_gpu or not self.accelerated_modules_loaded:
            # Implementação fallback CPU
            result = self._cpu_batch_sha256(data_list)
        else:
            # Implementações específicas por dispositivo
            if self.device == "cuda" and self.torch:
                result = self._cuda_batch_sha256(data_list)
            elif self.device == "rocm" and self.torch:
                result = self._rocm_batch_sha256(data_list)
            elif self.device == "mps" and self.torch:
                result = self._mps_batch_sha256(data_list)
            else:
                result = self._cpu_batch_sha256(data_list)
        
        duration = time.time() - start_time
        print(f"🔐 Calculados {len(data_list)} hashes SHA-256 em {duration:.4f}s")
        return result
    
    def _cpu_batch_sha256(self, data_list: List[bytes]) -> List[bytes]:
        """Implementação CPU para cálculo de SHA-256 em lote"""
        result = []
        for data in data_list:
            hash_obj = hashlib.sha256()
            hash_obj.update(data)
            result.append(hash_obj.digest())
        return result
    
    def _cuda_batch_sha256(self, data_list: List[bytes]) -> List[bytes]:
        """Implementação CUDA para SHA-256 em lote"""
        # Para GPU, usamos CPU como fallback já que SHA-256 otimizado em GPU
        # requer implementações específicas complexas
        # Em um cenário real, usaríamos bibliotecas como hashcat ou implementações CUDA personalizadas
        try:
            # Tentativa de usar PyTorch para paralelização básica
            import torch
            import concurrent.futures
            
            # Divide o trabalho em chunks para processamento paralelo
            chunk_size = max(1, len(data_list) // (self.env_detector.cpu_info.get('threads', 4) * 2))
            chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
            
            result = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.env_detector.cpu_info.get('threads', 4)) as executor:
                futures = [executor.submit(self._cpu_batch_sha256, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    result.extend(future.result())
            
            return result
            
        except Exception as e:
            print(f"⚠️ Erro no SHA-256 CUDA: {e}")
            return self._cpu_batch_sha256(data_list)
    
    def _rocm_batch_sha256(self, data_list: List[bytes]) -> List[bytes]:
        """Implementação ROCm para SHA-256 em lote"""
        return self._cuda_batch_sha256(data_list)  # Mesmo código por enquanto
    
    def _mps_batch_sha256(self, data_list: List[bytes]) -> List[bytes]:
        """Implementação MPS (Apple) para SHA-256 em lote"""
        return self._cuda_batch_sha256(data_list)  # Mesmo código por enquanto

    # =========== MÉTODOS DE ANÁLISE PROBABILÍSTICA ===========

    def batch_bayesian_inference(self, 
                                samples: List[int], 
                                fitness_values: List[float],
                                exploration_factor: float = 0.2) -> List[float]:
        """
        Realiza inferência bayesiana para orientar a exploração de novas áreas
        
        Args:
            samples: Lista de amostras (chaves privadas)
            fitness_values: Lista de valores de fitness correspondentes
            exploration_factor: Fator de exploração (0.0 a 1.0)
            
        Returns:
            Lista de probabilidades posteriores para cada amostra
        """
        if not self.has_gpu or not self.accelerated_modules_loaded:
            # Fallback para implementação CPU
            return self._cpu_bayesian_inference(samples, fitness_values, exploration_factor)
        
        # Implementações específicas para GPUs
        if self.device == "cuda" and self.torch:
            return self._cuda_bayesian_inference(samples, fitness_values, exploration_factor)
        elif self.device == "rocm" and self.torch:
            return self._rocm_bayesian_inference(samples, fitness_values, exploration_factor)
        elif self.device == "mps" and self.torch:
            return self._mps_bayesian_inference(samples, fitness_values, exploration_factor)
        
        # Fallback para CPU se nenhuma implementação específica for disponível
        return self._cpu_bayesian_inference(samples, fitness_values, exploration_factor)
    
    def _cpu_bayesian_inference(self,
                              samples: List[int],
                              fitness_values: List[float],
                              exploration_factor: float) -> List[float]:
        """Implementação CPU para análise bayesiana"""
        import numpy as np
        
        # Normaliza os fitness values (menor é melhor)
        fitness_array = np.array(fitness_values)
        max_fitness = np.max(fitness_array) + 1.0  # Evita divisão por zero
        
        # Inverte os valores para que maiores sejam melhores
        likelihood = (max_fitness - fitness_array) / max_fitness
        
        # Aplica um fator de exploração para evitar convergência prematura
        # Adiciona alguma aleatoriedade para explorar outras regiões
        exploration = np.random.random(len(likelihood)) * exploration_factor
        posterior = (1.0 - exploration_factor) * likelihood + exploration
        
        # Normaliza para soma = 1.0
        posterior = posterior / np.sum(posterior)
        
        return posterior.tolist()
    
    def _cuda_bayesian_inference(self,
                              samples: List[int],
                              fitness_values: List[float],
                              exploration_factor: float) -> List[float]:
        """Implementação CUDA da análise bayesiana"""
        try:
            import torch
            
            # Prepara tensores CUDA
            fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32, device='cuda')
            
            # Normaliza os fitness values (menor é melhor)
            max_fitness = torch.max(fitness_tensor) + 1.0  # Evita divisão por zero
            
            # Inverte os valores para que maiores sejam melhores
            likelihood = (max_fitness - fitness_tensor) / max_fitness
            
            # Aplica um fator de exploração
            exploration = torch.rand(len(likelihood), device='cuda') * exploration_factor
            posterior = (1.0 - exploration_factor) * likelihood + exploration
            
            # Normaliza para soma = 1.0
            posterior = posterior / torch.sum(posterior)
            
            return posterior.detach().cpu().numpy().tolist()
            
        except Exception as e:
            print(f"⚠️ Erro na inferência bayesiana CUDA: {e}")
            return self._cpu_bayesian_inference(samples, fitness_values, exploration_factor)

    def _rocm_bayesian_inference(self,
                              samples: List[int],
                              fitness_values: List[float],
                              exploration_factor: float) -> List[float]:
        """Implementação ROCm da análise bayesiana"""
        # Mesmo código do CUDA com torch em dispositivo ROCm
        return self._cuda_bayesian_inference(samples, fitness_values, exploration_factor)
    
    def _mps_bayesian_inference(self,
                              samples: List[int],
                              fitness_values: List[float],
                              exploration_factor: float) -> List[float]:
        """Implementação MPS (Apple) da análise bayesiana"""
        try:
            import torch
            
            # Prepara tensores MPS
            fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32, device='mps')
            
            # Normaliza os fitness values (menor é melhor)
            max_fitness = torch.max(fitness_tensor) + 1.0  # Evita divisão por zero
            
            # Inverte os valores para que maiores sejam melhores
            likelihood = (max_fitness - fitness_tensor) / max_fitness
            
            # Aplica um fator de exploração
            exploration = torch.rand(len(likelihood), device='mps') * exploration_factor
            posterior = (1.0 - exploration_factor) * likelihood + exploration
            
            # Normaliza para soma = 1.0
            posterior = posterior / torch.sum(posterior)
            
            return posterior.detach().cpu().numpy().tolist()
            
        except Exception as e:
            print(f"⚠️ Erro na inferência bayesiana MPS: {e}")
            return self._cpu_bayesian_inference(samples, fitness_values, exploration_factor)
            
    def monte_carlo_optimization(self,
                               fitness_func: callable,
                               sample_space: List[int],
                               num_samples: int = 1000,
                               temperature: float = 1.0) -> List[int]:
        """
        Realiza amostragem de Monte Carlo para otimização
        
        Args:
            fitness_func: Função que calcula fitness para uma amostra
            sample_space: Espaço de amostras para exploração
            num_samples: Número de amostras a gerar
            temperature: Temperatura para o algoritmo (maior = mais exploração)
            
        Returns:
            Lista de melhores amostras encontradas
        """
        if len(sample_space) <= num_samples:
            # Se o espaço amostral é menor que o número de amostras solicitadas
            return sample_space
        
        # Usa implementação específica para GPU se disponível, CPU caso contrário
        if self.has_gpu and self.accelerated_modules_loaded and self.torch:
            try:
                import torch
                
                # Seleciona dispositivo para cálculos
                device = self.device
                
                # Converte espaço amostral para tensor
                sample_tensor = torch.tensor(sample_space, device=device)
                
                # Amostragem inicial aleatória
                idx = torch.randperm(len(sample_tensor), device=device)[:num_samples]
                samples = sample_tensor[idx].tolist()
                
                # Calcula fitness para as amostras selecionadas
                fitness_values = [fitness_func(s) for s in samples]
                
                # Aplica inferência bayesiana para refinar a busca
                posterior = self.batch_bayesian_inference(samples, fitness_values, 
                                                         exploration_factor=0.1)
                
                # Converte posterior para tensor
                posterior_tensor = torch.tensor(posterior, device=device)
                
                # Amostra final baseada nas probabilidades posteriores
                # Normaliza novamente por segurança
                posterior_tensor = posterior_tensor / torch.sum(posterior_tensor)
                
                # Amostragem multinomial para selecionar melhores candidatos
                selected_idx = torch.multinomial(posterior_tensor, 
                                              num_samples=min(num_samples, len(posterior_tensor)),
                                              replacement=True)
                
                result = [samples[i] for i in selected_idx.cpu().numpy()]
                return result
                
            except Exception as e:
                print(f"⚠️ Erro na amostragem Monte Carlo: {e}")
                # Fallback para implementação CPU
        
        # Implementação CPU (fallback)
        import random
        import numpy as np
        
        # Amostragem inicial aleatória
        samples = random.sample(sample_space, min(num_samples*2, len(sample_space)))
        
        # Calcula fitness para as amostras
        fitness_values = [fitness_func(s) for s in samples]
        
        # Normaliza fitness (menor é melhor)
        max_fitness = max(fitness_values) + 1.0
        weights = [(max_fitness - f) / max_fitness for f in fitness_values]
        
        # Aplica temperatura para controlar exploração vs. explotação
        weights = [w ** (1.0 / temperature) for w in weights]
        
        # Normaliza pesos
        sum_weights = sum(weights)
        weights = [w / sum_weights for w in weights]
        
        # Amostragem com substituição baseada nos pesos
        result = np.random.choice(samples, size=num_samples, replace=True, p=weights)
        return result.tolist()
# Função global para obter uma instância do GPUKernels
_GPU_KERNELS_INSTANCE = None

def get_gpu_kernels(env_detector=None):
    """Obtém uma instância singleton do GPUKernels"""
    global _GPU_KERNELS_INSTANCE
    
    if _GPU_KERNELS_INSTANCE is None:
        if env_detector is None:
            # Importação circular, mas só ocorre se env_detector não for fornecido
            from environment_detector import get_environment_detector
            env_detector = get_environment_detector()
            
        _GPU_KERNELS_INSTANCE = GPUKernels(env_detector)
        
    return _GPU_KERNELS_INSTANCE

if __name__ == "__main__":
    # Teste básico do módulo
    from environment_detector import get_environment_detector
    env_detector = get_environment_detector()
    
    gpu_kernels = get_gpu_kernels(env_detector)
    
    # Teste simples
    print(f"\nDetectado dispositivo: {gpu_kernels.device}")
    print(f"Tamanho de lote otimizado: {gpu_kernels.batch_size}")
    
    # Teste de geração de chaves
    test_keys = [random.randint(2**30, 2**32) for _ in range(10)]
    pubkeys = gpu_kernels.batch_generate_pubkeys(test_keys)
    
    print(f"\nExemplo de chave pública gerada: {pubkeys[0]}")
