#!/usr/bin/env python3
"""
🌍 DETECTOR DE AMBIENTE E ACELERAÇÃO GPU
=======================================

Detecta automaticamente o ambiente de execução (local, Kaggle, Colab) 
e configura aceleração GPU/CUDA quando disponível.

Funcionalidades:
- Detecção automática de GPU NVIDIA e AMD
- Configuração otimizada para cada ambiente
- Fallback seguro para CPU
- Instalação automática de dependências
- Paralelismo máximo com base no hardware disponível
- Detecção de recursos avançados de computação
"""

import os
import sys
import platform
import subprocess
import multiprocessing as mp
import json
import time
import psutil
from typing import Dict, List, Optional, Any, Tuple
import warnings
import random
import math

class EnvironmentDetector:
    """Detecta e configura o ambiente de execução automaticamente"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.cpu_info = self._get_cpu_info()
        self.ram_info = self._get_ram_info()
        self.cuda_available = self._check_cuda()
        self.rocm_available = self._check_rocm()  # Suporte para GPUs AMD
        self.mps_available = self._check_mps()  # Suporte Metal Performance Shaders (Apple)
        self.gpu_info = self._get_gpu_info()
        self.distributed_available = self._check_distributed()
        
        # Configurações otimizadas por ambiente
        self.config = self._get_environment_config()
        
        # Exibe informações do ambiente
        self._print_environment_info()
    
    def _detect_environment(self) -> str:
        """Detecta o ambiente de execução atual"""
        if os.path.exists('/kaggle'):
            return 'kaggle'
        elif 'google.colab' in sys.modules:
            return 'colab'
        elif 'COLAB_GPU' in os.environ:
            return 'colab'
        elif 'DATABRICKS_RUNTIME_VERSION' in os.environ:
            return 'databricks'
        elif 'AWS_EXECUTION_ENV' in os.environ:
            return 'aws'
        elif 'AZURE_NOTEBOOKS_HOST' in os.environ:
            return 'azure'
        elif os.environ.get('JUPYTER_SERVER_URL'):
            return 'jupyter'
        elif 'ipykernel' in sys.modules:
            return 'jupyter'
        else:
            return 'local'
    
    def _check_cuda(self) -> bool:
        """Verifica se CUDA está disponível no sistema"""
        # Método 1: PyTorch
        if self._is_module_available('torch'):
            torch = self._safe_import('torch')
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                return True
        
        # Método 2: nvidia-ml-py
        if self._is_module_available('pynvml'):
            pynvml = self._safe_import('pynvml')
            if pynvml:
                try:
                    pynvml.nvmlInit()
                    return pynvml.nvmlDeviceGetCount() > 0
                except:
                    pass
        
        # Método 3: nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  timeout=10)
            return result.returncode == 0
        except:
            pass
        
        # Método 4: Verificar diretório de drivers NVIDIA
        try:
            return os.path.exists('/proc/driver/nvidia')
        except:
            pass
        
        return False
    
    def _check_rocm(self) -> bool:
        """Verifica se ROCm (AMD GPU) está disponível no sistema"""
        # Método 1: rocminfo
        try:
            result = subprocess.run(['rocminfo'], 
                                  capture_output=True, 
                                  timeout=10)
            return result.returncode == 0
        except:
            pass
        
        # Método 2: PyTorch com ROCm
        if self._is_module_available('torch'):
            torch = self._safe_import('torch')
            if torch and hasattr(torch, 'hip') and torch.hip.is_available():
                return True
        
        # Método 3: Verificar diretório ROCm
        try:
            return os.path.exists('/opt/rocm')
        except:
            pass
        
        return False

    def _check_mps(self) -> bool:
        """Verifica se Metal Performance Shaders (Apple Silicon) está disponível"""
        if platform.system() == 'Darwin' and 'arm' in platform.processor():
            # Verificar se o ambiente MPS está disponível
            if self._is_module_available('torch'):
                torch = self._safe_import('torch')
                if torch and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                    return torch.backends.mps.is_available()
                    
            # Verificar se a variável de ambiente está definida
            return os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'
        return False

    def _check_distributed(self) -> bool:
        """Verifica se existe suporte para computação distribuída"""
        # Verifica se torch.distributed está disponível
        if self._is_module_available('torch.distributed'):
            return True
        
        # Verifica se ray está instalado
        if self._is_module_available('ray'):
            return True
        
        # Verifica se dask está instalado
        if self._is_module_available('dask'):
            return True
        
        return False
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Coleta informações detalhadas da GPU"""
        # Garante que os atributos existam, mesmo que ainda não tenham sido verificados
        if not hasattr(self, 'cuda_available'):
            self.cuda_available = self._check_cuda()
            
        if not hasattr(self, 'rocm_available'):
            self.rocm_available = self._check_rocm()
            
        if not hasattr(self, 'mps_available'):
            self.mps_available = self._check_mps()
            
        if not (self.cuda_available or self.rocm_available or self.mps_available):
            return {'available': False}
        
        gpu_info = {
            'available': True,
            'count': 0,
            'total_memory_gb': 0,
            'devices': [],
            'type': 'unknown'
        }
        
        # Verifica NVIDIA usando nvidia-smi
        if self.cuda_available:
            self._get_nvidia_gpu_info(gpu_info)
            
        # Verifica AMD usando rocminfo
        if gpu_info['count'] == 0 and self.rocm_available:
            self._get_rocm_gpu_info(gpu_info)
            
        # Verifica Apple Silicon
        if gpu_info['count'] == 0 and self.mps_available:
            self._get_apple_gpu_info(gpu_info)
            
        return gpu_info
        
    def _get_nvidia_gpu_info(self, gpu_info):
        """Coleta informações de GPUs NVIDIA"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', 
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            
            lines = result.stdout.strip().split('\n')
            gpu_info['count'] = len(lines)
            gpu_info['type'] = 'cuda'
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        idx = parts[0].strip()
                        name = parts[1].strip()
                        mem = parts[2].strip()
                        mem_gb = float(mem) / 1024
                        gpu_info['total_memory_gb'] += mem_gb
                        gpu_info['devices'].append({
                            'id': int(idx),
                            'name': name,
                            'memory_gb': mem_gb
                        })
        except:
            # Tenta obter informações básicas
            if self._is_module_available('torch'):
                torch = self._safe_import('torch')
                if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                    try:
                        gpu_info['count'] = torch.cuda.device_count()
                        gpu_info['type'] = 'cuda'
                        gpu_info['total_memory_gb'] = gpu_info['count'] * 8  # Estimativa
                    except:
                        pass
    
    def _get_rocm_gpu_info(self, gpu_info):
        """Coleta informações de GPUs AMD"""
        try:
            result = subprocess.run(['rocminfo', '--devices'], 
                                  capture_output=True, text=True)
            
            # Análise básica da saída do rocminfo
            lines = result.stdout.strip().split('\n')
            devices = [l for l in lines if 'Device Type:' in l and 'GPU' in l]
            gpu_info['count'] = len(devices)
            gpu_info['type'] = 'rocm'
            
            for i, device in enumerate(devices):
                gpu_info['devices'].append({
                    'id': i,
                    'name': f'AMD GPU {i}',
                    'memory_gb': 4  # Valor padrão estimado
                })
                gpu_info['total_memory_gb'] += 4
        except:
            pass
    
    def _get_apple_gpu_info(self, gpu_info):
        """Coleta informações de GPU Apple Silicon"""
        try:
            if platform.system() == 'Darwin' and 'arm' in platform.processor():
                gpu_info['count'] = 1  # MPS geralmente tem apenas 1 dispositivo
                gpu_info['type'] = 'mps'
                
                # Estima memória baseada na RAM total
                if hasattr(self, 'ram_info') and 'total_gb' in self.ram_info:
                    estimated_mem = self.ram_info['total_gb'] / 2  # Estimativa
                else:
                    estimated_mem = 6  # Valor padrão
                    
                gpu_info['devices'].append({
                    'id': 0,
                    'name': 'Apple Silicon GPU',
                    'memory_gb': estimated_mem,
                    'compute_capability': 'mps'
                })
                gpu_info['total_memory_gb'] = estimated_mem
        except ImportError:
            pass
        
        # Fallback para nvidia-smi se PyTorch falhar
        if gpu_info['count'] == 0 and self.cuda_available:
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', 
                                       '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True)
                
                lines = result.stdout.strip().split('\n')
                gpu_info['count'] = len(lines)
                gpu_info['type'] = 'cuda'
                
                for line in lines:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            idx = parts[0].strip()
                            name = parts[1].strip()
                            mem = parts[2].strip()
                            mem_gb = float(mem) / 1024
                            gpu_info['total_memory_gb'] += mem_gb
                            gpu_info['devices'].append({
                                'id': int(idx),
                                'name': name,
                                'memory_gb': mem_gb
                            })
            except:
                pass
        
        # Fallback para rocminfo se PyTorch falhar
        if gpu_info['count'] == 0 and self.rocm_available:
            try:
                result = subprocess.run(['rocminfo', '--devices'], 
                                      capture_output=True, text=True)
                
                # Análise básica da saída do rocminfo
                lines = result.stdout.strip().split('\n')
                devices = [l for l in lines if 'Device Type:' in l and 'GPU' in l]
                gpu_info['count'] = len(devices)
                gpu_info['type'] = 'rocm'
                
                for i, device in enumerate(devices):
                    gpu_info['devices'].append({
                        'id': i,
                        'name': f'AMD GPU {i}',
                        'memory_gb': 4  # Valor padrão estimado
                    })
                    gpu_info['total_memory_gb'] += 4
            except:
                pass
        
        return gpu_info
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Coleta informações da CPU"""
        # Tenta obter informações de memória
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback baseado no ambiente
            if self.environment == 'kaggle':
                memory_gb = 13.0
            elif self.environment == 'colab':
                memory_gb = 12.0
            else:
                memory_gb = 8.0  # Estimativa conservadora
        
        return {
            'cores': mp.cpu_count(),
            'memory_gb': memory_gb,
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version()
        }
    
    def _get_ram_info(self) -> Dict[str, Any]:
        """Coleta informações detalhadas da RAM"""
        try:
            # Usar psutil para obter informações precisas da RAM
            import psutil
            total_memory = psutil.virtual_memory().total
            swap_memory = psutil.swap_memory().total
            
            return {
                'total_gb': total_memory / (1024 ** 3),
                'available_gb': psutil.virtual_memory().available / (1024 ** 3),
                'used_gb': psutil.virtual_memory().used / (1024 ** 3),
                'free_gb': psutil.virtual_memory().free / (1024 ** 3),
                'swap_total_gb': swap_memory / (1024 ** 3),
                'swap_used_gb': psutil.swap_memory().used / (1024 ** 3),
                'swap_free_gb': psutil.swap_memory().free / (1024 ** 3)
            }
        except ImportError:
            # Fallback para ambientes sem psutil
            return {
                'total_gb': 8.0,
                'available_gb': 4.0,
                'used_gb': 2.0,
                'free_gb': 2.0,
                'swap_total_gb': 2.0,
                'swap_used_gb': 1.0,
                'swap_free_gb': 1.0
            }
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Configurações otimizadas para cada ambiente"""
        # Calcular limites de memória seguros
        if hasattr(self, 'ram_info') and 'total_gb' in self.ram_info:
            available_ram = self.ram_info['total_gb']
        else:
            available_ram = self.cpu_info['memory_gb']
        
        # Limite de 60% da RAM para nossos processos
        safe_memory_limit = available_ram * 0.6
        
        # Determinar configuração para GPU
        has_gpu = self.cuda_available or self.rocm_available or self.mps_available
        gpu_memory = 0
        if has_gpu and 'total_memory_gb' in self.gpu_info:
            gpu_memory = self.gpu_info['total_memory_gb']
        
        # Determinar o número ideal de workers com base em CPU e GPU
        if has_gpu:
            # Com GPU, usamos menos threads de CPU
            ideal_workers = min(max(2, self.cpu_info['cores'] // 2), 8)
        else:
            # Sem GPU, usamos mais threads de CPU
            ideal_workers = min(max(4, self.cpu_info['cores'] - 2), 16)
        
        # Configuração base otimizada
        base_config = {
            'max_workers': ideal_workers,
            'batch_size': 3000000,
            'memory_limit_gb': safe_memory_limit,
            'use_gpu': has_gpu,
            'gpu_type': self.gpu_info.get('type', 'none') if has_gpu else 'none',
            'genetic_population': int(3000 if has_gpu else 1000),
            'genetic_generations': 1000,
            'genetic_elite_rate': 0.15,
            'genetic_diversity_threshold': 0.5,
            'use_distributed': self.distributed_available,
            'parallel_strategies': {
                'batch_regions': True,
                'multi_algorithm': True,
                'adaptive_exploration': True
            },
            'timeout_hours': None,  # Sem limite por padrão
            'checkpoint_interval_minutes': 5,
            'monte_carlo_samples': 10000 if has_gpu else 1000,
            'bayesian_optimization': has_gpu,
            'use_multi_precision': has_gpu,
            'advanced_exploration': {
                'pattern_detection': True,
                'entropy_weighting': True,
                'adaptive_mutation': True,
                'parallel_islands': True if self.cpu_info['cores'] > 8 else False,
                'specialized_operators': True
            },
            'evolutionary_strategies': [
                'standard',
                'differential_evolution' if has_gpu else None,
                'covariance_matrix_adaptation' if has_gpu else None,
                'island_model',
                'microbial'
            ]
        }
        
        # Remover valores None da lista de estratégias evolutivas
        base_config['evolutionary_strategies'] = [
            s for s in base_config['evolutionary_strategies'] if s is not None
        ]
        
        # Configurações específicas para cada ambiente
        if self.environment == 'kaggle':
            return {
                **base_config,
                'max_workers': min(8, self.cpu_info['cores']),
                'batch_size': int(4000000 if has_gpu else 2000000),
                'memory_limit_gb': min(10, safe_memory_limit),
                'genetic_population': int(4000 if has_gpu else 2000),
                'timeout_hours': 9,  # Limite do Kaggle
                'environment_specific': {
                    'gpu_memory_fraction': 0.9,
                    'optimize_for_kaggle': True,
                }
            }
        
        elif self.environment == 'colab':
            return {
                **base_config,
                'max_workers': min(6, self.cpu_info['cores']),
                'batch_size': int(5000000 if has_gpu else 2000000),
                'memory_limit_gb': min(10, safe_memory_limit),
                'genetic_population': int(5000 if has_gpu else 2000),
                'timeout_hours': 11.5,  # Limite do Colab com margem
                'environment_specific': {
                    'gpu_memory_fraction': 0.95,
                    'optimize_for_colab': True,
                    'tpu_available': 'COLAB_TPU_ADDR' in os.environ
                }
            }
            
        elif self.environment == 'jupyter':
            return {
                **base_config,
                'max_workers': min(max(4, self.cpu_info['cores'] // 2), 16),
                'batch_size': 3000000,
                'memory_limit_gb': safe_memory_limit,
                'genetic_population': int(3000 if has_gpu else 1500),
                'checkpoint_interval_minutes': 10,
                'environment_specific': {
                    'interactive': True,
                    'progress_bars': True,
                }
            }
            
        elif self.environment == 'databricks':
            return {
                **base_config,
                'max_workers': min(self.cpu_info['cores'] - 2, 32),  # Databricks tem muitos cores
                'batch_size': 10000000,
                'memory_limit_gb': safe_memory_limit * 0.8,  # Spark precisa de memória
                'genetic_population': 5000,
                'use_distributed': True,
                'environment_specific': {
                    'use_spark': True,
                    'optimize_for_databricks': True
                }
            }
            
        elif self.environment == 'aws':
            # Idealmente otimizado para instâncias AWS específicas
            return {
                **base_config,
                'max_workers': self.cpu_info['cores'] - 1,
                'batch_size': 5000000,
                'memory_limit_gb': safe_memory_limit * 0.9,
                'genetic_population': int(5000 if has_gpu else 2000),
                'environment_specific': {
                    'optimize_for_aws': True
                }
            }
        
        # Ambiente local - mais conservador com recursos
        return {
            **base_config,
            'max_workers': min(max(2, self.cpu_info['cores'] - 2), 8),  # Deixar cores para o sistema
            'batch_size': int(2000000 if has_gpu else 1000000),  
            'memory_limit_gb': safe_memory_limit * 0.8,  # Conservador com memória
            'genetic_population': int(3000 if has_gpu else 1000),
            'checkpoint_interval_minutes': 5,
            'monte_carlo_samples': 5000 if has_gpu else 500,  # Menos amostras
            'environment_specific': {
                'interactive': True,
                'optimize_for_local': True,
                'conserve_resources': True
            }
        }
    
    def _print_environment_info(self):
        """Exibe informações sobre o ambiente detectado"""
        print(f"\n🌍 AMBIENTE DETECTADO: {self.environment.upper()}")
        print("=" * 55)
        print(f"🖥️  CPU: {self.cpu_info['cores']} cores")
        
        if hasattr(self, 'ram_info') and 'total_gb' in self.ram_info:
            print(f"🧠 RAM: {self.ram_info['total_gb']:.1f} GB")
        else:
            print(f"🧠 RAM: {self.cpu_info['memory_gb']:.1f} GB")
            
        print(f"🏗️  Arquitetura: {self.cpu_info['architecture']}")
        
        if self.cuda_available:
            print(f"🚀 GPU: NVIDIA {self.gpu_info['count']} ({self.gpu_info['total_memory_gb']:.1f} GB)")
        elif self.rocm_available:
            print(f"🚀 GPU: AMD {self.gpu_info['count']} ({self.gpu_info['total_memory_gb']:.1f} GB)")
        elif self.mps_available:
            print(f"🚀 GPU: Apple Silicon ({self.gpu_info.get('total_memory_gb', 0):.1f} GB)")
        else:
            print("🚀 GPU: ❌ Apenas CPU")
        
        print(f"\n⚙️  CONFIGURAÇÕES OTIMIZADAS:")
        print(f"   └─ Workers: {self.config['max_workers']}")
        print(f"   └─ Batch size: {self.config['batch_size']:,}")
        print(f"   └─ População genética: {self.config['genetic_population']:,}")
        print(f"   └─ Limite memória: {self.config['memory_limit_gb']} GB")
        
        timeout = self.config.get('timeout_hours')
        if timeout:
            print(f"   └─ Timeout: {timeout} horas")
        else:
            print("   └─ Timeout: Ilimitado")
        
        print()

    def setup_cuda_environment(self):
        """Configura ambiente CUDA/GPU para máximo desempenho"""
        if not (self.cuda_available or self.rocm_available or self.mps_available):
            print("⚠️  GPU não detectada. Usando apenas CPU.")
            return False
        
        # Verifica se PyTorch está disponível
        if not self._is_module_available('torch'):
            print("⚠️  PyTorch não encontrado. Necessário para aceleração GPU.")
            return False
            
        # Importa PyTorch de forma segura
        torch = self._safe_import('torch')
        if not torch:
            return False
        
        if self.cuda_available:
            print("🔧 Configurando ambiente CUDA...")
            try:
                # Definir o número máximo de threads para operações de CPU
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(self.config['max_workers'])
                
                # Configurar cudnn para benchmark
                if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                
                # Tentativa de liberar memória da GPU
                if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                # Verificar se a configuração foi bem-sucedida
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    # Teste rápido para verificar se a GPU está funcionando
                    try:
                        x = torch.Tensor([1.0, 2.0, 3.0]).to(device)
                        result = x + 1.0
                        
                        # Mostrar informações CUDA
                        print(f"✅ GPU CUDA disponível: {torch.cuda.get_device_name(0)}")
                        return True
                    except Exception as e:
                        print(f"⚠️ Erro ao testar GPU: {e}")
                
            except Exception as e:
                print(f"⚠️  Erro ao configurar PyTorch+CUDA: {e}")
        
        elif self.rocm_available:
            print("🔧 Configurando ambiente ROCm para AMD GPU...")
            try:
                if hasattr(torch, 'hip') and torch.hip.is_available():
                    print("✅ GPU AMD ROCm disponível")
                    return True
            except Exception as e:
                print(f"⚠️  Erro ao configurar ROCm: {e}")
        
        elif self.mps_available:
            print("🔧 Configurando Metal Performance Shaders para Apple Silicon...")
            try:
                if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Configurar para usar MPS
                    device = torch.device("mps")
                    print("✅ Apple Silicon MPS disponível")
                    return True
            except Exception as e:
                print(f"⚠️  Erro ao configurar MPS: {e}")
        
        return False
    
    def optimize_search_parallelism(self, search_range_start: int, search_range_end: int):
        """Otimiza a paralelização da busca baseado no ambiente"""
        range_size = search_range_end - search_range_start
        
        # Detectar número ideal de divisões para paralelismo
        if hasattr(self, 'cuda_available') and hasattr(self, 'rocm_available'):
            if self.cuda_available or self.rocm_available:
                # Com GPU: menos divisões, chunks maiores
                gpu_count = self.gpu_info.get('count', 1) if hasattr(self, 'gpu_info') else 1
                ideal_chunks = max(gpu_count * 2, self.config['max_workers'])
            else:
                # Apenas CPU: mais divisões, chunks menores
                ideal_chunks = self.config['max_workers'] * 4
        else:
            # Configuração padrão se não tivermos informações de GPU
            ideal_chunks = self.config['max_workers'] * 2
        
        # Ajustar com base no tamanho do espaço de busca
        if range_size > 2**60:  # Espaço extremamente grande
            chunks = ideal_chunks * 2
        else:
            chunks = ideal_chunks
        
        chunk_size = max(1, range_size // chunks)
        
        # Calcular divisões para algoritmos de busca em paralelo
        regions = []
        for i in range(chunks):
            start = search_range_start + (i * chunk_size)
            end = start + chunk_size if i < chunks - 1 else search_range_end
            regions.append((start, end))
        
        # Configuração de paralelismo
        parallelism_config = {
            'regions': regions,
            'chunk_size': chunk_size,
            'chunks': chunks,
            'max_workers': self.config['max_workers'],
            'optimal_batch_size': self._calculate_optimal_batch_size(),
            'exploration_strategy': self._get_optimal_exploration_strategy(),
            'parallel_execution_mode': self._get_optimal_parallel_mode()
        }
        
        return parallelism_config
        
    def _calculate_optimal_batch_size(self) -> int:
        """Calcula o tamanho de lote ótimo com base nos recursos disponíveis"""
        # Considera memória e processadores disponíveis
        if hasattr(self, 'ram_info') and self.ram_info['total_gb'] > 0:
            mem_gb = self.ram_info['total_gb']
        else:
            mem_gb = self.config['memory_limit_gb']
        
        # GPU geralmente consegue processar mais chaves por lote
        if hasattr(self, 'cuda_available') and self.cuda_available:
            base_size = 5000000
        else:
            base_size = 2000000
            
        # Ajuste baseado na memória
        if mem_gb > 32:
            return int(base_size * 2)
        elif mem_gb > 16:
            return int(base_size * 1.5)
        elif mem_gb > 8:
            return base_size
        else:
            return int(base_size * 0.5)
    
    def _get_optimal_exploration_strategy(self) -> str:
        """Determina a melhor estratégia de exploração para o ambiente"""
        # Com GPU, favorece exploração mais ampla e agressiva
        if hasattr(self, 'cuda_available') and self.cuda_available:
            return "adaptive_hybrid"
        # Com muitos cores, favorece exploração em ilhas
        elif self.cpu_info['cores'] >= 16:
            return "island_model"
        # CPU com cores moderados
        elif self.cpu_info['cores'] >= 8:
            return "differential_evolution"
        # Hardware limitado
        else:
            return "standard"
    
    def _get_optimal_parallel_mode(self) -> str:
        """Determina o modo de paralelismo ótimo para o hardware"""
        # Com GPU, favorece paralelismo de dados
        if hasattr(self, 'cuda_available') and self.cuda_available:
            return "data_parallel"
        # Verifica ambiente distribuído
        elif hasattr(self, 'distributed_available') and self.distributed_available:
            return "distributed"
        # Multi-CPU
        elif self.cpu_info['cores'] > 4:
            return "process_pool"
        # Hardware limitado
        else:
            return "thread_pool"
    
    def get_optimal_genetic_params(self, difficulty_level: int = 71) -> Dict[str, Any]:
        """
        Retorna parâmetros otimizados para o algoritmo genético
        baseado no ambiente detectado e no nível de dificuldade do puzzle
        
        Args:
            difficulty_level: O nível de dificuldade do puzzle (71 por padrão)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        # Configurações base
        has_gpu = (self.cuda_available or self.rocm_available or self.mps_available) 
                    
        # População deve ser maior em GPUs poderosas
        if has_gpu:
            base_population = 3000
            # Se temos muita VRAM, aumenta população
            if hasattr(self, 'gpu_info') and self.gpu_info.get('total_memory_gb', 0) > 8:
                base_population = 5000
        else:
            # Em CPU, população menor mas ainda adequada
            base_population = 1000
            
            # Se temos muitos cores, podemos aumentar um pouco
            if hasattr(self, 'cpu_info') and self.cpu_info['cores'] > 8:
                base_population = 1500
        
        # Ajustes para o nível de dificuldade
        difficulty_factor = min(2.0, difficulty_level / 35.0)  # Normalizado por dificuldade 35
        
        # Calcula população final
        final_population = int(base_population * difficulty_factor)
        
        # Taxa de mutação - maior para dificuldades mais altas
        mutation_rate = 0.01 + (difficulty_level / 1000)
        
        # Razão de elite - menor para dificuldades mais altas
        elite_ratio = max(0.05, 0.15 - (difficulty_level / 1000))
        
        # Número de gerações - maior para dificuldades mais altas
        generations = 500 + (difficulty_level * 10)
        
        # Retorna parâmetros otimizados
        return {
            'population_size': final_population,
            'mutation_rate': mutation_rate,
            'elite_ratio': elite_ratio,
            'crossover_rate': 0.8,
            'generations': int(generations),
            'difficulty_level': difficulty_level,
            'optimize_for_gpu': has_gpu,
            'threads': self.config['max_workers'],
            'batch_size': self._calculate_optimal_batch_size()
        }
    
    def _is_module_available(self, module_name: str) -> bool:
        """Verifica se um módulo Python está disponível"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _safe_import(self, module_name: str):
        """Importa um módulo de forma segura"""
        try:
            return __import__(module_name)
        except ImportError:
            return None

# Instância singleton do detector de ambiente
_ENVIRONMENT_DETECTOR_INSTANCE = None

def get_environment_detector():
    """Obtém uma instância única do detector de ambiente"""
    global _ENVIRONMENT_DETECTOR_INSTANCE
    
    if _ENVIRONMENT_DETECTOR_INSTANCE is None:
        _ENVIRONMENT_DETECTOR_INSTANCE = EnvironmentDetector()
        
    return _ENVIRONMENT_DETECTOR_INSTANCE

if __name__ == "__main__":
    # Teste do detector de ambiente
    detector = get_environment_detector()
    print(f"Ambiente detectado: {detector.environment}")
    print(f"CPU cores: {detector.cpu_info['cores']}")
    print(f"RAM total: {detector.ram_info['total_gb']:.1f} GB")
    print(f"GPU disponível: {detector.cuda_available or detector.rocm_available or detector.mps_available}")
    if detector.cuda_available:
        print(f"GPU CUDA: {detector.gpu_info['count']} dispositivo(s)")
    elif detector.rocm_available:
        print(f"GPU ROCm: {detector.gpu_info['count']} dispositivo(s)")
    elif detector.mps_available:
        print(f"GPU Apple MPS disponível")
