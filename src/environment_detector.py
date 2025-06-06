#!/usr/bin/env python3
"""
🌍 DETECTOR DE AMBIENTE E ACELERAÇÃO GPU
=======================================

Detecta automaticamente o ambiente de execução (local, Kaggle, Colab) 
e configura aceleração GPU/CUDA quando disponível.

Funcionalidades:
- Detecção automática de GPU NVIDIA
- Configuração otimizada para cada ambiente
- Fallback seguro para CPU
- Instalação automática de dependências
"""

import os
import sys
import platform
import subprocess
import multiprocessing as mp
from typing import Dict, List, Optional, Any
import warnings

class EnvironmentDetector:
    """Detecta e configura o ambiente de execução automaticamente"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.cuda_available = self._check_cuda()
        self.gpu_info = self._get_gpu_info()
        self.cpu_info = self._get_cpu_info()
        
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
        elif os.environ.get('JUPYTER_SERVER_URL'):
            return 'jupyter'
        elif 'ipykernel' in sys.modules:
            return 'jupyter'
        else:
            return 'local'
    
    def _check_cuda(self) -> bool:
        """Verifica se CUDA está disponível no sistema"""
        # Método 1: PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
        
        # Método 2: nvidia-ml-py
        try:
            import pynvml
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
        
        return False
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Coleta informações detalhadas da GPU"""
        if not self.cuda_available:
            return {'available': False}
        
        gpu_info = {
            'available': True,
            'count': 0,
            'total_memory_gb': 0,
            'devices': []
        }
        
        # Informações via PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['count']):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info['devices'].append({
                        'id': i,
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                    gpu_info['total_memory_gb'] += props.total_memory / (1024**3)
                
                return gpu_info
        except ImportError:
            pass
        
        # Fallback via nvidia-smi
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info['count'] = len(lines)
                
                for i, line in enumerate(lines):
                    if ',' in line:
                        name, memory = line.split(', ', 1)
                        memory_gb = int(memory) / 1024
                        
                        gpu_info['devices'].append({
                            'id': i,
                            'name': name.strip(),
                            'memory_gb': memory_gb,
                            'compute_capability': 'Unknown'
                        })
                        gpu_info['total_memory_gb'] += memory_gb
        except:
            # Se falhar, pelo menos confirma que CUDA está disponível
            gpu_info['count'] = 1
            gpu_info['devices'] = [{
                'id': 0,
                'name': 'CUDA Device',
                'memory_gb': 8.0,  # Estimativa padrão
                'compute_capability': 'Unknown'
            }]
            gpu_info['total_memory_gb'] = 8.0
        
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
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Configurações otimizadas para cada ambiente"""
        base_config = {
            'max_workers': min(4, self.cpu_info['cores']),
            'batch_size': 1000000,
            'memory_limit_gb': min(4, self.cpu_info['memory_gb'] // 2),
            'use_gpu': self.cuda_available,
            'genetic_population': 1000,
            'timeout_hours': 2
        }
        
        if self.environment == 'kaggle':
            return {
                **base_config,
                'max_workers': min(2, self.cpu_info['cores']),  # Kaggle CPU limitado
                'batch_size': 2000000,
                'memory_limit_gb': 10,  # Kaggle tem ~13GB, deixa margem
                'genetic_population': 2000,
                'timeout_hours': 9,  # Limite do Kaggle
                'use_gpu': self.cuda_available,
                'environment_specific': {
                    'install_packages': ['bitcoin', 'ecdsa'],
                    'gpu_memory_fraction': 0.8
                }
            }
        
        elif self.environment == 'colab':
            return {
                **base_config,
                'max_workers': min(2, self.cpu_info['cores']),
                'batch_size': 1500000, 
                'memory_limit_gb': 8,
                'genetic_population': 1500,
                'timeout_hours': 12,  # Limite do Colab
                'use_gpu': self.cuda_available,
                'environment_specific': {
                    'install_packages': ['bitcoin', 'ecdsa', 'requests'],
                    'gpu_memory_fraction': 0.7
                }
            }
        
        elif self.environment == 'local':
            return {
                **base_config,
                'max_workers': min(self.cpu_info['cores'], 8),
                'batch_size': 3000000,
                'memory_limit_gb': max(4, self.cpu_info['memory_gb'] * 0.6),
                'genetic_population': min(3000, 400 * self.cpu_info['cores']),
                'timeout_hours': None,  # Sem limite local
                'use_gpu': self.cuda_available,
                'environment_specific': {
                    'install_packages': ['bitcoin', 'ecdsa', 'requests', 'numpy'],
                    'gpu_memory_fraction': 0.9
                }
            }
        
        return base_config
    
    def _print_environment_info(self):
        """Exibe informações detalhadas do ambiente"""
        print(f"\n🌍 AMBIENTE DETECTADO: {self.environment.upper()}")
        print("=" * 55)
        
        # Informações de CPU
        print(f"🖥️  CPU: {self.cpu_info['cores']} cores")
        print(f"🧠 RAM: {self.cpu_info['memory_gb']:.1f} GB")
        print(f"🏗️  Arquitetura: {self.cpu_info['architecture']}")
        
        # Informações de GPU
        if self.cuda_available and self.gpu_info['available']:
            print(f"🚀 GPU: ✅ CUDA Disponível")
            print(f"   └─ Dispositivos: {self.gpu_info['count']}")
            
            for gpu in self.gpu_info['devices']:
                print(f"   └─ {gpu['name']}: {gpu['memory_gb']:.1f} GB")
                if gpu['compute_capability'] != 'Unknown':
                    print(f"      └─ Compute: {gpu['compute_capability']}")
        else:
            print(f"🚀 GPU: ❌ Apenas CPU")
        
        # Configurações otimizadas
        config = self.config
        print(f"\n⚙️  CONFIGURAÇÕES OTIMIZADAS:")
        print(f"   └─ Workers: {config['max_workers']}")
        print(f"   └─ Batch size: {config['batch_size']:,}")
        print(f"   └─ População genética: {config['genetic_population']:,}")
        print(f"   └─ Limite memória: {config['memory_limit_gb']} GB")
        
        if config['timeout_hours']:
            print(f"   └─ Timeout: {config['timeout_hours']} horas")
        else:
            print(f"   └─ Timeout: Ilimitado")
        
        print()
    
    def setup_cuda_environment(self) -> bool:
        """Configura o ambiente CUDA otimizado"""
        if not self.cuda_available:
            return False
        
        try:
            # Configurações gerais de CUDA
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Configurações específicas do ambiente
            if self.environment == 'kaggle':
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                warnings.filterwarnings('ignore', category=UserWarning)
            
            # Configurações PyTorch se disponível
            try:
                import torch
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.cuda.empty_cache()
                    print("✅ PyTorch CUDA configurado")
            except ImportError:
                pass
            
            print("✅ Ambiente CUDA configurado com sucesso!")
            return True
            
        except Exception as e:
            print(f"⚠️  Aviso: Erro ao configurar CUDA: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Instala dependências necessárias para o ambiente"""
        packages = self.config.get('environment_specific', {}).get('install_packages', [])
        
        if not packages:
            return True
        
        print(f"📦 Instalando dependências para {self.environment}...")
        
        success = True
        for package in packages:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package, '--quiet'
                ], check=True, capture_output=True)
                print(f"   ✅ {package}")
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️  {package} (falha)")
                success = False
        
        # Instala dependências CUDA se disponível
        if self.cuda_available:
            cuda_packages = []
            
            # nvidia-ml-py para monitoramento GPU
            try:
                import pynvml
            except ImportError:
                cuda_packages.append('nvidia-ml-py')
            
            for package in cuda_packages:
                try:
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', package, '--quiet'
                    ], check=True, capture_output=True)
                    print(f"   ✅ {package} (CUDA)")
                except subprocess.CalledProcessError:
                    print(f"   ⚠️  {package} (CUDA falha)")
        
        return success
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Retorna configuração otimizada para o ambiente atual"""
        return self.config.copy()
    
    def is_gpu_available(self) -> bool:
        """Verifica se GPU está disponível e configurada"""
        return self.cuda_available
    
    def get_gpu_memory_gb(self) -> float:
        """Retorna total de memória GPU disponível em GB"""
        if self.cuda_available and self.gpu_info['available']:
            return self.gpu_info['total_memory_gb']
        return 0.0
# Instância global do detector
_env_detector = None

def get_environment_detector() -> EnvironmentDetector:
    """Retorna a instância global do detector de ambiente"""
    global _env_detector
    if _env_detector is None:
        _env_detector = EnvironmentDetector()
    return _env_detector

# Funções de conveniência
def get_environment_config() -> Dict[str, Any]:
    """Retorna configuração otimizada para o ambiente"""
    return get_environment_detector().get_optimal_config()

def is_cuda_available() -> bool:
    """Verifica se CUDA está disponível"""
    return get_environment_detector().is_gpu_available()

def setup_cuda() -> bool:
    """Configura CUDA se disponível"""
    return get_environment_detector().setup_cuda_environment()

def get_environment() -> str:
    """Retorna o tipo de ambiente atual"""
    return get_environment_detector().environment

def install_dependencies() -> bool:
    """Instala dependências necessárias"""
    return get_environment_detector().install_dependencies()

# Auto-configuração na importação
if __name__ != "__main__":
    detector = get_environment_detector()
    if detector.cuda_available:
        detector.setup_cuda_environment()
