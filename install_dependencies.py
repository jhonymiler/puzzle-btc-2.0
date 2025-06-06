#!/usr/bin/env python3
"""
📦 INSTALADOR DE DEPENDÊNCIAS INTELIGENTE
========================================

Script que detecta o ambiente e instala apenas as dependências necessárias.
Funciona em local, Kaggle, Colab e outros ambientes.
"""

import sys
import subprocess
import os
from typing import List, Dict

def detect_environment() -> str:
    """Detecta o ambiente de execução"""
    if os.path.exists('/kaggle'):
        return 'kaggle'
    elif 'google.colab' in sys.modules:
        return 'colab'
    elif 'COLAB_GPU' in os.environ:
        return 'colab'
    else:
        return 'local'

def install_package(package: str, quiet: bool = True) -> bool:
    """Instala um pacote Python"""
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', package]
        if quiet:
            cmd.append('--quiet')
        
        subprocess.run(cmd, check=True, capture_output=quiet)
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package: str) -> bool:
    """Verifica se um pacote está instalado"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Instala dependências baseado no ambiente"""
    environment = detect_environment()
    
    print(f"🌍 Ambiente detectado: {environment.upper()}")
    print("📦 Verificando e instalando dependências...")
    
    # Dependências básicas necessárias
    basic_packages = {
        'bitcoin': 'bitcoin',
        'ecdsa': 'ecdsa', 
        'requests': 'requests',
        'hashlib': None,  # Built-in
        'multiprocessing': None,  # Built-in
        'json': None,  # Built-in
        'time': None,  # Built-in
        'os': None,  # Built-in
        'random': None,  # Built-in
    }
    
    # Dependências opcionais para melhor performance
    optional_packages = {
        'numpy': 'numpy',
        'psutil': 'psutil',
        'nvidia-ml-py': 'nvidia-ml-py'
    }
    
    # Dependências específicas do ambiente
    environment_packages = {
        'kaggle': [],  # Kaggle já tem a maioria
        'colab': [],   # Colab já tem a maioria  
        'local': ['numpy', 'psutil']  # Local precisa instalar
    }
    
    installed = []
    failed = []
    
    # Instala pacotes básicos
    for import_name, package_name in basic_packages.items():
        if package_name is None:  # Built-in module
            continue
            
        if not check_package(import_name):
            print(f"   📦 Instalando {package_name}...")
            if install_package(package_name):
                installed.append(package_name)
                print(f"   ✅ {package_name}")
            else:
                failed.append(package_name)
                print(f"   ❌ {package_name}")
        else:
            print(f"   ✅ {package_name} (já instalado)")
    
    # Instala pacotes opcionais
    for import_name, package_name in optional_packages.items():
        if not check_package(import_name):
            print(f"   📦 Instalando {package_name} (opcional)...")
            if install_package(package_name):
                installed.append(package_name)
                print(f"   ✅ {package_name}")
            else:
                print(f"   ⚠️  {package_name} (falha - opcional)")
        else:
            print(f"   ✅ {package_name} (já instalado)")
    
    # Verifica CUDA
    cuda_available = check_cuda()
    if cuda_available:
        print("   🚀 CUDA detectado - sistema pronto para GPU!")
    else:
        print("   💻 CUDA não detectado - usando apenas CPU")
    
    # Relatório final
    print(f"\n📊 RELATÓRIO DE INSTALAÇÃO:")
    print(f"   ✅ Instalados: {len(installed)}")
    if installed:
        print(f"      └─ {', '.join(installed)}")
    
    if failed:
        print(f"   ❌ Falharam: {len(failed)}")
        print(f"      └─ {', '.join(failed)}")
    
    print(f"   🚀 GPU: {'Disponível' if cuda_available else 'Não disponível'}")
    print(f"   🌍 Ambiente: {environment}")
    
    return len(failed) == 0

def check_cuda() -> bool:
    """Verifica se CUDA está disponível"""
    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    
    # nvidia-smi
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return True
    except:
        pass
    
    return False

def main():
    """Função principal"""
    print("🚀 INSTALADOR DE DEPENDÊNCIAS - Bitcoin Puzzle 71 Solver")
    print("=" * 60)
    
    success = install_dependencies()
    
    if success:
        print("\n✅ Todas as dependências foram instaladas com sucesso!")
        print("🎯 Sistema pronto para executar o Bitcoin Puzzle 71 Solver")
        print("\nPara executar:")
        print("   python3 main.py --master")
    else:
        print("\n⚠️  Algumas dependências falharam, mas o sistema pode ainda funcionar")
        print("🔄 Tente executar novamente ou instale manualmente")

if __name__ == "__main__":
    main()
