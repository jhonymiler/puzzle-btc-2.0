#!/usr/bin/env python3
"""
🚀 EXECUTOR PRINCIPAL - Bitcoin Puzzle 71 Solver
===============================================

Script principal de execução do sistema de resolução do Bitcoin Puzzle 71.
Este script facilita a execução dos diferentes módulos do sistema.

Uso:
    python3 main.py [opção]
    
Opções:
    --master      - Executa o Master Coordinator
    --continuous  - Executa o Continuous Runner  
    --test        - Executa os testes de validação
    --monitor     - Executa o monitor de execução
    --analyze     - Executa o analisador de resultados
    --install     - Instala dependências automaticamente
    --environment - Mostra informações do ambiente
    --help        - Mostra esta ajuda
"""

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
        
    option = sys.argv[1].lower()
    
    if option in ['--help', '-h']:
        print(__doc__)
        
    elif option == '--install':
        import subprocess
        subprocess.run([sys.executable, 'install_dependencies.py'])
        
    elif option == '--environment':
        from src.environment_detector import get_environment_detector
        detector = get_environment_detector()
        print("✅ Informações do ambiente exibidas acima")
        
    elif option == '--master':
        from src.master_coordinator import MasterCoordinator
        coordinator = MasterCoordinator()
        coordinator.run()
        
    elif option == '--continuous':
        from src.continuous_runner import main as continuous_main
        continuous_main()
        
    elif option == '--test':
        os.chdir('tests')
        os.system('python3 robust_validation_test.py')
        
    elif option == '--monitor':
        from src.monitor_execution import main as monitor_main
        monitor_main()
        
    elif option == '--analyze':
        from src.analyzer import main as analyzer_main
        analyzer_main()
        
    else:
        print(f"❌ Opção inválida: {option}")
        print("Use --help para ver as opções disponíveis")

if __name__ == "__main__":
    main()
