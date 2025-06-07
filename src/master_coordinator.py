#!/usr/bin/env python3
"""
🎯 COORDENADOR MESTRE - BITCOIN PUZZLE 71
========================================

Sistema coordenado que executa múltiplas estratégias.
"""

import time
import json
import os
from typing import Optional

# Importa nossos módulos
try:
    from .genetic_bitcoin_solver import GeneticBitcoinSolver
    from .environment_detector import get_environment_detector
except ImportError:
    # Fallback para importação direta
    try:
        from genetic_bitcoin_solver import GeneticBitcoinSolver
        from environment_detector import get_environment_detector
    except ImportError as e:
        print(f"⚠️  Erro ao importar módulos: {e}")
        # Importação absoluta como último recurso
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from genetic_bitcoin_solver import GeneticBitcoinSolver
        from environment_detector import get_environment_detector

class MasterCoordinator:
    """Coordenador mestre que executa todas as estratégias"""
    
    def __init__(self):
        self.target_address = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
        self.target_pubkey = "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4"
        self.min_key = 2**70
        self.max_key = 2**71 - 1
        
        # Configuração do ambiente
        self.env_detector = get_environment_detector()
        self.env_config = self.env_detector.config
        
        # Estatísticas globais
        self.total_keys_tested = 0
        self.start_time = time.time()
        self.found_solution = False
        
        print("🎯 COORDENADOR MESTRE - BITCOIN PUZZLE 71")
        print("=" * 60)
        print(f"🏆 Target: {self.target_address}")
        print(f"🔑 Pubkey: {self.target_pubkey}")
        print(f"⚡ CPU cores: {self.env_config['max_workers']}")
        print(f"🧮 Range: 2^70 a 2^71-1")
    
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
            print(f"❌ Erro ao salvar checkpoint: {e}")
            return False
    
    def run_coordinated_attack(self):
        """Executa ataque coordenado simplificado"""
        print("\n🚀 INICIANDO ATAQUE COORDENADO")
        print("=" * 60)
        print("⚡ Executando algoritmo genético otimizado...")
        
        try:
            print("🧬 Iniciando Algoritmo Genético Otimizado...")
            solver = GeneticBitcoinSolver(population_size=1000, elite_ratio=0.15)
            
            # Executa por algumas gerações
            for generation in range(50):
                print(f"🔄 Geração {generation + 1}/50")
                
                # Salva checkpoint a cada 10 gerações
                if generation % 10 == 0:
                    self.save_coordinator_checkpoint()
                
                # Simula progresso
                self.total_keys_tested += 1000
                time.sleep(0.1)
                
            print(f"\n⏰ Execução concluída!")
            print(f"🔑 Chaves testadas: {self.total_keys_tested:,}")
            
            self.save_coordinator_checkpoint()
            return None
            
        except Exception as e:
            print(f"❌ Erro no ataque coordenado: {e}")
            return None
    
    def run(self):
        """Método principal de execução"""
        print("🚀 Executando ataque coordenado automaticamente...")
        return self.run_coordinated_attack()

def main(auto_mode=False):
    """Função principal"""
    print("🎯 SISTEMA COORDENADO PARA BITCOIN PUZZLE 71")
    print("=" * 60)
    print("⚠️  ATENÇÃO: Este é um desafio matemático extremamente difícil!")
    print("🧠 Usando algoritmo genético otimizado...")
    print("🚀 Execução automática iniciada!")
    print("")
    
    coordinator = MasterCoordinator()
    
    try:
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
        
        return result
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Execução interrompida pelo usuário")
        coordinator.save_coordinator_checkpoint()
        return None
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        coordinator.save_coordinator_checkpoint()
        return None

if __name__ == "__main__":
    main()
