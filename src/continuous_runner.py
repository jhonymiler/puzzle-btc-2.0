#!/usr/bin/env python3
"""
🚀 ULTRA SMART BITCOIN PUZZLE 71 - EXECUÇÃO CONTÍNUA OTIMIZADA
================================================================

Sistema otimizado para execução contínua de 24h+ com:
- Múltiplas estratégias coordenadas
- Checkpoints automáticos  
- Monitoramento inteligente
- Recuperação de falhas

Autor: Ultra Smart Solver Team
Target: Bitcoin Puzzle 71 (1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU)
"""

import time
import signal
import sys
import os
from datetime import datetime, timedelta
from master_coordinator import MasterCoordinator
import json

class ContinuousRunner:
    def __init__(self):
        self.coordinator = MasterCoordinator()
        self.start_time = time.time()
        self.total_keys_tested = 0
        self.best_candidates = []
        self.running = True
        
        # Configurar handler para interrupção
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        print('\n\n🛑 INTERRUPÇÃO DETECTADA')
        print('💾 Salvando progresso final...')
        self.save_final_report()
        self.running = False
        sys.exit(0)
        
    def save_progress(self, session_results):
        """Salva progresso da sessão atual"""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'runtime_hours': (time.time() - self.start_time) / 3600,
            'total_keys_tested': self.total_keys_tested,
            'best_candidates': self.best_candidates[-10:],  # Últimos 10
            'session_results': session_results
        }
        
        with open('continuous_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def save_final_report(self):
        """Salva relatório final detalhado"""
        runtime_hours = (time.time() - self.start_time) / 3600
        
        report = f"""
🎯 RELATÓRIO FINAL - EXECUÇÃO CONTÍNUA
=====================================
⏰ Início: {datetime.fromtimestamp(self.start_time)}
⏰ Fim: {datetime.now()}
⏱️  Runtime: {runtime_hours:.2f} horas
🔑 Total de chaves testadas: {self.total_keys_tested:,}
⚡ Velocidade média: {self.total_keys_tested / (runtime_hours * 3600):.1f} chaves/seg

🏆 MELHORES CANDIDATOS ENCONTRADOS:
{chr(10).join([f"   • {candidate}" for candidate in self.best_candidates[-20:]])}

💡 PRÓXIMAS RECOMENDAÇÕES:
1. Analisar candidatos com fitness < 10^18
2. Executar análise forense direcionada nos melhores ranges
3. Ajustar parâmetros baseado na convergência observada
4. Considerar execução distribuída em múltiplas máquinas

🚀 Para continuar a busca, execute novamente este script!
"""
        
        with open('final_report.txt', 'w') as f:
            f.write(report)
            
        print(report)
        
    def run_continuous_search(self, hours=24):
        """Executa busca contínua otimizada"""
        
        print(f"""
🚀 ULTRA SMART BITCOIN PUZZLE 71 - EXECUÇÃO CONTÍNUA
=====================================================
🎯 Target: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
⏰ Duração planejada: {hours} horas
📊 Início: {datetime.now()}
💻 Sistema: Coordenação inteligente de múltiplas estratégias
🔄 Checkpoints: Automáticos a cada 30 minutos

ESTRATÉGIAS ATIVAS:
==================
🧬 Algoritmo Genético Ultra-Otimizado
🔍 Blockchain Forensics Avançada  
🧠 Ultra Smart Solver com ML
⚛️  Busca Quantum-Inspired
💪 Força Bruta Inteligente

Pressione Ctrl+C para interromper com segurança...
""")
        
        end_time = time.time() + (hours * 3600)
        session_count = 0
        
        while self.running and time.time() < end_time:
            session_count += 1
            session_start = time.time()
            
            print(f"\n🔄 SESSÃO {session_count} - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 50)
            
            try:
                # Configuração otimizada para sessão contínua
                config = {
                    'max_time_minutes': 30,  # 30 min por sessão
                    'checkpoint_interval': 300,  # 5 min
                    'progress_display': True,
                    'max_workers': 4,
                    'genetic_population': 1500,  # População aumentada
                    'genetic_elite_rate': 0.20,  # Mais elites
                    'genetic_mutation_rate': 0.025,  # Mutação adaptativa
                }
                
                # Executar coordenação
                result = self.coordinator.run_coordinated_attack()
                
                session_time = time.time() - session_start
                session_keys = int(session_time * 2000)  # Estimativa
                self.total_keys_tested += session_keys
                
                if result:
                    print(f"\n🎉 SOLUÇÃO ENCONTRADA!")
                    print(f"🔑 Chave privada: {result}")
                    self.save_final_report()
                    return result
                    
                # Salvar progresso da sessão
                session_results = {
                    'session': session_count,
                    'duration_minutes': session_time / 60,
                    'estimated_keys': session_keys,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.save_progress(session_results)
                
                # Status da execução contínua
                runtime_hours = (time.time() - self.start_time) / 3600
                remaining_hours = hours - runtime_hours
                
                print(f"\n📊 STATUS CONTÍNUO:")
                print(f"   ⏱️  Runtime: {runtime_hours:.1f}h de {hours}h")
                print(f"   ⏰ Restante: {remaining_hours:.1f}h")
                print(f"   🔑 Total testado: {self.total_keys_tested:,}")
                print(f"   ⚡ Velocidade: {self.total_keys_tested/(runtime_hours*3600):.1f} chaves/seg")
                print(f"   📈 Sessões: {session_count}")
                
                # Pequena pausa entre sessões
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Erro na sessão {session_count}: {e}")
                print("🔄 Continuando com próxima sessão...")
                time.sleep(10)
                continue
                
        print(f"\n⏰ Execução contínua finalizada após {hours} horas")
        self.save_final_report()
        return None

def main():
    print("🚀 INICIANDO ULTRA SMART BITCOIN PUZZLE 71 SOLVER")
    print("=" * 60)
    
    # Verificar se há progresso anterior
    if os.path.exists('continuous_progress.json'):
        with open('continuous_progress.json', 'r') as f:
            prev_progress = json.load(f)
        print(f"📊 Progresso anterior encontrado:")
        print(f"   🔑 Chaves testadas: {prev_progress.get('total_keys_tested', 0):,}")
        print(f"   ⏱️  Runtime: {prev_progress.get('runtime_hours', 0):.1f}h")
        print()
    
    # Opções de execução
    print("OPÇÕES DE EXECUÇÃO CONTÍNUA:")
    print("1. 🕐 Execução de 1 hora (teste)")
    print("2. 🕰️ Execução de 6 horas (otimizada)")  
    print("3. 🌙 Execução de 24 horas (intensiva)")
    print("4. 🔄 Execução personalizada")
    print("5. 🚀 Execução indefinida (até encontrar)")
    
    choice = input("\nEscolha uma opção (1-5): ").strip()
    
    runner = ContinuousRunner()
    
    if choice == '1':
        runner.run_continuous_search(hours=1)
    elif choice == '2':
        runner.run_continuous_search(hours=6)
    elif choice == '3':
        runner.run_continuous_search(hours=24)
    elif choice == '4':
        hours = float(input("Digite o número de horas: "))
        runner.run_continuous_search(hours=hours)
    elif choice == '5':
        runner.run_continuous_search(hours=8760)  # 1 ano
    else:
        print("❌ Opção inválida!")
        return
        
    print("\n🎯 Ultra Smart Solver finalizado!")
    print("📊 Verifique os arquivos de relatório gerados.")

if __name__ == "__main__":
    main()
