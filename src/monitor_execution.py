#!/usr/bin/env python3
"""
🔍 MONITOR DE EXECUÇÃO - BITCOIN PUZZLE 71
==========================================

Monitor em tempo real do progresso da busca
"""

import time
import json
import os
from datetime import datetime, timedelta

def monitor_progress():
    """Monitor contínuo do progresso"""
    start_time = datetime.now()
    target_duration = timedelta(hours=6)
    
    print("🔍 MONITOR DE EXECUÇÃO - BITCOIN PUZZLE 71")
    print("=" * 60)
    print(f"⏰ Início: {start_time.strftime('%H:%M:%S')}")
    print(f"🎯 Meta: 6 horas de execução")
    print(f"📊 Fim estimado: {(start_time + target_duration).strftime('%H:%M:%S')}")
    print("=" * 60)
    
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        remaining = target_duration - elapsed
        
        # Calcular progresso
        progress_percent = min((elapsed.total_seconds() / target_duration.total_seconds()) * 100, 100)
        
        # Verificar arquivos de progresso
        genetic_progress = check_genetic_progress()
        master_progress = check_master_progress()
        
        # Exibir status
        print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Progresso: {progress_percent:.1f}%")
        print(f"⌛ Tempo decorrido: {str(elapsed).split('.')[0]}")
        print(f"🕐 Tempo restante: {str(remaining).split('.')[0] if remaining.total_seconds() > 0 else 'CONCLUÍDO'}")
        
        if genetic_progress:
            print(f"🧬 Genético: Gen {genetic_progress.get('generation', 0)} | "
                  f"Fitness: {genetic_progress.get('best_fitness', 'N/A')} | "
                  f"Velocidade: {genetic_progress.get('keys_per_second', 0):.0f} k/s")
        
        if master_progress:
            print(f"🎯 Total chaves: {master_progress.get('total_keys_tested', 0):,}")
            
        if remaining.total_seconds() <= 0:
            print("\n🏁 EXECUÇÃO CONCLUÍDA!")
            break
            
        time.sleep(30)  # Atualizar a cada 30 segundos

def check_genetic_progress():
    """Verificar progresso do algoritmo genético"""
    try:
        if os.path.exists('genetic_checkpoint.json'):
            with open('genetic_checkpoint.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return None

def check_master_progress():
    """Verificar progresso do coordenador mestre"""
    try:
        if os.path.exists('master_progress.json'):
            with open('master_progress.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return None

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\n👋 Monitor interrompido pelo usuário")
