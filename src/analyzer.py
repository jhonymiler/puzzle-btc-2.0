#!/usr/bin/env python3
"""
🔍 ANALISADOR DE RESULTADOS - ULTRA SMART BITCOIN PUZZLE 71 SOLVER
===================================================================

Sistema de análise avançada dos resultados para:
- Identificar padrões nos candidatos encontrados
- Otimizar parâmetros baseado na performance
- Sugerir melhorias nas estratégias
- Avaliar convergência dos algoritmos

Uso: python analyzer.py
"""

import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

class ResultAnalyzer:
    def __init__(self):
        self.progress_data = self.load_progress_data()
        self.genetic_data = self.load_genetic_data()
        
    def load_progress_data(self):
        """Carrega dados de progresso contínuo"""
        try:
            if os.path.exists('continuous_progress.json'):
                with open('continuous_progress.json', 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
        
    def load_genetic_data(self):
        """Carrega dados do checkpoint genético"""
        try:
            if os.path.exists('genetic_checkpoint.json'):
                with open('genetic_checkpoint.json', 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
        
    def analyze_candidate_patterns(self):
        """Analisa padrões nos candidatos encontrados"""
        if not self.progress_data or not self.progress_data.get('best_candidates'):
            print("❌ Sem candidatos para análise")
            return
            
        candidates = self.progress_data['best_candidates']
        print(f"🔍 ANÁLISE DE PADRÕES EM {len(candidates)} CANDIDATOS")
        print("=" * 60)
        
        # Converter para valores numéricos
        values = []
        for candidate in candidates:
            try:
                if isinstance(candidate, str) and 'x' in candidate:
                    val = int(candidate.split('x')[1], 16)
                elif isinstance(candidate, str):
                    val = int(candidate, 16)
                else:
                    val = int(candidate)
                values.append(val)
            except:
                continue
                
        if not values:
            print("❌ Não foi possível converter candidatos")
            return
            
        values = np.array(values)
        
        # Estatísticas básicas
        print(f"📊 ESTATÍSTICAS DOS CANDIDATOS:")
        print(f"   🔢 Quantidade: {len(values)}")
        print(f"   📈 Máximo: {hex(int(np.max(values)))}")
        print(f"   📉 Mínimo: {hex(int(np.min(values)))}")
        print(f"   📊 Média: {hex(int(np.mean(values)))}")
        print(f"   📐 Desvio: {hex(int(np.std(values)))}")
        print()
        
        # Análise de distribuição no range do puzzle 71
        puzzle_71_min = 2**70
        puzzle_71_max = 2**71 - 1
        
        valid_candidates = values[(values >= puzzle_71_min) & (values <= puzzle_71_max)]
        print(f"✅ CANDIDATOS VÁLIDOS: {len(valid_candidates)}/{len(values)}")
        
        if len(valid_candidates) > 0:
            # Posições relativas no range
            positions = ((valid_candidates - puzzle_71_min) / (puzzle_71_max - puzzle_71_min)) * 100
            
            print(f"📍 DISTRIBUIÇÃO NO RANGE:")
            print(f"   🔽 Posição mínima: {np.min(positions):.3f}%")
            print(f"   🔼 Posição máxima: {np.max(positions):.3f}%")
            print(f"   📊 Posição média: {np.mean(positions):.3f}%")
            print()
            
            # Análise de clusters
            if len(valid_candidates) > 1:
                distances = np.diff(np.sort(valid_candidates))
                avg_distance = np.mean(distances)
                min_distance = np.min(distances)
                
                print(f"🎯 ANÁLISE DE CLUSTERING:")
                print(f"   📏 Distância média: {avg_distance:.0f}")
                print(f"   📏 Distância mínima: {min_distance:.0f}")
                
                # Identificar clusters (distância < 10% da média)
                cluster_threshold = avg_distance * 0.1
                clusters = []
                current_cluster = [valid_candidates[0]]
                
                for i in range(1, len(valid_candidates)):
                    if distances[i-1] < cluster_threshold:
                        current_cluster.append(valid_candidates[i])
                    else:
                        if len(current_cluster) > 1:
                            clusters.append(current_cluster)
                        current_cluster = [valid_candidates[i]]
                        
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                    
                print(f"   🔗 Clusters encontrados: {len(clusters)}")
                for i, cluster in enumerate(clusters, 1):
                    center = np.mean(cluster)
                    pos = ((center - puzzle_71_min) / (puzzle_71_max - puzzle_71_min)) * 100
                    print(f"      Cluster {i}: {len(cluster)} candidatos em {pos:.3f}%")
                print()
        
    def analyze_genetic_performance(self):
        """Analisa performance do algoritmo genético"""
        if not self.genetic_data:
            print("❌ Sem dados genéticos para análise")
            return
            
        print("🧬 ANÁLISE DE PERFORMANCE GENÉTICA")
        print("=" * 50)
        
        # Estatísticas básicas
        generation = self.genetic_data.get('generation', 0)
        best_fitness = self.genetic_data.get('best_fitness', 0)
        keys_per_sec = self.genetic_data.get('keys_per_second', 0)
        mutation_rate = self.genetic_data.get('mutation_rate', 0)
        
        print(f"📈 EVOLUÇÃO ATUAL:")
        print(f"   🔢 Geração: {generation}")
        print(f"   🏆 Melhor fitness: {best_fitness}")
        print(f"   ⚡ Velocidade: {keys_per_sec:.1f} chaves/seg")
        print(f"   🎲 Taxa de mutação: {mutation_rate:.1f}%")
        print()
        
        # Análise de convergência
        if generation > 50:
            convergence_rate = generation / best_fitness if best_fitness > 0 else 0
            print(f"📊 ANÁLISE DE CONVERGÊNCIA:")
            print(f"   📈 Taxa de convergência: {convergence_rate:.2e}")
            
            if generation > 100:
                if mutation_rate < 2.0:
                    print("   ⚠️  Taxa de mutação baixa - pode estar convergindo prematuramente")
                elif mutation_rate > 5.0:
                    print("   ⚠️  Taxa de mutação alta - pode estar explorando demais")
                else:
                    print("   ✅ Taxa de mutação adequada")
                    
            print()
        
    def generate_optimization_suggestions(self):
        """Gera sugestões de otimização baseadas na análise"""
        print("🚀 SUGESTÕES DE OTIMIZAÇÃO")
        print("=" * 40)
        
        suggestions = []
        
        # Baseado na performance genética
        if self.genetic_data:
            generation = self.genetic_data.get('generation', 0)
            keys_per_sec = self.genetic_data.get('keys_per_second', 0)
            mutation_rate = self.genetic_data.get('mutation_rate', 0)
            
            if keys_per_sec < 1500:
                suggestions.append("🔧 Reduzir população genética para aumentar velocidade")
                
            if keys_per_sec > 2500:
                suggestions.append("🔧 Aumentar população genética para melhor exploração")
                
            if generation > 200 and mutation_rate < 3.0:
                suggestions.append("🔧 Aumentar diversidade genética com mais mutação")
                
            if mutation_rate > 6.0:
                suggestions.append("🔧 Reduzir taxa de mutação para melhor convergência")
        
        # Baseado nos candidatos encontrados
        if self.progress_data and self.progress_data.get('best_candidates'):
            num_candidates = len(self.progress_data['best_candidates'])
            
            if num_candidates < 10:
                suggestions.append("🔧 Aumentar tempo de execução para mais candidatos")
                
            if num_candidates > 100:
                suggestions.append("🔧 Implementar filtros mais rigorosos para candidatos")
        
        # Baseado no runtime
        if self.progress_data:
            runtime_hours = self.progress_data.get('runtime_hours', 0)
            total_keys = self.progress_data.get('total_keys_tested', 0)
            
            if runtime_hours > 6 and total_keys < 1000000:
                suggestions.append("🔧 Otimizar algoritmos para maior throughput")
                
            if runtime_hours < 1:
                suggestions.append("🔧 Executar por mais tempo para resultados consistentes")
        
        # Sugestões estratégicas gerais
        suggestions.extend([
            "🔧 Focar busca em regiões com maior densidade de candidatos",
            "🔧 Implementar busca direcionada baseada em padrões encontrados",
            "🔧 Considerar análise temporal mais profunda",
            "🔧 Explorar vulnerabilidades RNG identificadas",
            "🔧 Executar em múltiplas máquinas para paralelização"
        ])
        
        # Exibir sugestões
        for i, suggestion in enumerate(suggestions[:10], 1):
            print(f"{i:2d}. {suggestion}")
        print()
        
    def generate_optimization_script(self):
        """Gera script otimizado baseado na análise"""
        print("📜 GERANDO CONFIGURAÇÃO OTIMIZADA...")
        
        # Configuração base otimizada
        config = {
            "genetic_population": 1500,
            "genetic_elite_rate": 0.20,
            "genetic_mutation_rate": 0.03,
            "genetic_crossover_rate": 0.85,
            "max_time_minutes": 60,
            "checkpoint_interval": 300,
            "max_workers": 4
        }
        
        # Ajustes baseados na análise
        if self.genetic_data:
            keys_per_sec = self.genetic_data.get('keys_per_second', 0)
            mutation_rate = self.genetic_data.get('mutation_rate', 0)
            
            if keys_per_sec < 1500:
                config["genetic_population"] = 1000  # Reduzir população
                config["max_workers"] = 6  # Mais workers
                
            if mutation_rate > 5.0:
                config["genetic_mutation_rate"] = 0.025  # Reduzir mutação
                
        # Salvar configuração
        with open('optimized_config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        print("✅ Configuração salva em 'optimized_config.json'")
        print()
        
    def display_summary_report(self):
        """Exibe relatório resumido completo"""
        print("\n" + "🎯 RELATÓRIO RESUMIDO DE ANÁLISE" + "\n")
        print("=" * 70)
        
        if self.progress_data:
            runtime = self.progress_data.get('runtime_hours', 0)
            keys_tested = self.progress_data.get('total_keys_tested', 0)
            candidates = len(self.progress_data.get('best_candidates', []))
            
            print(f"⏱️  Runtime total: {runtime:.2f} horas")
            print(f"🔑 Chaves testadas: {keys_tested:,}")
            print(f"🎯 Candidatos encontrados: {candidates}")
            
            if runtime > 0:
                rate = keys_tested / (runtime * 3600)
                print(f"⚡ Velocidade média: {rate:.1f} chaves/segundo")
        
        if self.genetic_data:
            print(f"🧬 Geração genética: {self.genetic_data.get('generation', 0)}")
            print(f"🏆 Melhor fitness: {self.genetic_data.get('best_fitness', 'N/A')}")
            
        print("\n✅ Análise completa! Verifique as sugestões acima.")
        print("🚀 Para aplicar otimizações, use 'optimized_config.json'")

def main():
    print("🔍 ULTRA SMART BITCOIN PUZZLE 71 - ANALISADOR DE RESULTADOS")
    print("=" * 70)
    
    analyzer = ResultAnalyzer()
    
    # Executar todas as análises
    analyzer.analyze_candidate_patterns()
    analyzer.analyze_genetic_performance()
    analyzer.generate_optimization_suggestions()
    analyzer.generate_optimization_script()
    analyzer.display_summary_report()

if __name__ == "__main__":
    main()
