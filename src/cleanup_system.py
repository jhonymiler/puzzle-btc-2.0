#!/usr/bin/env python3
"""
🧹 SCRIPT DE LIMPEZA DO SISTEMA
===============================

Remove arquivos de teste, temporários e desnecessários do sistema
Bitcoin Puzzle 71 Solver, mantendo apenas os arquivos essenciais.
"""

import os
import json
import shutil
from pathlib import Path

def cleanup_test_files():
    """Remove arquivos de teste específicos"""
    print("🧹 Removendo arquivos de teste...")
    
    test_files = [
        'test_key_saver.py',
        'advanced_validation_test.py',
        'honest_validation_test.py', 
        'robust_validation_test.py',
        'validation_test.py'
    ]
    
    removed = 0
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  ✅ Removido: {file}")
                removed += 1
            except Exception as e:
                print(f"  ❌ Erro ao remover {file}: {e}")
    
    print(f"📊 {removed} arquivo(s) de teste removido(s)")
    return removed

def cleanup_test_keys():
    """Remove chaves de teste do sistema de descobertas"""
    print("\n🔑 Limpando chaves de teste...")
    
    found_keys_dir = "found_keys"
    if not os.path.exists(found_keys_dir):
        print("  ℹ️  Diretório found_keys não existe")
        return 0
    
    # Backup do arquivo principal
    main_file = os.path.join(found_keys_dir, "discovered_keys.json")
    backup_file = os.path.join(found_keys_dir, "discovered_keys.json.backup")
    
    cleaned_keys = 0
    
    if os.path.exists(main_file):
        try:
            with open(main_file, 'r') as f:
                data = json.load(f)
            
            original_count = len(data.get('discovered_keys', []))
            
            # Remove chaves marcadas como teste
            production_keys = []
            for key in data.get('discovered_keys', []):
                additional_info = key.get('additional_info', {})
                solver = key.get('solver', '')
                
                # Verifica se é chave de teste
                is_test = (
                    additional_info.get('test', False) or
                    'TESTE' in solver.upper() or
                    'TEST' in solver.upper()
                )
                
                if not is_test:
                    production_keys.append(key)
                else:
                    cleaned_keys += 1
            
            # Atualiza os dados
            data['discovered_keys'] = production_keys
            data['metadata']['total_keys'] = len(production_keys)
            data['metadata']['last_updated'] = data['metadata'].get('last_updated', '')
            
            # Salva arquivo limpo
            with open(main_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ {cleaned_keys} chave(s) de teste removida(s)")
            print(f"  📊 {len(production_keys)} chave(s) de produção mantida(s)")
            
        except Exception as e:
            print(f"  ❌ Erro ao limpar arquivo principal: {e}")
    
    # Remove arquivo de backup se existir
    if os.path.exists(backup_file):
        try:
            os.remove(backup_file)
            print(f"  ✅ Backup removido: {backup_file}")
        except Exception as e:
            print(f"  ❌ Erro ao remover backup: {e}")
    
    return cleaned_keys

def cleanup_individual_key_files():
    """Remove arquivos individuais de chaves de teste"""
    print("\n📄 Limpando arquivos individuais de chaves...")
    
    found_keys_dir = "found_keys"
    if not os.path.exists(found_keys_dir):
        return 0
    
    removed = 0
    
    # Lista arquivos JSON individuais de puzzle
    for file in os.listdir(found_keys_dir):
        if file.startswith('puzzle_71_') and file.endswith('.json'):
            file_path = os.path.join(found_keys_dir, file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Verifica se é arquivo de teste
                additional_info = data.get('additional_info', {})
                solver = data.get('solver', '')
                
                is_test = (
                    additional_info.get('test', False) or
                    'TESTE' in solver.upper() or
                    'TEST' in solver.upper()
                )
                
                if is_test:
                    os.remove(file_path)
                    print(f"  ✅ Removido: {file}")
                    removed += 1
                else:
                    print(f"  📌 Mantido: {file} (produção)")
                    
            except Exception as e:
                print(f"  ❌ Erro ao processar {file}: {e}")
    
    print(f"📊 {removed} arquivo(s) individual(is) removido(s)")
    return removed

def cleanup_temp_files():
    """Remove arquivos temporários"""
    print("\n🗂️  Removendo arquivos temporários...")
    
    temp_patterns = [
        '*.tmp',
        '*.temp',
        'forensic_candidates.json',
        'genetic_checkpoint.json',
        'keys_backup.txt'
    ]
    
    removed = 0
    
    for pattern in temp_patterns:
        if '*' in pattern:
            # Padrão com wildcard - precisa ser implementado manualmente
            continue
        else:
            # Arquivo específico
            if os.path.exists(pattern):
                try:
                    os.remove(pattern)
                    print(f"  ✅ Removido: {pattern}")
                    removed += 1
                except Exception as e:
                    print(f"  ❌ Erro ao remover {pattern}: {e}")
    
    # Remove arquivos específicos do found_keys
    found_keys_dir = "found_keys"
    if os.path.exists(found_keys_dir):
        temp_files = ['keys_backup.txt']
        
        for temp_file in temp_files:
            temp_path = os.path.join(found_keys_dir, temp_file)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"  ✅ Removido: {temp_path}")
                    removed += 1
                except Exception as e:
                    print(f"  ❌ Erro ao remover {temp_path}: {e}")
    
    print(f"📊 {removed} arquivo(s) temporário(s) removido(s)")
    return removed

def cleanup_pycache():
    """Remove cache do Python"""
    print("\n🐍 Limpando cache do Python...")
    
    removed = 0
    
    if os.path.exists('__pycache__'):
        try:
            shutil.rmtree('__pycache__')
            print("  ✅ Removido: __pycache__/")
            removed += 1
        except Exception as e:
            print(f"  ❌ Erro ao remover __pycache__: {e}")
    
    # Remove arquivos .pyc individuais
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                try:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"  ✅ Removido: {file_path}")
                    removed += 1
                except Exception as e:
                    print(f"  ❌ Erro ao remover {file}: {e}")
    
    print(f"📊 {removed} arquivo(s) de cache removido(s)")
    return removed

def show_remaining_structure():
    """Mostra a estrutura final do projeto"""
    print("\n📁 ESTRUTURA FINAL DO PROJETO:")
    print("=" * 50)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        items = sorted(os.listdir(directory))
        dirs = [item for item in items if os.path.isdir(os.path.join(directory, item))]
        files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
        
        # Primeiro as pastas
        for i, dir_name in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            print(f"{prefix}{'└── ' if is_last_dir else '├── '}{dir_name}/")
            
            new_prefix = prefix + ("    " if is_last_dir else "│   ")
            try:
                print_tree(os.path.join(directory, dir_name), new_prefix, max_depth, current_depth + 1)
            except PermissionError:
                pass
        
        # Depois os arquivos
        for i, file_name in enumerate(files):
            is_last = i == len(files) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{file_name}")
    
    try:
        print_tree(".")
    except Exception as e:
        print(f"Erro ao mostrar estrutura: {e}")

def main():
    """Executa limpeza completa do sistema"""
    print("🚀 INICIANDO LIMPEZA COMPLETA DO SISTEMA")
    print("=" * 60)
    print("⚠️  Esta operação removerá arquivos de teste e temporários")
    print("💾 Arquivos de produção serão preservados")
    print()
    
    total_removed = 0
    
    # Executa todas as limpezas
    cleanups = [
        cleanup_test_files,
        cleanup_test_keys,
        cleanup_individual_key_files,
        cleanup_temp_files,
        cleanup_pycache
    ]
    
    for cleanup_func in cleanups:
        try:
            removed = cleanup_func()
            total_removed += removed
        except Exception as e:
            print(f"❌ Erro durante limpeza: {e}")
    
    print(f"\n🎉 LIMPEZA CONCLUÍDA!")
    print("=" * 30)
    print(f"📊 Total de arquivos removidos: {total_removed}")
    print("✅ Sistema limpo e organizado")
    
    # Mostra estrutura final
    show_remaining_structure()

if __name__ == "__main__":
    main()
