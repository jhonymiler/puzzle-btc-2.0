"""
Sistema de Salvamento de Chaves Privadas Bitcoin
Garante que todas as chaves encontradas sejam preservadas com segurança
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

class KeySaver:
    """Sistema robusto para salvar chaves privadas encontradas"""
    
    def __init__(self, base_dir: str = "/home/jhony/Projetos/GeneticBitcoinSolver"):
        self.base_dir = base_dir
        self.keys_dir = os.path.join(base_dir, "found_keys")
        self.json_file = os.path.join(self.keys_dir, "discovered_keys.json")
        self.backup_file = os.path.join(self.keys_dir, "keys_backup.txt")
        self.setup_directories()
        
    def setup_directories(self):
        """Cria diretórios necessários se não existirem"""
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Inicializa arquivo JSON se não existir
        if not os.path.exists(self.json_file):
            with open(self.json_file, 'w') as f:
                json.dump({"discovered_keys": [], "metadata": {"created": datetime.now().isoformat()}}, f, indent=2)
    
    def save_found_key(self, 
                      puzzle_number: int,
                      private_key: str,
                      address: str,
                      solver_name: str,
                      additional_info: Optional[Dict] = None) -> bool:
        """
        Salva uma chave privada encontrada com todos os detalhes
        
        Args:
            puzzle_number: Número do puzzle Bitcoin
            private_key: Chave privada em formato hexadecimal
            address: Endereço Bitcoin correspondente
            solver_name: Nome do algoritmo que encontrou a chave
            additional_info: Informações adicionais opcionais
            
        Returns:
            bool: True se salvou com sucesso, False caso contrário
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Dados da chave encontrada
            key_data = {
                "puzzle": puzzle_number,
                "private_key": private_key,
                "address": address,
                "solver": solver_name,
                "timestamp": timestamp,
                "hash": hashlib.sha256(f"{puzzle_number}{private_key}{address}".encode()).hexdigest()[:16]
            }
            
            if additional_info:
                key_data["additional_info"] = additional_info
            
            # Salvar no arquivo JSON
            self._save_to_json(key_data)
            
            # Salvar no backup TXT
            self._save_to_backup(key_data)
            
            # Criar arquivo individual para esta chave
            self._save_individual_file(key_data)
            
            print(f"🔑 CHAVE SALVA COM SUCESSO!")
            print(f"   Puzzle: {puzzle_number}")
            print(f"   Chave: {private_key}")
            print(f"   Endereço: {address}")
            print(f"   Algoritmo: {solver_name}")
            print(f"   Arquivo: {self.json_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERRO ao salvar chave: {e}")
            return False
    
    def _save_to_json(self, key_data: Dict[str, Any]):
        """Salva no arquivo JSON principal"""
        try:
            # Carregar dados existentes
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            # Verificar se já existe esta chave
            existing_keys = [k["private_key"] for k in data["discovered_keys"]]
            if key_data["private_key"] not in existing_keys:
                data["discovered_keys"].append(key_data)
                data["metadata"]["last_updated"] = datetime.now().isoformat()
                data["metadata"]["total_keys"] = len(data["discovered_keys"])
                
                # Salvar com backup do arquivo anterior
                backup_json = self.json_file + ".backup"
                if os.path.exists(self.json_file):
                    import shutil
                    shutil.copy2(self.json_file, backup_json)
                
                with open(self.json_file, 'w') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    
        except Exception as e:
            print(f"❌ Erro ao salvar no JSON: {e}")
    
    def _save_to_backup(self, key_data: Dict[str, Any]):
        """Salva no arquivo de backup TXT"""
        try:
            line = f"{key_data['timestamp']} | Puzzle {key_data['puzzle']} | {key_data['private_key']} | {key_data['address']} | {key_data['solver']}\n"
            
            with open(self.backup_file, 'a') as f:
                f.write(line)
                
        except Exception as e:
            print(f"❌ Erro ao salvar backup: {e}")
    
    def _save_individual_file(self, key_data: Dict[str, Any]):
        """Cria arquivo individual para cada chave encontrada"""
        try:
            filename = f"puzzle_{key_data['puzzle']}_{key_data['hash']}.json"
            filepath = os.path.join(self.keys_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(key_data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Erro ao criar arquivo individual: {e}")
    
    def list_found_keys(self) -> list:
        """Lista todas as chaves encontradas"""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            return data["discovered_keys"]
        except:
            return []
    
    def get_key_for_puzzle(self, puzzle_number: int) -> Optional[Dict]:
        """Busca chave específica para um puzzle"""
        keys = self.list_found_keys()
        for key in keys:
            if key["puzzle"] == puzzle_number:
                return key
        return None
    
    def export_summary(self) -> str:
        """Gera relatório resumido das chaves encontradas"""
        keys = self.list_found_keys()
        if not keys:
            return "Nenhuma chave encontrada ainda."
        
        summary = f"📊 RESUMO DAS CHAVES ENCONTRADAS ({len(keys)} total):\n\n"
        
        for key in keys:
            summary += f"🔑 Puzzle {key['puzzle']}:\n"
            summary += f"   Chave: {key['private_key']}\n"
            summary += f"   Endereço: {key['address']}\n"
            summary += f"   Algoritmo: {key['solver']}\n"
            summary += f"   Data: {key['timestamp']}\n\n"
        
        return summary

# Instância global para uso fácil
key_saver = KeySaver()

def save_discovered_key(puzzle_number: int, private_key: str, address: str, solver_name: str, **kwargs) -> bool:
    """Função helper para salvar chaves facilmente"""
    return key_saver.save_found_key(puzzle_number, private_key, address, solver_name, kwargs)

if __name__ == "__main__":
    # Teste do sistema
    saver = KeySaver()
    
    # Teste com uma chave fictícia
    success = saver.save_found_key(
        puzzle_number=32,
        private_key="0000000000000000000000000000000000000000000000000000000000000001",
        address="1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm",
        solver_name="TestSolver",
        additional_info={"test": True}
    )
    
    print(f"Teste de salvamento: {'✅ Sucesso' if success else '❌ Falhou'}")
    print("\n" + saver.export_summary())
