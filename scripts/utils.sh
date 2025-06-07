#!/bin/bash
# 🛠️ UTILITÁRIOS DO BITCOIN PUZZLE 71 SOLVER
# ==========================================
# Script com funções úteis para gerenciar o projeto

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

show_help() {
    echo "🛠️  UTILITÁRIOS DO BITCOIN PUZZLE 71 SOLVER"
    echo "==========================================="
    echo ""
    echo "Uso: ./scripts/utils.sh [comando]"
    echo ""
    echo "Comandos disponíveis:"
    echo "  status      - Mostra status do projeto"
    echo "  clean       - Executa limpeza automática"
    echo "  install     - Instala dependências"
    echo "  test        - Executa testes"
    echo "  run         - Executa o solver principal"
    echo "  monitor     - Mostra monitor de execução"
    echo "  resume      - Resume da última execução"
    echo "  env         - Mostra informações do ambiente"
    echo "  help        - Mostra esta ajuda"
    echo ""
}

show_status() {
    echo "📊 STATUS DO PROJETO"
    echo "==================="
    echo ""
    
    echo "📁 Tamanho do projeto:"
    du -sh . | cut -f1
    echo ""
    
    echo "🔍 Arquivos principais:"
    ls -la *.py *.md 2>/dev/null || echo "Nenhum arquivo Python/Markdown na raiz"
    echo ""
    
    echo "📂 Estrutura de diretórios:"
    find . -type d -name ".*" -prune -o -type d -print | head -10
    echo ""
    
    echo "💾 Checkpoints existentes:"
    ls -la *progress*.json *checkpoint*.json 2>/dev/null || echo "Nenhum checkpoint encontrado"
    echo ""
    
    echo "🔑 Chaves encontradas:"
    if [ -f "found_keys/discovered_keys.json" ]; then
        echo "Arquivo de chaves existe"
        cat found_keys/discovered_keys.json 2>/dev/null || echo "Erro ao ler arquivo"
    else
        echo "Nenhuma chave encontrada ainda"
    fi
}

case "$1" in
    "status")
        show_status
        ;;
    "clean")
        ./scripts/cleanup.sh
        ;;
    "install")
        python3 install_dependencies.py
        ;;
    "test")
        python3 main.py --test
        ;;
    "run")
        python3 main.py --master
        ;;
    "monitor")
        python3 main.py --monitor
        ;;
    "resume")
        python3 main.py --resume
        ;;
    "env")
        python3 main.py --environment
        ;;
    "help"|"")
        show_help
        ;;
    *)
        echo "❌ Comando inválido: $1"
        echo "Use 'help' para ver comandos disponíveis"
        ;;
esac
