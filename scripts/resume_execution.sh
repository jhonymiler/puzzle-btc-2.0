#!/usr/bin/env bash
# Script para resumir rapidamente uma execução a partir do último checkpoint

# Caminho base do projeto
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "🔄 Verificando checkpoints disponíveis..."

# Verifica se existem checkpoints
has_genetic_checkpoint=false
has_continuous_checkpoint=false
has_master_checkpoint=false

if [ -f "genetic_checkpoint.json" ]; then
    has_genetic_checkpoint=true
    genetic_timestamp=$(stat -c %y "genetic_checkpoint.json")
    echo "  ✅ Checkpoint genético encontrado (${genetic_timestamp})"
else
    echo "  ❌ Nenhum checkpoint genético encontrado"
fi

if [ -f "continuous_progress.json" ]; then
    has_continuous_checkpoint=true
    continuous_timestamp=$(stat -c %y "continuous_progress.json")
    echo "  ✅ Checkpoint de execução contínua encontrado (${continuous_timestamp})"
else
    echo "  ❌ Nenhum checkpoint de execução contínua encontrado"
fi

if [ -f "master_progress.json" ]; then
    has_master_checkpoint=true
    master_timestamp=$(stat -c %y "master_progress.json")
    echo "  ✅ Checkpoint do coordenador mestre encontrado (${master_timestamp})"
else
    echo "  ❌ Nenhum checkpoint do coordenador mestre encontrado"
fi

echo ""

# Verifica se pelo menos um checkpoint foi encontrado
if $has_genetic_checkpoint || $has_continuous_checkpoint || $has_master_checkpoint; then
    echo "💡 Checkpoints disponíveis! Escolha como retomar a execução:"
    echo "  1. Retomar com execução contínua (recomendado)"
    echo "  2. Retomar com coordenador mestre"
    echo "  3. Executar em modo teste (30 minutos)"
    echo "  4. Cancelar"
    
    read -p "Escolha uma opção (1-4): " option
    
    case $option in
        1)
            echo "🚀 Retomando com execução contínua..."
            python3 main.py --resume
            ;;
        2)
            echo "🚀 Retomando com coordenador mestre..."
            python3 main.py --master
            ;;
        3)
            echo "🚀 Executando em modo teste por 30 minutos..."
            cd "$PROJECT_DIR"
            python3 src/continuous_runner.py <<< "4
0.5"  # Responde às perguntas do script para executar por 30 minutos (0.5h)
            ;;
        4)
            echo "❌ Operação cancelada."
            ;;
        *)
            echo "❌ Opção inválida. Execute o script novamente."
            ;;
    esac
else
    echo "❌ Nenhum checkpoint encontrado para retomar execução."
    echo "💡 Execute primeiro o script principal com uma das opções de execução."
    echo "    python3 main.py --continuous"
fi
