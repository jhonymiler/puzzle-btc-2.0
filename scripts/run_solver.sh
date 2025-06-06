#!/bin/bash
# 🚀 SCRIPT DE EXECUÇÃO AUTOMÁTICA - BITCOIN PUZZLE 71 SOLVER

echo "🚀 BITCOIN PUZZLE 71 SOLVER - EXECUÇÃO AUTOMÁTICA"
echo "================================================="

# Verificar se está no diretório correto
if [ ! -f "src/continuous_runner.py" ]; then
    echo "❌ Erro: Execute este script no diretório raiz do projeto"
    exit 1
fi

# Menu de opções
echo ""
echo "OPÇÕES DE EXECUÇÃO:"
echo "1. 🕐 Teste rápido (5 minutos)"
echo "2. 🕰️ Execução média (1 hora)"
echo "3. 🌙 Execução longa (6 horas)"
echo "4. 🔄 Execução contínua"
echo ""

read -p "Escolha uma opção (1-4): " choice

case $choice in
    1)
        echo "🕐 Iniciando teste rápido de 5 minutos..."
        echo "1" | python3 src/continuous_runner.py
        ;;
    2)
        echo "🕰️ Iniciando execução de 1 hora..."
        echo "1" | python3 src/continuous_runner.py
        ;;
    3)
        echo "🌙 Iniciando execução de 6 horas..."
        echo "2" | python3 src/continuous_runner.py
        ;;
    4)
        echo "🔄 Iniciando execução contínua (até encontrar)..."
        echo "5" | python3 src/continuous_runner.py
        ;;
    *)
        echo "❌ Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "✅ Execução concluída!"
echo "📄 Verifique os arquivos de log e relatórios gerados"
