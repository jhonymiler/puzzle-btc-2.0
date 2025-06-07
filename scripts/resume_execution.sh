#!/usr/bin/env bash
# Script para resumir rapidamente uma execução a partir do último checkpoint
# Versão 2.0 - Com detecção de ambiente e aceleração

# Caminho base do projeto
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "🔧 Verificando ambiente de execução..."

# Verifica disponibilidade de GPU
has_nvidia_gpu=false
has_amd_gpu=false
has_apple_silicon=false

# Verifica NVIDIA
if command -v nvidia-smi &> /dev/null; then
    has_nvidia_gpu=true
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
    echo "  ✅ GPU NVIDIA detectada: ${gpu_info}"
# Verifica AMD
elif command -v rocminfo &> /dev/null; then
    has_amd_gpu=true
    echo "  ✅ GPU AMD detectada"
# Verifica Apple Silicon
elif [ "$(uname)" == "Darwin" ] && [[ "$(uname -m)" == "arm64" ]]; then
    has_apple_silicon=true
    echo "  ✅ Apple Silicon detectado"
else
    echo "  ⚠️ Nenhuma GPU detectada, usando apenas CPU"
fi

# Verifica RAM disponível
total_ram=$(free -g | awk '/^Mem:/{print $2}')
echo "  📊 RAM disponível: ${total_ram}GB"

# Verifica núcleos de CPU
cpu_cores=$(nproc)
echo "  💻 Núcleos de CPU: ${cpu_cores}"

echo ""
echo "🔄 Verificando checkpoints disponíveis..."

# Verifica se existem checkpoints
has_genetic_checkpoint=false
has_continuous_checkpoint=false
has_master_checkpoint=false

# Encontra o checkpoint genético mais recente
latest_genetic_checkpoint="genetic_checkpoint.json"
if [ -f "$latest_genetic_checkpoint" ]; then
    has_genetic_checkpoint=true
    genetic_timestamp=$(stat -c %y "$latest_genetic_checkpoint")
    
    # Verifica se há backups mais recentes
    backup_checkpoints=$(ls -t checkpoint_gen*.json 2>/dev/null | head -n 1)
    if [ ! -z "$backup_checkpoints" ]; then
        backup_timestamp=$(stat -c %y "$backup_checkpoints")
        echo "  ✅ Checkpoint genético encontrado (${genetic_timestamp})"
        echo "  💾 Backup de checkpoint encontrado: ${backup_checkpoints} (${backup_timestamp})"
    else
        echo "  ✅ Checkpoint genético encontrado (${genetic_timestamp})"
    fi
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
    echo "  1. Retomar com execução otimizada para GPU (recomendado se disponível)"
    echo "  2. Retomar com execução contínua padrão"
    echo "  3. Retomar com coordenador mestre"
    echo "  4. Executar em modo teste (30 minutos)"
    echo "  5. Verificar integridade do checkpoint"
    echo "  6. Cancelar"
    
    read -p "Escolha uma opção (1-6): " option
    
    case $option in
        1)
            if $has_nvidia_gpu || $has_amd_gpu || $has_apple_silicon; then
                echo "🚀 Retomando com aceleração GPU otimizada..."
                # Verifica ou cria um ambiente virtual Python se não existir
                if [ ! -d "venv" ]; then
                    echo "🔧 Configurando ambiente Python para GPU..."
                    python3 -m venv venv
                    source venv/bin/activate
                    
                    # Instala dependências específicas para GPU
                    if $has_nvidia_gpu; then
                        echo "🔧 Instalando pacotes para NVIDIA GPU..."
                        pip install -r requirements.txt
                        pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
                    elif $has_amd_gpu; then
                        echo "🔧 Instalando pacotes para AMD GPU..."
                        pip install -r requirements.txt
                        pip install torch==2.0.0 -f https://download.pytorch.org/whl/rocm5.2/torch_stable.html
                    elif $has_apple_silicon; then
                        echo "🔧 Instalando pacotes para Apple Silicon..."
                        pip install -r requirements.txt
                        pip install torch
                    fi
                else
                    source venv/bin/activate
                fi
                
                # Executa com otimização para GPU
                python3 src/genetic_bitcoin_solver.py
            else
                echo "⚠️ Nenhuma GPU detectada, usando modo padrão..."
                python3 main.py --resume
            fi
            ;;
        2)
            echo "🚀 Retomando com execução contínua..."
            python3 main.py --resume
            ;;
        3)
            echo "🚀 Retomando com coordenador mestre..."
            python3 main.py --master
            ;;
        4)
            echo "🚀 Executando em modo teste por 30 minutos..."
            cd "$PROJECT_DIR"
            python3 src/continuous_runner.py <<< "4
0.5"  # Responde às perguntas do script para executar por 30 minutos (0.5h)
            ;;
        5)
            echo "🔍 Verificando integridade do checkpoint..."
            python3 -c "
import json
try:
    with open('genetic_checkpoint.json') as f:
        data = json.load(f)
    print('✅ Checkpoint válido com {} indivíduos em população'.format(len(data.get('population', []))))
    print('📊 Melhor fitness: {:.2f}'.format(data.get('best_fitness', 0)))
    print('🧬 Geração: {}'.format(data.get('generation', 0)))
except Exception as e:
    print('❌ Erro ao validar checkpoint: {}'.format(e))
"
            ;;
        6)
            echo "❌ Operação cancelada."
            ;;
        *)
            echo "❌ Opção inválida. Execute o script novamente."
            ;;
    esac
else
    echo "❌ Nenhum checkpoint encontrado para retomar execução."
    echo "💡 Execute primeiro o algoritmo genético para criar um checkpoint:"
    echo ""
    echo "    python3 src/genetic_bitcoin_solver.py      # Execução direta otimizada"
    echo "    python3 main.py --continuous              # Execução contínua com retomada"
    
    if $has_nvidia_gpu || $has_amd_gpu || $has_apple_silicon; then
        echo ""
        echo "💡 Detecção de GPU: Você pode executar uma versão otimizada para GPU:"
        echo ""
        echo "    1. Executar solver com otimização GPU"
        echo "    2. Cancelar"
        
        read -p "Escolha uma opção (1-2): " gpu_option
        
        if [ "$gpu_option" == "1" ]; then
            echo "🚀 Iniciando execução otimizada para GPU..."
            python3 src/genetic_bitcoin_solver.py
        fi
    fi
fi
