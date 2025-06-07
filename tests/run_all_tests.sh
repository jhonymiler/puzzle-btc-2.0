#!/usr/bin/env bash
# Script para executar testes end-to-end do sistema de Bitcoin Puzzle Solver
# Este script executa uma série de testes para validar todas as implementações

# Caminho base do projeto
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "🚀 TESTE END-TO-END DO BITCOIN PUZZLE SOLVER"
echo "==========================================="
echo "Executando testes em $(date)"
echo ""

# Ativa o ambiente virtual se existir
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "🔧 Ambiente virtual ativado"
else
    echo "⚠️ Ambiente virtual não encontrado. Continuando sem ativação."
fi

# Função para executar um teste e verificar resultado
run_test() {
    test_name="$1"
    test_cmd="$2"
    
    echo ""
    echo "🧪 EXECUTANDO TESTE: $test_name"
    echo "--------------------------------------"
    
    eval "$test_cmd"
    result=$?
    
    if [ $result -eq 0 ]; then
        echo "✅ Teste '$test_name' concluído com sucesso!"
    else
        echo "❌ Teste '$test_name' falhou com código de erro $result"
        failed_tests+=("$test_name")
    fi
    
    return $result
}

# Armazena os testes que falharam
failed_tests=()

# 1. Teste do detector de ambiente
run_test "Detector de Ambiente" "python -c \"
from src.environment_detector import get_environment_detector
env = get_environment_detector()
print('Ambiente detectado:', env.environment)
print('GPU disponível:', env.cuda_available or env.rocm_available or env.mps_available)
\""

# 2. Teste dos kernels GPU
run_test "Kernels GPU" "python tests/test_gpu_kernels.py"

# 3. Teste de integração GPU
run_test "Integração GPU" "python tests/test_gpu_integration.py --population 50 --generations 1"

# 4. Teste de checkpoint
run_test "Checkpoint" "python tests/test_checkpoint.py"

# 5. Teste resumo de execução (simulação rápida)
mkdir -p test_output
run_test "Resumo de Execução" "python -c \"
import sys
sys.path.append('$PROJECT_DIR')
from src.genetic_bitcoin_solver import GeneticBitcoinSolver
# Primeiro, criar um checkpoint
solver1 = GeneticBitcoinSolver(population_size=50)
population = solver1.initialize_population()
solver1.generation = 1
solver1.save_checkpoint(population)
print('Checkpoint inicial criado')
# Agora, carregar esse checkpoint em uma nova instância
solver2 = GeneticBitcoinSolver(population_size=50)
loaded_gen, loaded_pop = solver2.load_checkpoint()
print(f'Checkpoint carregado: geração {loaded_gen}, população {len(loaded_pop)}')
# Validar que funcionou
assert loaded_gen == 1, 'A geração carregada não corresponde à salva'
assert len(loaded_pop) > 0, 'Nenhuma população foi carregada'
print('Validação de checkpoint concluída com sucesso!')
\""

# Exibe resumo dos testes
echo ""
echo "📊 RESUMO DOS TESTES"
echo "====================================="

if [ ${#failed_tests[@]} -eq 0 ]; then
    echo "✅ TODOS OS TESTES FORAM BEM-SUCEDIDOS!"
else
    echo "⚠️ ${#failed_tests[@]} TESTES FALHARAM:"
    for test in "${failed_tests[@]}"; do
        echo "  - $test"
    done
fi

echo ""
echo "Testes concluídos em $(date)"
echo "====================================="

# Se algum teste falhou, retorna código de erro
if [ ${#failed_tests[@]} -gt 0 ]; then
    exit 1
fi
exit 0
