#!/usr/bin/env bash
# Script para testar aceleração GPU e kernels
# Este script executa os testes de kernels GPU para validar a implementação

# Caminho base do projeto
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "🧪 TESTANDO ACELERAÇÃO GPU E KERNELS"
echo "===================================="
echo "Executando testes em $(date)"
echo ""

# Ativa o ambiente virtual se existir
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "🔧 Ambiente virtual ativado"
else
    echo "⚠️ Ambiente virtual não encontrado. Continuando sem ativação."
fi

# Verifica dependências
echo "🔍 Verificando dependências necessárias..."
python -c "
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print('✅ PyTorch disponível: {}'.format(torch.__version__))
    print('   - CUDA suportado: {}'.format('Sim' if cuda_available else 'Não'))
    if cuda_available:
        print('   - Dispositivo: {}'.format(torch.cuda.get_device_name(0)))
except ImportError:
    print('❌ PyTorch não instalado')

try:
    import ecdsa
    print('✅ ECDSA disponível: {}'.format(ecdsa.__version__))
except ImportError:
    print('❌ ECDSA não instalado')

try:
    import numpy
    print('✅ NumPy disponível: {}'.format(numpy.__version__))
except ImportError:
    print('❌ NumPy não instalado')
"

echo ""
echo "🧪 Executando testes de kernels GPU..."
echo ""

# Executa o script de teste
python "$PROJECT_DIR/tests/test_gpu_kernels.py"

# Verifica o resultado
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✅ Os testes foram executados com sucesso!"
else
    echo ""
    echo "❌ Os testes falharam com código de erro $TEST_RESULT"
fi

echo ""
echo "💡 Dicas:"
echo "  - Se os testes falharam, verifique se todas as dependências estão instaladas"
echo "  - Instale torch com suporte a GPU: pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116"
echo "  - Para GPUs AMD: pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2"
echo ""
echo "Script concluído em $(date)"
