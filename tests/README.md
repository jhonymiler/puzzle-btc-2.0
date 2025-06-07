# Testes do GeneticBitcoinSolver

Este diretório contém testes abrangentes para validar a funcionalidade do solucionador genético do Bitcoin Puzzle.

## Lista de Testes

1. **test_gpu_kernels.py** - Testa operações básicas dos kernels GPU
2. **test_gpu_integration.py** - Testa integração entre kernels GPU e o solucionador genético
3. **test_checkpoint.py** - Testa funcionalidade de salvar e carregar checkpoints
4. **run_all_tests.sh** - Script para executar todos os testes de uma vez

## Executando os Testes

### Teste Individual

Para executar um teste específico:

```bash
python tests/test_gpu_kernels.py
```

### Todos os Testes

Para executar todos os testes de uma vez:

```bash
./tests/run_all_tests.sh
```

## Checagem de GPU

Para testar se a aceleração GPU está funcionando corretamente:

```bash
./scripts/test_gpu_acceleration.sh
```

## Resultados dos Testes

Os resultados dos testes serão exibidos no console. Se todos os testes passarem, você verá mensagens de sucesso. Caso contrário, mensagens de erro específicas serão exibidas.

## Notas sobre Desempenho

- O desempenho da aceleração GPU depende do hardware disponível
- Para GPUs NVIDIA, é esperado um ganho de desempenho significativo
- Para GPUs AMD, o suporte é experimental
- Para Apple Silicon (MPS), o suporte também é experimental
