# GPT-2 do zero em português: vanilla, Mixture of Experts e RoPE

Trabalho final da disciplina de Large Language Models da UFCG, trilha de
pós-graduação, individual. Um GPT-2 é treinado do zero no corpus de introduções
de trabalhos acadêmicos da UFCG, e quatro treinos são comparados sob o mesmo
orçamento de tokens: o vanilla, duas variantes de Mixture of Experts e uma
variante com RoPE.

Entrega: 14 de dezembro de 2026, 23:59.

## Estado

| Fase | Situação |
|---|---|
| 0. Fundação | concluída |
| 1. Dados | concluída, exceto a comparação com o `tiktoken` |
| 2. Modelo | concluída |
| 3. Treino | concluída; as quatro otimizações medidas na T4 |
| 4. Benchmark | concluída |
| 5. E1 vanilla | concluída |
| 6. MoE | duas variantes treinadas e analisadas |
| 7. RoPE | concluída, com a extrapolação de contexto |
| 8. Avaliação | os quatro BPB medidos e o roteamento analisado |
| 9. Artigo e oral | pendente |

O acompanhamento detalhado, tarefa por tarefa, está em [TASKS.md](TASKS.md). O
desenho experimental e os tempos estão em [PLANO.md](PLANO.md).

## Experimentos

Todos usam o tokenizador de referência de 16k, contexto de 256 tokens,
27,4M tokens de treino e semente 123. Os quatro treinos rodaram em 20/11/2026
na T4 do Colab, 837 passos cada, em 5 a 7 minutos.

| ID | Modelo | Parâmetros totais / ativos | Loss val | BPB val | Log |
|---|---|---|---|---|---|
| E1 | vanilla otimizado | 11,41M / 11,41M | 4,9350 | 1,4231 | `logs/e1_vanilla.csv` |
| E2 | MoE, 8 especialistas, top-2, h=512 | 17,73M / 11,42M | 4,8871 | 1,4093 | `logs/e2_moe_h512.csv` |
| E3 | MoE, 8 especialistas, top-2, h=128 | 11,43M / 9,85M | 4,9173 | 1,4180 | `logs/e3_moe_h128.csv` |
| E4 | RoPE | 11,35M / 11,35M | 4,6892 | 1,3522 | `logs/e4_rope.csv` |

A perda e o BPB são da partição de validação inteira, 1.575.936 tokens e
7.884.234 bytes, medidos por `src/evaluate.py` a partir do checkpoint final. Os
JSONs de cada avaliação estão em `logs/`. O RoPE reduz o BPB em 5,0% em relação
ao vanilla; as duas variantes de MoE ficam 1,0% e 0,4% abaixo, com menos
parâmetros ativos na segunda do que o vanilla tem.

A coluna `loss_val` do CSV de treino lê cerca de 0,008 acima desta tabela. Os
CSVs foram gerados antes da correção do peso do último lote parcial na função
de validação; a tabela e os JSONs vêm de `evaluate.py`, que sempre ponderou por
token. Um novo treino com o código atual reproduz os valores da tabela.

## Extrapolação de contexto

O RoPE avalia além dos 256 tokens de treino sem ajuste fino, e o vanilla não
avalia. O mecanismo é o que muda: o embedding posicional aprendido tem uma
linha por posição vista no treino, e não existe linha 512; o RoPE gira Q e K
com um ângulo calculado a partir do índice da posição, que continua definido.

A perda do RoPE sobe pouco com a janela maior: 4,6883 em 256, 4,7053 em 512 e
4,7591 em 1024, ou 0,4% e 1,5% acima. O modelo não ganha com mais contexto,
porque nunca foi treinado nele, e é isso que a medição separa: o mecanismo
aceita a janela, o modelo não a aproveita.

| Contexto | E1 vanilla | E4 RoPE |
|---|---|---|
| 128 | 1,4248 | 1,3560 |
| 256 | 1,4226 | 1,3517 |
| 512 | não suportado | 1,3566 |
| 1024 | não suportado | 1,3721 |

As janelas têm o tamanho do contexto e não se sobrepõem, então cada linha
avalia 1.572.864 tokens e 7.870.583 bytes. É um pouco menos que os 1.575.936
tokens da avaliação padrão, porque as pontas que não fecham uma janela inteira
ficam de fora; por isso o BPB de 256 do RoPE aqui é 1,3517 e não 1,3522.


## Análise do roteamento

`python -m src.figuras_moe` lê `logs/e2_moe_h512_roteamento.json` e
`logs/e3_moe_h128_roteamento.json` e escreve três figuras em `figuras/`: a
fatia do tráfego top-1 por especialista e bloco, com escala divergente
centrada em 12,5%; a entropia normalizada por bloco; e a repartição dos dez
tokens mais frequentes no bloco 0. A tabela resumo sai em
`tabelas/roteamento.tex`.

Nenhuma das duas variantes colapsa. A fatia fica entre 0,092 e 0,159, contra
0,125 do valor justo, e a entropia entre 0,69 e 0,81. A especialização por
token é forte mesmo assim: no E2, 60% dos despachos de ` de` vão para o
especialista 4 do bloco 0, e 33% dos da vírgula vão para o especialista 0.

## Estrutura

```
src/          modelo, dados, treino, avaliação
configs/      um YAML por experimento
tests/        testes de shapes, máscara causal, parâmetros e retomada
logs/         CSV de cada treino, versionados
figuras/      figuras do MoE, geradas por src/figuras_moe.py
tabelas/      tabela LaTeX do roteamento, gerada pelo mesmo script
notebooks/    material do professor e o notebook de treino no Colab
docs/         enunciado, README do corpus e notas
```

## Ambiente

Verificado em 20/11/2026: Colab gratuito, Tesla T4 com 16 GB, Python 3.13.16,
PyTorch 2.14.1 com CUDA 13.0. A máquina local não tem GPU e serve para
pré-tokenização e testes de CPU.

O corpus não é versionado. Ele está no Google Drive do projeto e no link
compartilhado pelo professor, descrito em [docs/corpus-readme.md](docs/corpus-readme.md).

## Como reproduzir

Verificado localmente em 20/11/2026, em CPU, com o corpus e o tokenizador em
`WORK_DIR/corpus/`. O pré-processamento é feito uma vez e reutilizado; sem ele,
todo comando falha com uma mensagem de caminho ausente.

```bash
# 1. Pré-tokenização e conferência contra o .parquet (uma vez, ~2 min)
python -m src.data --work-dir . --verificar-parquet

# 2. Testes
pytest

# 3. Fumaça de CPU, 30 passos, para checar o pipeline
python -m src.train --config configs/smoke_cpu.yaml

# 4. Benchmark de custo das otimizações (na GPU)
python -m src.benchmark --config configs/benchmark.yaml

# 5. Treino completo de um experimento
python -m src.run_session --config configs/e1_vanilla.yaml --max-hours 3

# 6. Avaliação de um checkpoint
python -m src.evaluate --checkpoint checkpoints/e1_vanilla_ultimo.pt \
    --saida logs/e1_avaliacao.json

# 7. Amostras de texto
python -m src.generate --checkpoint checkpoints/e1_vanilla_ultimo.pt \
    --temperature 0.8 --top-k 50 --top-p 0.95

# 8. Extrapolação de contexto; o vanilla só aceita até 256 e o erro é anotado
python -m src.extrapolacao --checkpoint checkpoints/e4_rope_ultimo.pt \
    --contextos 128 256 512 1024 --saida logs/e4_extrapolacao.json
```

O runner retoma do último checkpoint sozinho. Para recomeçar do zero, apague
`WORK_DIR/checkpoints/` ou passe `--sem-retomar`.

## Checkpoints e hashes

Treinados em 20/11/2026, na T4 do Colab, 837 passos e 27.426.816 tokens cada. O
corpus tem partição de treino e de validação, sem partição de teste; todo BPB
reportado é de validação.

| Modelo | Arquivo | SHA-256 | Loss val | BPB validação |
|---|---|---|---|---|
| E1 | `e1_vanilla_ultimo.pt` | `51d1179b5007aa05176d43759e57e05a9d978e02524a32cc64a0298d137e08fb` | 4,9350 | 1,4231 |
| E2 | `e2_moe_h512_ultimo.pt` | `b42674cf8e32bdc41fe13b4325b0c44e257510f235db5e5b709d71f5d163b92d` | 4,8871 | 1,4093 |
| E3 | `e3_moe_h128_ultimo.pt` | `a354e2a40ba3a3e0959205f4bb19541c4fa5ab3066e00229e772c6eecf6b2723` | 4,9173 | 1,4180 |
| E4 | `e4_rope_ultimo.pt` | `3ae40e8cc60b96f85935c846bf2f858e778dbc33eb2a90d1c65d7d2832808a41` | 4,6892 | 1,3522 |

## Documentos

- [PLANO.md](PLANO.md): desenho experimental, etapas e riscos.
- [TASKS.md](TASKS.md): 50 tarefas em ordem, com status.
- [docs/enunciado-classroom.md](docs/enunciado-classroom.md): texto da atividade no Classroom.
- [docs/enunciado-oficial.pdf](docs/enunciado-oficial.pdf): enunciado completo da atividade.
- [docs/corpus-readme.md](docs/corpus-readme.md): descrição do corpus e do tokenizador.
- [notebooks/atividade1_inicio_professor.ipynb](notebooks/atividade1_inicio_professor.ipynb): material de partida do professor.
