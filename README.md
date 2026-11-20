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
| 0. Fundação | em andamento |
| 1. Dados | pendente |
| 2. Modelo | pendente |
| 3. Treino | pendente |
| 4. Benchmark | pendente |
| 5. E1 vanilla | pendente |
| 6. MoE | pendente |
| 7. RoPE | pendente |
| 8. Avaliação | pendente |
| 9. Artigo e oral | pendente |

O acompanhamento detalhado, tarefa por tarefa, está em [TASKS.md](TASKS.md). O
desenho experimental e os tempos estão em [PLANO.md](PLANO.md).

## Experimentos

Todos usam o tokenizador de referência de 16k, contexto de 256 tokens,
27,4M tokens de treino e semente 123.

| ID | Modelo | Parâmetros totais / ativos | Checkpoint | Log | BPB val |
|---|---|---|---|---|---|
| E1 | vanilla otimizado | 11,41M / 11,41M | pendente | pendente | pendente |
| E2 | MoE, 8 especialistas, top-2, h=512 | 17,73M / 11,42M | pendente | pendente | pendente |
| E3 | MoE, 8 especialistas, top-2, h=128 | 11,43M / 9,85M | pendente | pendente | pendente |
| E4 | RoPE | 11,35M / 11,35M | pendente | pendente | pendente |

## Estrutura

```
src/          modelo, dados, treino, avaliação
configs/      um YAML por experimento
tests/        testes de shapes, máscara causal, parâmetros e retomada
logs/         CSV de cada treino, versionados
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

Preenchido na T42, ao final dos experimentos. Cada experimento terá um comando
único, por exemplo:

```bash
python -m src.train --config configs/e1_vanilla.yaml
```

## Checkpoints e hashes

Preenchido ao final de cada treino. O BPB de teste só vale para o checkpoint cujo
hash estiver registrado aqui.

| Modelo | Arquivo | SHA-256 | BPB validação | BPB teste |
|---|---|---|---|---|
| E1 | pendente | pendente | pendente | pendente |
| E2 | pendente | pendente | pendente | pendente |
| E3 | pendente | pendente | pendente | pendente |
| E4 | pendente | pendente | pendente | pendente |

## Documentos

- [PLANO.md](PLANO.md): desenho experimental, etapas e riscos.
- [TASKS.md](TASKS.md): 50 tarefas em ordem, com status.
- [docs/enunciado-classroom.md](docs/enunciado-classroom.md): texto da atividade no Classroom.
- [docs/enunciado-oficial.pdf](docs/enunciado-oficial.pdf): enunciado completo da atividade.
- [docs/corpus-readme.md](docs/corpus-readme.md): descrição do corpus e do tokenizador.
- [notebooks/atividade1_inicio_professor.ipynb](notebooks/atividade1_inicio_professor.ipynb): material de partida do professor.
