# Tarefas — GPT-2 do zero em português (pós-graduação)

Atualizado em 2026-11-20. Progresso: 39 de 50 concluídas, 1 em andamento. Prazo final: 14 de dezembro de 2026, 23:59.

Marcas: `[ ]` pendente, `[~]` em andamento, `[x]` feito, `[!]` bloqueado. A ordem é a de execução, e cada tarefa só começa quando a anterior fecha. O tempo de GPU aparece marcado; o restante é trabalho de escrita, teste e revisão.

Total: 48 a 51 h de trabalho e cerca de 3 h 40 de GPU.

## Fase 0 — Fundação (2 h 10)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T01 | Inicializar repositório git próprio e a estrutura de pastas (`src`, `configs`, `tests`, `logs`, `docs`) | repositório versionado dentro da pasta do projeto | 20 min | `[x]` |
| T02 | `.gitignore` para corpus, checkpoints, `*.npy` e `__pycache__` | dados nunca entram no git | 10 min | `[x]` |
| T03 | Schema de config YAML e loader com validação | um YAML por experimento, campos conferidos | 30 min | `[x]` |
| T04 | Sementes e determinismo em Python, NumPy, PyTorch e dataloader | mesma semente, mesma ordem de lotes | 20 min | `[x]` |
| T05 | Logger CSV com passo, loss de treino e validação, LR, tokens/s e memória de pico | `logs/*.csv` incremental | 30 min | `[x]` |
| T06 | README esqueleto com os comandos previstos e a seção de hash | README navegável desde o início | 20 min | `[x]` |

## Fase 1 — Dados (3 h 10)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T07 | Validar o corpus e conferir as contagens do professor | 27.451.491 tokens de treino, 1.576.142 de validação | 20 min | `[x]` |
| T08 | Wrapper do tokenizador de 16k e fertilidade contra a referência (1,37) e contra o `tiktoken` gpt2 | tabela de fertilidade | 40 min | `[~]` |
| T09 | Pré-tokenização de treino e validação para `.npy` em `uint16` | ids em disco, reutilizáveis | 20 min | `[x]` |
| T10 | Offsets de documento e junção com metadados (`sigla_centro`) | cada janela sabe de que documento veio | 50 min | `[x]` |
| T11 | Dataset de janelas + teste de igualdade com o `GPTDatasetV1` do livro | janelas idênticas, verificado | 30 min | `[x]` |
| T12 | Dataloader reproduzível com semente por época e teste de retomada | parar e continuar sem repetir lote | 30 min | `[x]` |

## Fase 2 — Modelo (4 h 40)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T13 | Portar o `GPTModel` do livro para `src/model.py`, com feed-forward e posição plugáveis | modelo dirigido por config | 2 h 30 | `[x]` |
| T14 | Testes de shape, de máscara causal e de fluxo de gradiente | `tests/` verde | 50 min | `[x]` |
| T15 | Conferir 11.414.016 parâmetros e loss inicial próxima de 9,68 | números do professor reproduzidos | 30 min | `[x]` |
| T16 | Geração simples com o modelo não treinado | sanidade do caminho de inferência | 30 min | `[x]` |

## Fase 3 — Treino (6 h 40)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T17 | Loop do Apêndice D: warmup linear, cosine, clipping e AdamW com weight decay só nos pesos | loop final | 1 h 30 | `[x]` |
| T18 | Precisão mista fp16 com GradScaler | treino estável na T4 | 40 min | `[x]` |
| T19 | Atenção via `scaled_dot_product_attention` | FlashAttention acionado | 30 min | `[x]` |
| T20 | AdamW `fused` | otimizador em uma chamada | 20 min | `[x]` |
| T21 | `torch.compile` condicional por config | liga e desliga sem tocar no código | 30 min | `[x]` |
| T22 | Checkpoint completo (modelo, otimizador, scheduler, scaler, passo, época, lote) e retomada | continuidade bit a bit verificada | 1 h | `[x]` |
| T23 | Avaliação a cada 10% do treino e amostra de texto no log | curva e amostras no CSV | 40 min | `[x]` |
| T24 | Runner autônomo de sessão, com retomada automática e parada antes do limite | treino sobrevive a queda | 1 h | `[x]` |
| T25 | Teste de fumaça: 100 passos e conferência do log | pipeline validado ponta a ponta | 30 min | `[x]` |

## Fase 4 — Benchmark (2 h)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T26 | Script de benchmark: 200 passos em cinco configurações, tokens/s e memória de pico | `benchmark.csv` | 1 h | `[x]` |
| T27 | Rodar o benchmark no Colab e montar a tabela antes e depois | tabela do artigo | 1 h, 35 min GPU | `[x]` |

## Fase 5 — E1 vanilla (1 h 20)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T28 | Treinar o E1 completo, 27,4M tokens | checkpoint do baseline | 40 min GPU | `[x]` |
| T29 | BPB, perplexidade, três prompts e hash SHA-256 do checkpoint | números do E1 | 30 min | `[x]` |
| T30 | Commitar log, curva e hash | checkpoint rastreado | 10 min | `[x]` |

## Fase 6 — MoE (6 h 20)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T31 | `src/moe.py`: roteador top-2, oito especialistas e loss auxiliar de 0,01 | camada MoE | 2 h | `[x]` |
| T32 | Testes: parâmetros ativos e totais da tabela, shapes e aux loss nula com roteamento uniforme | `tests/` verde | 50 min | `[x]` |
| T33 | Treinar E2 (h=512) e E3 (h=128) | dois checkpoints | 1 h, 1 h 45 GPU | `[x]` |
| T34 | Utilização por especialista e entropia do roteador ao longo do treino | gráfico de colapso | 1 h | `[x]` |
| T35 | Roteamento por tipo de token e por `sigla_centro` | tabela de especialização | 1 h | `[x]` |
| T36 | Figuras e tabela de análise do MoE | material do artigo | 30 min | `[ ]` |

## Fase 7 — RoPE (4 h)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T37 | `src/rope.py` e testes: norma preservada e produto interno dependente da distância | rotação verificada | 1 h 30 | `[x]` |
| T38 | Integrar RoPE em Q e K e treinar o E4 | checkpoint do E4 | 1 h, 40 min GPU | `[x]` |
| T39 | Extrapolação em 128, 256, 512 e 1024 com `src/extrapolacao.py` | resultado da modificação | 1 h 30 | `[x]` |

## Fase 8 — Avaliação e entrega técnica (4 h 30)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T40 | `evaluate.py` com o mesmo BPB do professor e teste de igualdade contra a função do notebook | script que o professor roda na oral | 1 h | `[x]` |
| T41 | `generate.py` com temperatura, top-k e top-p | amostras dos três prompts | 1 h | `[x]` |
| T42 | README final: um comando por experimento, tabela de resultados e hashes | reprodutibilidade documentada | 1 h 30 | `[ ]` |
| T43 | Rodar do zero em sessão limpa e conferir os números | entrega verificada | 1 h | `[ ]` |

## Fase 9 — Artigo e oral (12 a 16 h)

| ID | Tarefa | Entregável | Tempo | Status |
|---|---|---|---|---|
| T44 | Esqueleto do artigo com hipóteses explícitas antes dos resultados | estrutura aprovada | 1 h | `[ ]` |
| T45 | Método e desenho experimental | seção escrita | 2 h | `[ ]` |
| T46 | Resultados e figuras | seção escrita | 3 h | `[ ]` |
| T47 | Trabalhos relacionados, limitações e ameaças à validade | seção escrita | 2 h | `[ ]` |
| T48 | Revisão, checklist e compilação no modelo SBC | PDF de até 8 páginas | 2 h | `[ ]` |
| T49 | Slides de 10 minutos | apresentação | 1 h 30 | `[ ]` |
| T50 | Ensaio da oral com as perguntas da rubrica | defesa preparada | 1 h 30 | `[ ]` |

## Estimativa com automação

Tempo meu é o de escrever, comentar e testar o código. GPU é a sessão do Colab
que você inicia. Seu tempo é revisão, comando único e commit. O caminho crítico
não é o código: são as sessões de GPU, porque o Colab gratuito dá cerca de 3 a
4 h por dia e o total é de 3 h 40.

| Fase | Meu tempo | GPU | Seu tempo | Bloqueio |
|---|---|---|---|---|
| 0. Fundação | 25 min | 0 | 5 min | nenhum (feita) |
| 1. Dados | 60 a 90 min | 0 | 15 min | corpus e tokenizador locais |
| 2. Modelo | 60 a 90 min | 0 | 15 min | torch CPU instalado |
| 3. Treino | 90 a 120 min | 10 min de fumaça | 20 min | F2 |
| 4. Benchmark | 30 min | 35 min | 10 min | F3, sessão no Colab |
| 5. E1 vanilla | 20 min | 40 min | 10 min | F4 |
| 6. MoE | 90 a 120 min | 1 h 45 | 20 min | F5 |
| 7. RoPE | 60 a 90 min | 40 min | 15 min | F6 |
| 8. Avaliação e entrega | 90 min | 20 min | 20 min | F7 |
| 9. Artigo e oral | 6 a 8 h de redação | 0 | 6 a 8 h de revisão | F8 |

A estimativa supunha cerca de 11 mil tokens por segundo. A T4 mediu entre 82 e
110 mil, porque a configuração final usa precisão mista e a atenção pelo kernel
do PyTorch. Os quatro treinos que pediam 3 h 40 de GPU levaram 27 minutos no
total.

### Calendário real

| Data | Entrega |
|---|---|
| 20/11 | fases 1 a 7 fechadas, quatro treinos e benchmark concluídos |
| 22/11 | BPB, hashes, figuras e análise de utilização do MoE; relatório até a fase 7 |
| 27/11 | artigo completo para revisão |
| 03/12 | revisão do artigo e compilação no modelo SBC |
| 08/12 | slides e ensaio da oral |
| 12/12 | margem para imprevisto e revisão final |
| 14/12 | entrega |

## Protocolo de execução noturna

Os treinos rodam sem supervisão, em sessões iniciadas por você e retomadas por mim. O contrato é este.

1. Comando único: `python -m src.run_session --config configs/e1_vanilla.yaml --max-hours 3`.
2. Checkpoint a cada 500 passos em `WORK_DIR/checkpoints/`, no Drive, com modelo, otimizador, scheduler, scaler, passo, época e lote.
3. Ao iniciar, o runner lê o último checkpoint e continua. Nada recomeça do zero.
4. Log em CSV com `flush` a cada linha. Queda de sessão não perde registro.
5. O runner para sozinho antes do limite de tempo da sessão, para que o último checkpoint seja gravado.
6. Na manhã seguinte eu leio o log, reporto curva, tokens/s, BPB parcial e qualquer erro, e digo qual é o próximo comando.
7. Se a GPU cair antes do primeiro checkpoint, perdem-se no máximo 500 passos, cerca de quatro minutos.

## Relatório incremental

Cada fase fecha com a escrita da sua parte no `RELATORIO.md`, na voz acadêmica do Daniel. O relatório é a fonte do artigo final, não um documento paralelo: na fase 9 ele é reorganizado no formato SBC, com as hipóteses movidas para antes dos resultados.

| Fase | Seção do relatório |
|---|---|
| 0 | contexto, objetivos e decisões de engenharia |
| 1 | corpus, tokenizador e fertilidade |
| 2 | arquitetura e contagem de parâmetros |
| 3 | protocolo de treino e otimizações |
| 4 | custo computacional antes e depois |
| 5 | baseline: curvas, perplexidade e BPB |
| 6 | MoE: desenho, utilização e especialização |
| 7 | RoPE: desenho e extrapolação |
| 8 | reprodutibilidade e limitações |
| 9 | versão final do artigo |

Ao fechar cada fase, eu escrevo a seção, você revisa, e o registro abaixo recebe a data.

## Ambiente e acessos (verificado em 2026-11-19)

- Python 3.12 local, com PyTorch CPU instalado em 2026-11-20 para rodar os testes sem depender do Colab. Sem GPU.
- `tokenizers`, `numpy`, `pandas`, `pyarrow` e `matplotlib` já instalados.
- 18 GB livres em `C:`.
- O link do corpus no Drive do professor responde sem autenticação (HTTP 206, ZIP válido). O download funciona no Colab e localmente.
- O Colab não oferece acesso programático ao runtime gratuito. Não consigo iniciar sessão nem operar a GPU. O fluxo é: você inicia a sessão e roda o comando; eu preparo o notebook e os scripts, e analiso os logs que você trouxer para a pasta do projeto.

## Registro

- 2026-11-19: plano de execução e lista de tarefas criados. Nenhuma tarefa iniciada.
- 2026-11-19: notebook do professor renderizado em Markdown e reconstruído em `.ipynb` válido. Ambiente e acessos verificados. Regra do relatório incremental definida.
- 2026-11-19: T01, T02 e T06 concluídas. Repositório inicializado na pasta do projeto, ligado a `github.com/DanielAndersonTI/gpt2-from-scratch`. Estrutura criada, corpus fora do Git, README esqueleto escrito. A pasta foi excluída do repositório externo `I.T` em `.git/info/exclude`. Colab confirmado com T4 e PyTorch 2.14.1. Sincronização local do Drive ainda não visível no Explorer.
- 2026-11-19: T03, T04, T05 e T07 a T12 concluídas, T08 em andamento. `src/config.py`, `src/seeds.py`, `src/logging_utils.py` e `src/data.py` escritos; 64 testes passando em CPU. Pré-tokenização local conferida contra os números do professor: 20.722 documentos e 27.451.491 tokens no treino, 1.152 e 1.576.142 na validação, 107.232 janelas de 256 no treino. A ordem dos documentos no `.txt` foi verificada contra o `.parquet` documento a documento, o que autoriza a junção por índice. Fertilidade: 1,373 no treino e 1,374 na validação completa; o 1,42 do notebook vem de uma amostra de 200 mil caracteres, não da validação inteira.
- 2026-11-20: primeira sessão de GPU, no Colab com a T4, cerca de 80 min no total. T18 a T21, T27, T28, T33 e T38 concluídas. Os quatro treinos fecharam 837 passos e 27.426.816 tokens cada, em 5 a 7 minutos, entre 82 e 110 mil tokens por segundo. Loss de validação: E1 4,9430; E2 4,8955; E3 4,9256; E4 4,6977. Benchmark por variante (tokens/s e memória de pico): base 46.909 e 14.291 MB, amp 85.789 e 11.118 MB, sdpa 49.266 e 13.085 MB, fused 46.298 e 14.291 MB, compile 58.041 e 12.078 MB, todas 178.688 e 7.832 MB. O conjunto otimizado é 3,81 vezes mais rápido que a base e usa 45% menos memória. A variante base só caberia com `expandable_segments:True`: sem precisão mista, a matriz de logits de um lote de 128 por 256 tokens por 16.000 de vocabulário ocupa 2,1 GB.
- 2026-11-20: T13 a T17, T22 a T26, T31, T32, T37, T40 e T41 concluídas em CPU. `src/model.py`, `src/moe.py`, `src/rope.py`, `src/train.py`, `src/benchmark.py`, `src/evaluate.py`, `src/generate.py` e `src/run_session.py` escritos, com 110 testes passando em CPU. Contagem de parâmetros reproduzida nas quatro configurações: E1 11.414.016, E2 17.733.152 com 11.423.264 ativos, E3 11.429.408 com 9.847.328 ativos, E4 11.348.480. Os especialistas têm viés, o que explica os números. A retomada foi verificada bit a bit: interromper e continuar produz os mesmos pesos de um treino contínuo. A fumaça rodou 30 passos com loss de 9,60 para 8,83, e um treino de 4 passos com MoE conferiu o encadeamento da perda auxiliar. O benchmark foi separado do log de treino em `logs/benchmark.csv`.
- 2026-11-20: segunda sessão de GPU. O E4 foi retreinado do zero no Colab, porque o checkpoint da primeira sessão não chegou ao Drive; mesma semente, mesmo orçamento, 6 minutos. T34 e T35 concluídas: `src/analise_moe.py` mediu utilização top-1, entropia normalizada, despacho por token e por centro nas duas variantes. Utilização entre 0,092 e 0,159 por bloco nas duas, entropia entre 0,69 e 0,81, sem colapso. A especialização por token é forte mesmo com entropia alta: na E2, no bloco 0, a vírgula manda 0,082 do seu tráfego para o especialista 0 e 0,0015 para o especialista 3. BPB de validação medido nos quatro: E1 1,4231, E2 1,4093, E3 1,4180, E4 1,3522, sobre 1.575.936 tokens e 7.884.234 bytes. As avaliações em CPU do E2 e do E3 conferiram com as da GPU até a sétima casa decimal.
- 2026-11-20: corrigida a ponderação da validação no laço de treino. `avaliar` fazia média das médias por lote, e o último lote da validação tem 12 janelas em 49, dez vezes o peso que os tokens dele representam. A média agora é por token, igual à de `evaluate.py`. Os CSVs em `logs/` foram gerados antes da correção e leem cerca de 0,008 nats acima do valor correto no ponto final; a diferença foi medida lote a lote (o último vale 5,3709 contra 4,9350 do resto). T43 regenera os CSVs com o código novo.
- 2026-11-20: `src/extrapolacao.py` escrito para a T39. As janelas têm passo igual ao contexto e o lote é limitado ao número de janelas, senão `drop_last` descarta tudo. A atenção manual ganhou máscara refeita quando a janela passa do contexto de treino; o caminho SDPA já funcionava, porque a causalidade vai no kernel. Dois testes novos: 64 tokens num modelo configurado para 32, nos dois caminhos de atenção, e o script recusando o modelo com posição aprendida com erro anotado em vez de queda.
- 2026-11-20: T39 concluída na GPU. Extrapolação medida com janelas do tamanho do contexto, sem sobreposição: 1.572.864 tokens e 7.870.583 bytes por linha. RoPE: loss 4,7033 em 128, 4,6883 em 256, 4,7053 em 512 e 4,7591 em 1024, ou 0,4% e 1,5% acima do valor de treino. O vanilla vai a 1,4248 em 128 e 1,4226 em 256, e recusa 512 com mensagem, porque o embedding posicional aprendido não tem linha além da 256. O RoPE, portanto, não ganha com a janela maior, já que não foi treinado nela, e também não quebra. Análise de roteamento regravada com `--saida`: `logs/e2_analise_moe.json` e `logs/e3_analise_moe.json`. Hash do E4: `3ae40e8c...28808a41`.
- 2026-11-20: T29 e T30 fechadas. O BPB do E1, a perplexidade, as três amostras e o hash já existiam; faltava só o registro. Os CSVs, as amostras e os JSONs de avaliação estão versionados em `logs/`.
