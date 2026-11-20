# Plano de execução — GPT-2 do zero em português (pós-graduação)

Atividade final da disciplina de LLM. Trilha de pós, individual. Quatro treinos
completos no corpus de introduções da UFCG, artigo SBC de até 8 páginas e oral.

## 1. Regra do jogo

Um treino só vale se muda uma coisa de cada vez. Os quatro treinos usam o mesmo
tokenizador de referência de 16k, o mesmo contexto de 256 tokens, o mesmo
orçamento de 27,4M tokens, a mesma semente e o mesmo cronograma de learning
rate. A única coisa que muda é a arquitetura.

Matriz de experimentos:

| Treino | Modelo | Parâmetros totais / ativos | Pergunta |
|---|---|---|---|
| E1 | vanilla otimizado | 11,41M / 11,41M | baseline |
| E2 | MoE, 8 especialistas, top-2, h=512 | 17,73M / 11,42M | o que se ganha com mais capacidade pelo mesmo custo por token |
| E3 | MoE, 8 especialistas, top-2, h=128 | 11,43M / 9,85M | o que se perde com o mesmo total e menos ativos |
| E4 | RoPE no lugar do posicional aprendido | 11,35M / 11,35M | posição relativa melhora a modelagem e permite extrapolar |

A conta fecha na aritmética: o FFN denso tem 525.568 parâmetros por bloco.
Oito especialistas de h=512 somam 2.103.296 por bloco, com 525.824 ativos no
top-2. Oito de h=128 somam 527.360 totais e 131.840 ativos. O roteador entra na
contagem.

## 2. Repositório

```
gpt2-ufcg/
  configs/          # um YAML por experimento (e1..e4, benchmark)
  src/
    tokenizer.py    # wrapper do BPE de 16k
    data.py         # pré-tokenização, janelas, dataloaders
    model.py        # GPT com blocos plugáveis
    moe.py          # roteador top-2, especialistas, loss auxiliar
    rope.py         # rotação aplicada em Q e K
    train.py        # loop do Apêndice D, AMP, checkpoint, CSV
    evaluate.py     # BPB de um arquivo de texto
    generate.py     # temperatura, top-k, top-p
    analysis.py     # utilização dos especialistas, roteamento por token e centro
  tests/            # shapes, máscara causal, parâmetros, retomada
  logs/             # CSV versionado, um por treino
  README.md         # um comando por experimento e hash dos checkpoints
```

## 3. Etapas, entregáveis e tempo

Tempo de trabalho inclui escrever, testar e revisar. Tempo de GPU é o que roda
no Colab.

| Etapa | O que é | Entregável | Trabalho | GPU |
|---|---|---|---|---|
| 0. Fundação | estrutura, configs, sementes, logger CSV | repositório vazio que roda | 2 h | 0 |
| 1. Dados | tokenizar o corpus uma vez, janelas de 256, dataloaders determinísticos | `train_ids.npy`, teste de igualdade com o dataset do livro | 3 h | 5 min CPU |
| 2. Modelo | GPT do livro adaptado, blocos plugáveis, testes de shape e de máscara | contagem de parâmetros batendo, loss inicial ≈ 9,68 | 5 h | 0 |
| 3. Treino | loop do Apêndice D, fp16 com GradScaler, SDPA, AdamW fused, compile, warmup e cosine, clipping, checkpoint e retomada, validação a cada 10% | treino interrompível e retomável | 6 h | 0 |
| 4. Benchmark | 200 passos em cinco configurações, medindo tokens/s e memória | tabela antes e depois de cada otimização | 2 h | 35 min |
| 5. E1 vanilla | treino completo, curvas, perplexidade, BPB, três prompts | checkpoint, log, curva | 1 h | 35 min |
| 6. MoE | roteador, loss auxiliar, treinos E2 e E3, análise de utilização | dois checkpoints, gráficos de roteamento | 6 h | 1 h 45 |
| 7. RoPE | rotação em Q e K, treino E4, avaliação em 256, 512 e 1024 | checkpoint, curvas de extrapolação | 4 h | 40 min |
| 8. Entrega | `evaluate.py`, README, hashes, artigo, slides, ensaio da oral | pacote completo | 14 a 18 h | 0 |

Total: 43 a 47 h de trabalho e cerca de 3 h 40 de GPU. Em quatro semanas, dá
cerca de 11 h por semana.

## 4. Caminho rápido para o checkpoint da semana 1

O checkpoint pede apenas a curva do vanilla e o BPB de validação. Nada de MoE
nem de RoPE.

| Dia | Etapa | Horas |
|---|---|---|
| 1 | 0, 1 e 2 | 10 |
| 2 | 3 e começo do E1 | 8 |
| 3 | fim do E1, BPB, README com comando e hash | 5 |

Mantém-se o loop definitivo desde o início. Escrever um loop descartável para o
checkpoint custaria as mesmas horas e seria jogado fora.

## 5. Como cada parte funciona

### Dados

O corpus vem em texto com `<|endoftext|>` entre documentos. Tokenizamos cada
documento uma única vez e salvamos os ids em `uint16`, que cabe metade da
memória de `int32`. O dataset corta janelas de 256 tokens com passo 256, sem
sobreposição, e o alvo é a janela deslocada uma posição. O dataloader usa
gerador com semente fixa, para que a retomada veja a mesma ordem de lotes.

### Modelo

O GPT do livro é uma pilha de blocos. Cada bloco aplica normalização, atenção
causal multi-cabeça, conexão residual, normalização de novo, feed-forward com
GELU e outra residual. A máscara causal é triangular porque o token na posição
t precisa prever t+1 sem enxergar o futuro. O embedding posicional aprendido
tem uma linha por posição, o que impede o modelo de rodar além de 256 tokens.

Para não reescrever o modelo a cada variante, o bloco recebe dois componentes
plugáveis: o feed-forward (denso ou MoE) e a codificação de posição (aprendida
ou RoPE). Cada experimento é um YAML, não um arquivo novo. Isso reduz o risco de
as variantes divergirem em algo que não deveria mudar.

### Treino

Cinco decisões definem o pipeline.

1. **fp16 com GradScaler.** A T4 não tem bf16 nativo. O escalonador multiplica a
   loss antes do backward para que gradientes pequenos não virem zero em fp16.
2. **SDPA.** A atenção usa `scaled_dot_product_attention`, que aciona o kernel
   FlashAttention e reduz memória e tempo.
3. **AdamW fused.** O otimizador em uma única operação de GPU.
4. **warmup linear e cosine decay.** O LR sobe nos primeiros passos e desce em
   cosseno, como no Apêndice D. Clipping em 1,0.
5. **Checkpoint e retomada.** A cada 500 passos gravamos modelo, otimizador,
   scheduler e posição do data loader no Drive. Sessão gratuita cai, e sem isso
   o treino recomeça do zero.

Os logs vão para CSV com passo, loss de treino, loss de validação, LR, tokens/s
e memória de pico. Todo número do artigo sai desse arquivo.

### MoE

O feed-forward denso vira oito especialistas e um roteador. Para cada token, o
roteador produz uma probabilidade por especialista; os dois maiores processam o
token e a saída é a soma ponderada pelas probabilidades. A loss auxiliar de
balanceamento, com coeficiente 0,01, penaliza a concentração de tokens em poucos
especialistas. Sem ela, o roteador colapsa: dois especialistas recebem quase
tudo e os outros seis não aprendem.

A análise olha três coisas: a fração de tokens por especialista ao longo do
treino, o tipo de token que cada um recebe (palavra funcional, número, citação,
pontuação) e a distribuição condicionada ao `sigla_centro` do documento. A
última exige propagar o id do documento até a janela, para saber de que centro
veio cada token.

### RoPE

O RoPE gira pares de dimensões de Q e K por um ângulo proporcional à posição. O
produto interno entre Q e K passa a depender só da distância entre os tokens,
não da posição absoluta. Por isso não há embedding posicional e o modelo pode,
em princípio, operar em contextos maiores que os vistos no treino. A avaliação
mede isso: treino em 256 e teste em 256, 512 e 1024 sem ajuste. O vanilla não
consegue nem rodar em 512, porque o embedding aprendido tem 256 linhas. Essa
assimetria é resultado e precisa aparecer no artigo.

### Métricas

- **Loss** é entropia cruzada média por token, em nats.
- **Perplexidade** é `exp(loss)`. Depende do tokenizador e só compara modelos que
  usam o mesmo.
- **BPB** é a loss convertida em bits por byte do texto original:
  `loss / ln(2) × tokens / bytes`. É a métrica que compara tokenizadores
  diferentes e a que o professor usa no teste.
- **Tokens por segundo e memória de pico** vêm de rodadas curtas de 200 passos.

### Avaliação

`evaluate.py` recebe um arquivo de texto e um checkpoint e devolve o BPB. É o
script que o professor roda na oral, sobre o checkpoint cujo hash está no README.
Precisa funcionar antes do prazo, testado na validação.

## 6. Opções para acelerar sem perder qualidade

1. **Bloco plugável desde a primeira versão.** Cada variante vira configuração.
   Evita reescrever e evita divergência entre treinos.
2. **Pré-tokenizar uma vez e guardar no Drive.** Economiza minutos por sessão.
3. **Checkpoint a cada 500 passos desde o primeiro treino.** Sessão que cai não
   custa retrabalho.
4. **Rodar um treino por sessão do Colab.** Sessões de 2 h, sem disputa de
   memória e com log fechado ao final.
5. **`torch.compile` só se der ganho.** Se não der, o artigo reporta a ausência
   do ganho, o que também pontua.
6. **Kaggle como reserva gratuita.** Mesmo código, 30 h semanais de T4 e output
   persistente. Serve se o Colab bloquear a GPU.
7. **Decodificação com KV cache.** Só para gerar as amostras dos três prompts
   com rapidez; não muda a loss.

## 7. Controle de qualidade

- Testes de shape em cada bloco, de máscara causal (posição futura com peso
  zero) e de contagem de parâmetros contra a aritmética da seção 1.
- Loss inicial próxima de ln(16.000) = 9,68. Um valor muito distante indica erro
  nos dados ou no modelo.
- Mesma semente em todos os treinos e dataloader determinístico.
- Um comando por experimento, logs em CSV no repositório e hash SHA-256 de cada
  checkpoint no README.
- Cautela com diferenças pequenas: uma rodada por configuração não permite
  afirmar significância. O artigo discute isso e se apoia na forma das curvas.

## 8. Orçamento no Colab gratuito

Quatro treinos e o benchmark somam cerca de 3 h 40 de GPU. O gratuito cobre
isso, desde que o trabalho seja dividido em sessões curtas e o checkpoint vá
para o Drive. O risco real é o bloqueio temporário de GPU; a reserva é o Kaggle
gratuito com o mesmo código.
