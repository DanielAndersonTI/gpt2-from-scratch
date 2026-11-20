# Configurações de experimento

Um arquivo YAML por treino. O schema é definido na T03, em `src/config.py`, e
validado na carga: campo desconhecido ou ausente interrompe o experimento em vez
de usar um padrão escondido.

Arquivos previstos:

| Arquivo | Experimento |
|---|---|
| `e1_vanilla.yaml` | baseline: GPT-2 do livro, pipeline otimizado |
| `e2_moe_h512.yaml` | MoE com 8 especialistas, top-2, h=512 (mesmo ativo por token) |
| `e3_moe_h128.yaml` | MoE com 8 especialistas, top-2, h=128 (mesmo total de parâmetros) |
| `e4_rope.yaml` | RoPE no lugar do embedding posicional aprendido |
| `benchmark.yaml` | rodadas curtas de 200 passos para medir tokens/s e memória |

Regra comum: mesmo tokenizador, mesmo contexto, mesmo orçamento de tokens, mesma
semente e mesmo cronograma de learning rate. A única diferença entre os arquivos
é a arquitetura.
