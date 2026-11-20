"""Pacote do projeto: GPT-2 do zero no corpus de introduções da UFCG.

Módulos previstos:
    data.py      pré-tokenização, janelas e dataloaders
    model.py     GPT com feed-forward e codificação de posição plugáveis
    moe.py       roteador top-2 e especialistas
    rope.py      rotação aplicada em Q e K
    train.py     loop de treino, precisão mista, checkpoint e logs
    evaluate.py  bits por byte
    generate.py  decodificação por temperatura, top-k e top-p
"""
