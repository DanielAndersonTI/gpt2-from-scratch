"""Sementes e determinismo.

Um experimento comparativo só vale se a diferença entre dois treinos vier da
arquitetura. Tudo o que é aleatório precisa ser controlado: inicialização dos
pesos, ordem dos lotes, dropout e aumento de dados. Este módulo concentra esse
controle em um lugar só, para que o `train.py` não espalhe chamadas de seed.

Reprodutibilidade na GPU não é absoluta. Com `deterministic=True` o cuDNN
escolhe algoritmos determinísticos e desliga o autotuner, o que deixa as
execuções iguais na mesma máquina e na mesma versão de PyTorch. Entre versões
ou entre T4 e outra GPU, os últimos dígitos podem mudar. O teste de retomada
(T22) usa CPU, onde a igualdade é exata.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, strict: bool = False) -> None:
    """Fixa a semente em Python, NumPy e PyTorch.

    `deterministic` ajusta o cuDNN. `strict` liga o modo estrito do PyTorch,
    que transforma operação não determinística em erro; serve para os testes em
    CPU, não para o treino real, porque algumas operações da GPU não têm versão
    determinística e o treino pararia.
    """
    if seed < 0:
        raise ValueError(f"semente precisa ser não negativa, recebido {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cobre CPU e todas as GPUs de uma vez

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Sem autotuner: o mesmo algoritmo é escolhido em toda execução.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)


def make_generator(seed: int) -> torch.Generator:
    """Gerador próprio para o DataLoader, independente do estado global.

    Assim a ordem dos lotes não depende de quantas vezes o modelo foi
    inicializado antes, o que mantém a retomada previsível.
    """
    gerador = torch.Generator()
    gerador.manual_seed(seed)
    return gerador


def worker_init_fn(worker_id: int) -> None:
    """Semeia cada processo do DataLoader.

    O PyTorch já deriva uma semente por worker a partir do gerador do loader;
    aqui ela é propagada para `random` e `numpy`, que o worker também usa.
    """
    semente = torch.initial_seed() % 2**32
    np.random.seed(semente)
    random.seed(semente)


def capture_rng_state() -> dict[str, Any]:
    """Fotografa o estado de todos os geradores, para o checkpoint.

    Sem isso, retomar um treino repete a sequência de dropout a partir do
    estado inicial, e a curva de retomada não coincide com a curva contínua.
    """
    estado: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        estado["cuda"] = torch.cuda.get_rng_state_all()
    return estado


def restore_rng_state(estado: dict[str, Any]) -> None:
    """Devolve aos geradores o estado gravado por `capture_rng_state`."""
    random.setstate(estado["python"])
    np.random.set_state(estado["numpy"])
    torch.set_rng_state(estado["torch"])
    if "cuda" in estado and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(estado["cuda"])
