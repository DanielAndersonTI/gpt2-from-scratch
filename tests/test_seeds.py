"""Testes de determinismo.

O que está sendo garantido: mesma semente produz a mesma sequência em Python,
NumPy e PyTorch; o gerador do DataLoader é independente do estado global; e o
estado dos geradores pode ser gravado e devolvido, que é o que a retomada de
treino precisa.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.seeds import capture_rng_state, make_generator, restore_rng_state, set_seed


def test_mesma_semente_reproduz_numpy_e_torch():
    set_seed(123)
    a = (np.random.rand(4), torch.rand(4))
    set_seed(123)
    b = (np.random.rand(4), torch.rand(4))
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_mesma_semente_reproduz_python():
    set_seed(7)
    a = [random.random() for _ in range(3)]
    set_seed(7)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_sementes_diferentes_divergem():
    set_seed(1)
    a = torch.rand(8)
    set_seed(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_semente_negativa_falha():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_gerador_reproduz_a_mesma_ordem():
    g1 = make_generator(42)
    g2 = make_generator(42)
    assert torch.equal(torch.randperm(16, generator=g1), torch.randperm(16, generator=g2))


def test_gerador_nao_e_afetado_pelo_estado_global():
    # O primeiro sorteio de dois geradores recém-criados com a mesma semente
    # precisa ser igual, mesmo que o estado global tenha avançado no meio.
    g = make_generator(5)
    _ = torch.rand(100)  # avança apenas o estado global
    primeira = torch.randperm(10, generator=g)
    segunda = torch.randperm(10, generator=make_generator(5))
    assert torch.equal(primeira, segunda)


def test_estado_dos_geradores_volta_ao_ponto_gravado():
    set_seed(11)
    _ = random.random(), np.random.rand(), torch.rand(1)

    estado = capture_rng_state()
    depois = (random.random(), np.random.rand(), torch.rand(3))

    restore_rng_state(estado)
    de_novo = (random.random(), np.random.rand(), torch.rand(3))

    assert depois[0] == de_novo[0]
    assert depois[1] == de_novo[1]
    assert torch.equal(depois[2], de_novo[2])
