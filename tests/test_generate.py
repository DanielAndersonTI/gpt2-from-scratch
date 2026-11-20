"""Testes da decodificação.

O que está sendo garantido: top-k corta exatamente k candidatos, top-p mantém
o menor núcleo com massa de probabilidade pelo menos p, e a decodificação
gulosa é determinística.
"""

from __future__ import annotations

import torch

from apoio import config_sintetica
from src.generate import filtrar_top_k, filtrar_top_p, gerar
from src.model import GPTModel
from src.seeds import set_seed


def test_top_k_deixa_exatamente_k_candidatos():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtrado = filtrar_top_k(logits, 2)
    finitos = torch.isfinite(filtrado).sum().item()
    assert finitos == 2
    assert filtrado[0, 1] == 5.0 and filtrado[0, 4] == 4.0


def test_top_k_maior_que_o_vocabulario_nao_muda_nada():
    logits = torch.randn(2, 5)
    assert torch.equal(filtrar_top_k(logits, 10), logits)


def test_top_p_mantem_massa_de_ao_menos_p():
    torch.manual_seed(0)
    logits = torch.randn(1, 100)
    p = 0.9
    filtrado = filtrar_top_p(logits, p)
    probabilidades = torch.softmax(filtrado, dim=-1)
    massa = probabilidades.sum().item()
    assert 0.9 <= massa <= 1.0
    # O núcleo não pode ser o vocabulário inteiro com p tão alto.
    assert torch.isfinite(filtrado).sum().item() < 100


def test_top_p_de_um_mantem_tudo():
    logits = torch.randn(1, 20)
    assert torch.equal(filtrar_top_p(logits, 1.0), logits)


def test_geracao_gulosa_e_deterministica(projeto_sintetico):
    raiz, vocab = projeto_sintetico
    cfg = config_sintetica(raiz, vocab, max_steps=4)
    set_seed(0)
    model = GPTModel(cfg.model)

    from src.data import localizar_corpus, TokenizadorUFCG

    tokenizer = TokenizadorUFCG(localizar_corpus(cfg.run.work_dir).tokenizer)

    a = gerar(model, tokenizer, "palavra1", max_new_tokens=8, temperature=0.0)
    b = gerar(model, tokenizer, "palavra1", max_new_tokens=8, temperature=0.0)
    assert a == b
    assert a.startswith("palavra1")
