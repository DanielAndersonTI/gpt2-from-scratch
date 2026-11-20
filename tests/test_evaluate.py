"""Testes da avaliação.

O que está sendo garantido: a conta de bits por byte é a mesma do notebook do
professor, byte a byte, e o avaliador reconstrói o modelo a partir do
checkpoint sem depender do YAML.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from apoio import config_sintetica
from src.data import criar_dataloader, localizar_corpus, preparar, TokenizadorUFCG
from src.evaluate import avaliar_checkpoint, bits_por_byte, bytes_por_token
from src.model import GPTModel
from src.seeds import set_seed
from src.train import treinar


@torch.no_grad()
def bits_por_byte_do_notebook(model, loader, tabela, max_lotes=None):
    """Cópia da função do notebook do professor, para comparação."""
    model.eval()
    soma_loss, soma_bytes = 0.0, 0
    for k, (xb, yb) in enumerate(loader):
        if max_lotes is not None and k >= max_lotes:
            break
        logits = model(xb)
        soma_loss += F.cross_entropy(
            logits.flatten(0, 1), yb.flatten(), reduction="sum"
        ).item()
        soma_bytes += tabela[yb].sum().item()
    model.train()
    return soma_loss / (math.log(2) * soma_bytes)


def _cenario(raiz, vocab):
    cfg = config_sintetica(raiz, vocab, max_steps=4)
    corpus = localizar_corpus(cfg.run.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    ids = preparar("val", corpus, tokenizer, cfg.abspath(cfg.data.preprocessed_dir)).ids
    loader = criar_dataloader(
        ids,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.context_length,
        stride=cfg.data.stride,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        seed=cfg.run.seed,
    )
    set_seed(cfg.run.seed)
    model = GPTModel(cfg.model)
    model.eval()
    return cfg, tokenizer, loader, model


def test_bits_por_byte_igual_ao_notebook(projeto_sintetico):
    raiz, vocab = projeto_sintetico
    cfg, tokenizer, loader, model = _cenario(raiz, vocab)
    tabela = bytes_por_token(tokenizer)

    esperado = bits_por_byte_do_notebook(model, loader, tabela, max_lotes=3)
    obtido, perda, ppl, n_tokens, n_bytes = bits_por_byte(
        model, loader, torch.device("cpu"), tabela, max_lotes=3
    )

    assert obtido == pytest.approx(esperado, rel=1e-12)
    assert perda > 0 and ppl > 1
    assert n_tokens > 0 and n_bytes >= n_tokens  # todo token tem ao menos 1 byte


def test_eot_conta_como_um_byte(projeto_sintetico):
    _, vocab = projeto_sintetico
    corpus = localizar_corpus(projeto_sintetico[0])
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    tabela = bytes_por_token(tokenizer)
    assert int(tabela[tokenizer.eot_id]) == 1
    assert (tabela > 0).all()


def test_avaliar_checkpoint_reconstroi_e_grava_json(projeto_sintetico, tmp_path):
    raiz, vocab = projeto_sintetico
    cfg = config_sintetica(raiz, vocab, max_steps=4)

    resumo = treinar(cfg, log_every=2)
    saida = tmp_path / "avaliacao.json"

    resultado = avaliar_checkpoint(
        resumo["checkpoint"], split="val", device="cpu", max_lotes=2, saida=saida
    )

    assert resultado["experimento"] == "teste"
    assert resultado["passos_treinados"] == 4
    assert resultado["parametros"] > 0
    assert 0 < resultado["bpb"] < 100
    assert json.loads(saida.read_text(encoding="utf-8"))["bpb"] == resultado["bpb"]
