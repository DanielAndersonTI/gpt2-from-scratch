"""Testes do loop de treino, de ponta a ponta, em CPU.

O que está sendo garantido: o pipeline inteiro roda (dados, modelo, loop,
avaliação, checkpoint, CSV), e retomar de um checkpoint produz exatamente o
mesmo modelo de um treino contínuo. O segundo teste é o mais importante do
repositório: sem ele, o protocolo de execução noturna é uma promessa sem prova.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apoio import config_sintetica, escrever_corpus_sintetico
from src.train import treinar


def _preparar_projeto(raiz: Path, vocab: int) -> Path:
    escrever_corpus_sintetico(raiz / "corpus")
    return raiz


def test_treino_curto_grava_log_checkpoint_e_resumo(projeto_sintetico):
    raiz, vocab = projeto_sintetico
    cfg = config_sintetica(raiz, vocab, max_steps=8)

    resumo = treinar(cfg, log_every=2)

    assert resumo["passos"] == 8
    assert resumo["tokens"] == 8 * 4 * 32
    assert np.isfinite(resumo["loss_val"])

    checkpoint = Path(resumo["checkpoint"])
    assert checkpoint.exists()
    estado = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert estado["passo"] == 8
    assert "modelo" in estado and "otimizador" in estado

    with cfg.abspath(cfg.logging.csv_path).open(encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    eventos = {linha["evento"] for linha in linhas}
    assert {"passo", "avaliacao", "fim"} <= eventos

    amostras = cfg.abspath(cfg.logging.csv_path).with_name("teste_amostras.jsonl")
    assert amostras.exists()
    assert json.loads(amostras.read_text(encoding="utf-8").splitlines()[0])["prompt"]


def test_retomada_reproduz_o_treino_continuo(projeto_sintetico, tmp_path, monkeypatch):
    """Treinar 8 passos direto == treinar 4, cair, retomar e chegar aos 8.

    A comparação é nos pesos, com igualdade exata, e não na perda: qualquer
    diferença na ordem dos lotes ou no estado do dropout apareceria neles.
    """
    raiz, vocab = projeto_sintetico

    cfg_ref = config_sintetica(_preparar_projeto(tmp_path / "continuo", vocab), vocab, max_steps=8)
    treinar(cfg_ref, log_every=4)

    cfg_int = config_sintetica(_preparar_projeto(tmp_path / "interrompido", vocab), vocab, max_steps=8)

    import src.train as modulo_train

    salvar_original = modulo_train.salvar_checkpoint
    estado = {"ja_caiu": False}

    def salvar_e_cair(*args, **kwargs):
        salvar_original(*args, **kwargs)
        # Dispara uma única vez: o próprio tratador de KeyboardInterrupt chama
        # salvar_checkpoint de novo, e essa segunda chamada precisa terminar.
        if args[4] == 4 and not estado["ja_caiu"]:
            estado["ja_caiu"] = True
            raise KeyboardInterrupt

    monkeypatch.setattr(modulo_train, "salvar_checkpoint", salvar_e_cair)
    parcial = treinar(cfg_int, log_every=4)
    assert parcial["passos"] == 4

    # Retomada: mesmo diretório, sem o gancho de queda.
    monkeypatch.setattr(modulo_train, "salvar_checkpoint", salvar_original)
    final = treinar(cfg_int, log_every=4)
    assert final["passos"] == 8

    do_continuo = torch.load(cfg_ref.abspath("checkpoints/teste_ultimo.pt"), weights_only=False)
    do_retomado = torch.load(cfg_int.abspath("checkpoints/teste_ultimo.pt"), weights_only=False)

    assert do_continuo["passo"] == do_retomado["passo"] == 8
    for chave, tensor in do_continuo["modelo"].items():
        assert torch.equal(tensor, do_retomado["modelo"][chave]), chave
