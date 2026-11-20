"""Testes do modelo.

O que está sendo garantido: a contagem de parâmetros é exatamente a do
professor, a máscara impede olhar para o futuro, o gradiente chega a todos os
pesos, o caminho com SDPA dá o mesmo resultado da conta manual e a loss inicial
está perto de ln(16.000).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.config import ModelConfig
from src.model import (
    GPTModel,
    calc_loss_batch,
    descrever_parametros,
    gerar_texto_simples,
)


def config_base(**mudancas) -> ModelConfig:
    """Configuração base com contexto curto, para os testes rodarem rápido."""
    padrao = dict(
        vocab_size=16000,
        context_length=64,
        n_layers=4,
        n_heads=4,
        emb_dim=256,
        drop_rate=0.0,  # sem dropout: os testes comparam saídas exatas
        qkv_bias=False,
        ffn_type="dense",
        pos_emb="learned",
        weight_tying=False,
    )
    padrao.update(mudancas)
    return ModelConfig(**padrao)


def test_contagem_de_parametros_da_configuracao_base():
    # O número do professor pressupõe contexto de 256, que são 65.536
    # parâmetros de posição. Com contexto 64 a conta seria outra.
    modelo = GPTModel(config_base(context_length=256))
    assert modelo.contar_parametros() == 11_414_016


def test_contagem_com_pesos_compartilhados():
    modelo = GPTModel(config_base(context_length=256, weight_tying=True))
    assert modelo.contar_parametros() == 7_318_016
    assert modelo.out_head.weight is modelo.tok_emb.weight


def test_detalhe_dos_parametros_fecha_com_o_total():
    modelo = GPTModel(config_base(context_length=256))
    detalhe = descrever_parametros(modelo)
    soma = (
        detalhe["tok_emb"]
        + detalhe["pos_emb"]
        + detalhe["out_head"]
        + detalhe["final_norm"]
        + detalhe["blocos"]
    )
    assert soma == detalhe["total"] == 11_414_016


def test_formato_da_saida():
    modelo = GPTModel(config_base())
    modelo.eval()
    x = torch.randint(0, 16000, (2, 16))
    with torch.no_grad():
        logits = modelo(x)
    assert logits.shape == (2, 16, 16000)


def test_mascara_causal_impede_ver_o_futuro():
    """Mudar o último token não pode alterar a predição das posições anteriores."""
    torch.manual_seed(0)
    modelo = GPTModel(config_base())
    modelo.eval()

    idx = torch.randint(0, 16000, (1, 8))
    outro = idx.clone()
    outro[0, -1] = (outro[0, -1] + 1) % 16000

    with torch.no_grad():
        a = modelo(idx)
        b = modelo(outro)

    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-6)


def test_gradiente_chega_a_todos_os_parametros():
    modelo = GPTModel(config_base())
    x = torch.randint(0, 16000, (2, 16))
    logits = modelo(x)
    perda = torch.nn.functional.cross_entropy(logits.flatten(0, 1), x.flatten())
    perda.backward()

    sem_gradiente = [nome for nome, p in modelo.named_parameters() if p.grad is None]
    assert sem_gradiente == []
    assert modelo.tok_emb.weight.grad.abs().sum() > 0


def test_loss_inicial_perto_de_ln_do_vocabulario():
    """Modelo aleatório deve ficar perto da distribuição uniforme sobre 16.000."""
    torch.manual_seed(0)
    modelo = GPTModel(config_base())
    modelo.eval()
    x = torch.randint(0, 16000, (4, 32))
    y = torch.randint(0, 16000, (4, 32))
    with torch.no_grad():
        perda = calc_loss_batch(x, y, modelo, torch.device("cpu")).item()
    assert abs(perda - math.log(16000)) < 0.3


def test_sdpa_e_conta_manual_dao_o_mesmo_resultado():
    torch.manual_seed(0)
    com_sdpa = GPTModel(config_base(), sdpa=True)
    com_sdpa.eval()
    manual = GPTModel(config_base(), sdpa=False)
    manual.load_state_dict(com_sdpa.state_dict())
    manual.eval()

    x = torch.randint(0, 16000, (2, 16))
    with torch.no_grad():
        a = com_sdpa(x)
        b = manual(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_geracao_acrescenta_tokens():
    modelo = GPTModel(config_base())
    modelo.eval()
    inicio = torch.randint(0, 16000, (1, 4))
    saida = gerar_texto_simples(modelo, inicio, max_new_tokens=5, context_size=64)
    assert saida.shape == (1, 9)
    assert torch.equal(saida[:, :4], inicio)


def test_geracao_com_top_k_nao_estoura_o_contexto():
    modelo = GPTModel(config_base())
    modelo.eval()
    inicio = torch.randint(0, 16000, (1, 10))
    saida = gerar_texto_simples(modelo, inicio, max_new_tokens=6, context_size=8, top_k=10)
    assert saida.shape == (1, 16)


def test_contexto_menor_que_a_posicao_aprendida_falha():
    """Passar mais tokens que o contexto configurado deve estourar, não truncar."""
    modelo = GPTModel(config_base(context_length=8))
    modelo.eval()
    with pytest.raises(IndexError):
        with torch.no_grad():
            modelo(torch.randint(0, 16000, (1, 16)))
