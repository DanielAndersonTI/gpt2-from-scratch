"""Testes do RoPE.

O que está sendo garantido: a rotação preserva a norma, o produto interno entre
uma consulta e uma chave depende só da distância entre as posições, posições
diferentes produzem vetores diferentes, e o modelo com RoPE continua causal e
funciona além do comprimento de treino.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from apoio import config_sintetica
from src.config import load_config
from src.model import GPTModel
from src.rope import RoPE
from src.seeds import set_seed
from src.train import treinar


@pytest.fixture
def rope() -> RoPE:
    return RoPE(head_dim=16, context_length=32, theta=10_000.0)


def _qk(pos_m: int, pos_n: int, d: int = 16, n: int = 32):
    """Coloca o mesmo vetor u na posição m de q e v na posição n de k."""
    torch.manual_seed(7)
    u, v = torch.randn(d), torch.randn(d)
    q = torch.zeros(1, 1, n, d)
    k = torch.zeros(1, 1, n, d)
    q[0, 0, pos_m] = u
    k[0, 0, pos_n] = v
    return q, k


def test_forma_preservada(rope):
    q, k = torch.randn(2, 3, 8, 16), torch.randn(2, 3, 8, 16)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape


def test_norma_preservada(rope):
    """Rotação é ortogonal: o comprimento do vetor não muda."""
    q = torch.randn(2, 3, 8, 16)
    qr, _ = rope(q, q.clone())
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_produto_interno_depende_so_da_distancia(rope):
    """Deslocar consulta e chave juntas não muda o produto interno."""
    q, k = _qk(3, 7)
    q2, k2 = _qk(3 + 5, 7 + 5)  # mesma semente: o mesmo par de vetores
    qr, kr = rope(q, k)
    qr2, kr2 = rope(q2, k2)

    a = torch.dot(qr[0, 0, 3], kr[0, 0, 7])
    b = torch.dot(qr2[0, 0, 8], kr2[0, 0, 12])
    assert a.item() == pytest.approx(b.item(), abs=1e-5)


def test_produto_interno_muda_com_a_distancia(rope):
    q, k = _qk(3, 7)
    qr, kr = rope(q, k)
    perto = torch.dot(qr[0, 0, 3], kr[0, 0, 7]).item()

    q, k = _qk(3, 20)
    qr, kr = rope(q, k)
    longe = torch.dot(qr[0, 0, 3], kr[0, 0, 20]).item()

    assert perto != pytest.approx(longe, abs=1e-3)


def test_posicao_zero_nao_gira(rope):
    """Na posição 0 o ângulo é zero, então o vetor sai igual."""
    q = torch.randn(1, 2, 16)
    qr, _ = rope(q.unsqueeze(2), q.unsqueeze(2))
    assert torch.allclose(qr[:, :, 0], q, atol=1e-6)


def test_funciona_alem_do_comprimento_de_treino(rope):
    """Sem tabela congelada no construtor, 256 tokens não estouram o índice."""
    q = torch.randn(1, 1, 256, 16)
    qr, _ = rope(q, q.clone())
    assert qr.shape == q.shape


def test_prefixo_identico_ao_estender_a_sequencia(rope):
    """Os primeiros tokens giram igual, com ou sem tokens a mais depois."""
    q = torch.randn(1, 1, 64, 16)
    curto, _ = rope(q[:, :, :32], q[:, :, :32].clone())
    longo, _ = rope(q, q.clone())
    assert torch.allclose(curto, longo[:, :, :32], atol=1e-6)


# -- integração com o modelo -------------------------------------------------


def test_modelo_com_rope_nao_tem_embedding_posicional():
    cfg = load_config("configs/e4_rope.yaml", work_dir=".")
    model = GPTModel(cfg.model)
    assert model.pos_emb is None
    # São exatamente os 65.536 do embedding posicional, 256 x 256, a menos.
    assert model.contar_parametros() == 11_414_016 - 256 * 256


def test_modelo_com_rope_continua_causal():
    """Mudar um token do futuro não pode alterar a previsão do passado."""
    cfg = load_config("configs/e4_rope.yaml", work_dir=".")
    set_seed(0)
    model = GPTModel(cfg.model)
    model.eval()

    idx = torch.randint(0, cfg.model.vocab_size, (1, 12))
    outro = idx.clone()
    outro[0, 8] = (outro[0, 8] + 1) % cfg.model.vocab_size

    with torch.no_grad():
        a = model(idx)
        b = model(outro)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)


def test_modelo_com_rope_produz_logits_finitos():
    cfg = load_config("configs/e4_rope.yaml", work_dir=".")
    set_seed(0)
    model = GPTModel(cfg.model)
    modelo_novo = model(torch.randint(0, cfg.model.vocab_size, (1, 16)))
    assert torch.isfinite(modelo_novo).all()


# -- extrapolação ------------------------------------------------------------


@pytest.mark.parametrize("sdpa", [True, False])
def test_modelo_roda_alem_do_contexto_de_treino(sdpa):
    """Janela de 64 tokens num modelo configurado para 32.

    Vale para os dois caminhos de atenção. O do SDPA recebe a causalidade pelo
    kernel; o manual usa a máscara gravada, que tem 32 linhas e é pequena
    demais para a janela, então precisa ser refeita na hora.
    """
    cfg = load_config("configs/e4_rope.yaml", work_dir=".")
    pequeno = replace(cfg.model, context_length=32, n_layers=1, emb_dim=64, n_heads=2)
    set_seed(0)
    model = GPTModel(pequeno, sdpa=sdpa)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randint(0, pequeno.vocab_size, (1, 64)))
    assert logits.shape == (1, 64, pequeno.vocab_size)
    assert torch.isfinite(logits).all()


def test_extrapolacao_mede_o_rope_e_recusa_o_modelo_com_posicao_aprendida(projeto_sintetico):
    """O script de extrapolação tem de medir um e barrar o outro.

    O modelo com posição aprendida só tem linha de embedding para 32 posições;
    a linha de contexto 64 vira erro anotado, não queda do script. É a
    limitação que a comparação quer mostrar.
    """
    from src.extrapolacao import comparar

    raiz, vocab = projeto_sintetico
    base = config_sintetica(raiz, vocab, max_steps=2, checkpoint_every=2)

    cfg_rope = replace(
        base,
        model=replace(base.model, pos_emb="rope"),
        run=replace(base.run, name="teste_extrapolacao"),
        logging=replace(base.logging, csv_path="logs/teste_extrapolacao.csv"),
    )
    resumo = treinar(cfg_rope, log_every=2)
    linhas = {
        linha["contexto"]: linha
        for linha in comparar(resumo["checkpoint"], [32, 64], device="cpu")["contextos"]
    }
    assert linhas[32]["tokens_avaliados"] > 0
    assert linhas[64]["tokens_avaliados"] > 0
    assert 0.0 < linhas[64]["bpb"] < 20.0

    cfg_aprendida = replace(
        cfg_rope,
        model=replace(base.model, pos_emb="learned"),
        run=replace(base.run, name="teste_sem_extrapolacao"),
        logging=replace(base.logging, csv_path="logs/teste_sem_extrapolacao.csv"),
    )
    resumo_aprendida = treinar(cfg_aprendida, log_every=2)
    comparacao = comparar(resumo_aprendida["checkpoint"], [64], device="cpu")
    assert "RoPE" in comparacao["contextos"][0]["erro"]
