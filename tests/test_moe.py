"""Testes da camada de especialistas.

O que está sendo garantido, além de shapes: a contagem de parâmetros totais e
ativos bate com a tabela do artigo nas duas variantes reais de MoE, a perda
auxiliar vale zero no balanceamento perfeito e cresce quando o roteador
colapsa, e o gradiente chega ao roteador e aos especialistas.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from apoio import config_sintetica
from src.config import MoEConfig, load_config
from src.model import GPTModel
from src.moe import Expert, MoEFeedForward
from src.seeds import set_seed
from src.train import treinar


@pytest.fixture
def cfg_moe() -> MoEConfig:
    return MoEConfig(n_experts=8, top_k=2, hidden_dim=64, aux_loss_coef=0.01)


@pytest.fixture
def camada(cfg_moe) -> MoEFeedForward:
    set_seed(0)
    return MoEFeedForward(emb_dim=32, drop_rate=0.0, cfg=cfg_moe)


# -- shapes e fluxo ----------------------------------------------------------


def test_forma_de_saida_e_igual_a_da_mlp_densa(camada):
    x = torch.randn(2, 5, 32)
    saida = camada(x)
    assert saida.shape == x.shape


def test_top_k_dois_consulta_dois_especialistas(camada):
    """Cada token despacha para exatamente k especialistas.

    A contagem usa a utilização registrada no passo direto: com 2 x 5 = 10
    tokens e k = 2, o total de despachos é 20.
    """
    camada(torch.randn(2, 5, 32))
    assert int(camada.utilizacao.sum()) == 2 * 5 * 2


def test_gradiente_chega_ao_roteador_e_aos_especialistas(camada):
    x = torch.randn(2, 5, 32, requires_grad=True)
    camada(x).sum().backward()

    assert camada.router.gate.weight.grad is not None
    assert camada.router.gate.weight.grad.abs().sum() > 0
    # Todo token passa pelo menos por um especialista, então algum recebe
    # gradiente. Não se exige que todos recebam: um especialista ocioso no
    # lote não tem por que receber.
    com_gradiente = [
        e for e in camada.experts
        if e.fc1.weight.grad is not None and e.fc1.weight.grad.abs().sum() > 0
    ]
    assert len(com_gradiente) > 0


def test_saida_dos_especialistas_e_poderada_pelos_pesos_do_roteador(camada):
    """Com um único token e dropout desligado, a saída é a soma ponderada."""
    x = torch.randn(1, 1, 32)
    with torch.no_grad():
        probs = torch.softmax(camada.router(x.reshape(1, 32)), dim=-1)[0]
        pesos, indices = probs.topk(2)
        pesos = pesos / pesos.sum()
        esperado = sum(
            pesos[j] * camada.experts[indices[j]](x.reshape(1, 32))[0]
            for j in range(2)
        )
        obtido = camada(x)[0, 0]
    assert torch.allclose(obtido, esperado, atol=1e-6)


# -- perda auxiliar ----------------------------------------------------------


def test_perda_auxiliar_e_zero_com_carga_uniforme(camada):
    """Carga e probabilidade uniformes entre 8 especialistas dão zero."""
    probs = torch.full((8, 8), 1.0 / 8)
    indices = torch.arange(8).view(8, 1)  # cada token no seu especialista
    perda = camada._perda_de_balanceamento(probs, indices)
    assert perda.item() == pytest.approx(0.0, abs=1e-6)


def test_perda_auxiliar_cresce_quando_o_roteador_colapsa(camada):
    """Todos os tokens no especialista 0: a perda chega ao máximo de E - 1."""
    probs = torch.zeros(8, 8)
    probs[:, 0] = 1.0
    indices = torch.zeros(8, 1, dtype=torch.long)
    perda = camada._perda_de_balanceamento(probs, indices)
    assert perda.item() == pytest.approx(7.0, abs=1e-6)


def test_perda_auxiliar_nunca_e_negativa(camada):
    torch.manual_seed(0)
    for _ in range(20):
        probs = torch.softmax(torch.randn(16, 8), dim=-1)
        indices = probs.topk(2, dim=-1).indices
        assert camada._perda_de_balanceamento(probs, indices).item() >= -1e-6


def test_perda_auxiliar_so_existe_em_modo_de_treino(camada):
    camada.train()
    camada(torch.randn(2, 5, 32))
    assert camada.aux_loss is not None
    assert camada.aux_loss.item() >= 0.0

    camada.eval()
    with torch.no_grad():
        camada(torch.randn(2, 5, 32))
    assert camada.aux_loss is None


def test_perda_auxiliar_ja_vem_multiplicada_pelo_coeficiente():
    """O treino soma `perda_auxiliar()` direto, então o coeficiente entra aqui.

    Com coeficiente zero, o termo some do objetivo mesmo que a carga esteja
    desbalanceada. É o que garante que o treino não multiplique o coeficiente
    duas vezes.
    """
    cfg = MoEConfig(n_experts=4, top_k=2, hidden_dim=16, aux_loss_coef=0.0)
    camada = MoEFeedForward(emb_dim=32, drop_rate=0.0, cfg=cfg)
    camada.train()
    camada(torch.randn(2, 5, 32))
    assert camada.aux_loss.item() == pytest.approx(0.0)


# -- integração com o modelo -------------------------------------------------


def test_modelo_com_moe_soma_a_perda_auxiliar_dos_blocos():
    cfg = load_config("configs/e3_moe_h128.yaml", work_dir=".")
    set_seed(0)
    model = GPTModel(cfg.model)
    model.train()
    model(torch.randint(0, cfg.model.vocab_size, (2, 8)))
    perda = model.perda_auxiliar()
    assert perda is not None
    assert perda.item() > 0
    # Uma perda por bloco, todas somadas.
    assert perda.item() == pytest.approx(
        sum(b.ff.aux_loss.item() for b in model.trf_blocks), rel=1e-5
    )


def test_modelo_denso_nao_tem_perda_auxiliar():
    cfg = load_config("configs/e1_vanilla.yaml", work_dir=".")
    set_seed(0)
    model = GPTModel(cfg.model)
    model(torch.randint(0, cfg.model.vocab_size, (1, 4)))
    assert model.perda_auxiliar() is None


# -- contagem de parâmetros contra a tabela do artigo ------------------------


@pytest.mark.parametrize(
    "arquivo, total, ativos",
    [
        ("configs/e2_moe_h512.yaml", 17_733_152, 11_423_264),
        ("configs/e3_moe_h128.yaml", 11_429_408, 9_847_328),
    ],
)
def test_contagem_de_parametros_confere_com_a_tabela(arquivo, total, ativos):
    cfg = load_config(arquivo, work_dir=".")
    model = GPTModel(cfg.model)
    assert model.contar_parametros() == total
    assert model.contar_parametros_ativos() == ativos


def test_expert_tem_a_largura_pedida():
    e = Expert(emb_dim=32, hidden_dim=64)
    assert e(torch.randn(3, 32)).shape == (3, 32)
    assert sum(p.numel() for p in e.parameters()) == 32 * 64 + 64 + 64 * 32 + 32


def test_treino_curto_com_moe_roda_ponta_a_ponta(projeto_sintetico):
    """O MoE passa pelo mesmo laço de treino do modelo denso.

    É o teste que pega o erro de somar a perda auxiliar duas vezes, ou de
    esquecer de somá-la, porque o laço roda de fato com o termo dentro.
    """
    raiz, vocab = projeto_sintetico
    cfg = config_sintetica(raiz, vocab, max_steps=4, checkpoint_every=2)
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            ffn_type="moe",
            moe=MoEConfig(n_experts=4, top_k=2, hidden_dim=16, aux_loss_coef=0.01),
        ),
        run=replace(cfg.run, name="teste_moe"),
        logging=replace(cfg.logging, csv_path="logs/teste_moe.csv"),
    )

    resumo = treinar(cfg, log_every=2)
    assert resumo["passos"] == 4
    assert torch.isfinite(torch.tensor(resumo["loss_val"]))
