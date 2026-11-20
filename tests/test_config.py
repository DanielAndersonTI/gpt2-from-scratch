"""Testes do schema de configuração.

O que está sendo garantido: o loader aceita os arquivos reais, rejeita erro de
digitação em nome de campo, rejeita combinação que quebraria o treino e
resolve a pasta de trabalho na ordem documentada.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = RAIZ / "configs"


def carregar_bruto(nome: str = "e1_vanilla.yaml") -> dict:
    """Lê um YAML real como dicionário, para os testes o modificarem."""
    with (CONFIGS / nome).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def escrever(tmp_path: Path, bruto: dict) -> Path:
    """Grava o dicionário modificado e devolve o caminho."""
    caminho = tmp_path / "config.yaml"
    caminho.write_text(yaml.safe_dump(bruto, allow_unicode=True), encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# Caminho feliz
# ---------------------------------------------------------------------------


def test_e1_carrega_com_o_orcamento_do_enunciado():
    cfg = load_config(CONFIGS / "e1_vanilla.yaml")
    assert cfg.run.name == "e1_vanilla"
    assert cfg.model.ffn_type == "dense"
    assert cfg.model.pos_emb == "learned"
    assert cfg.model.moe is None
    # 128 x 256 = 32.768 tokens por passo; 837 passos = 27.426.816 tokens.
    assert cfg.tokens_per_step == 32_768
    assert cfg.total_tokens == 27_426_816


def test_agenda_derivada_do_orcamento():
    cfg = load_config(CONFIGS / "e1_vanilla.yaml")
    assert cfg.warmup_steps == 42  # 5% de 837
    assert cfg.eval_every_steps == 84  # 10% de 837


@pytest.mark.parametrize(
    "arquivo, ffn, pos, tem_moe",
    [
        ("e1_vanilla.yaml", "dense", "learned", False),
        ("e2_moe_h512.yaml", "moe", "learned", True),
        ("e3_moe_h128.yaml", "moe", "learned", True),
        ("e4_rope.yaml", "dense", "rope", False),
        ("benchmark.yaml", "dense", "learned", False),
    ],
)
def test_todos_os_yaml_do_projeto_carregam(arquivo, ffn, pos, tem_moe):
    cfg = load_config(CONFIGS / arquivo)
    assert cfg.model.ffn_type == ffn
    assert cfg.model.pos_emb == pos
    assert (cfg.model.moe is not None) is tem_moe


def test_moe_carrega_o_bloco_aninhado():
    cfg = load_config(CONFIGS / "e2_moe_h512.yaml")
    assert cfg.model.moe is not None
    assert cfg.model.moe.n_experts == 8
    assert cfg.model.moe.top_k == 2
    assert cfg.model.moe.hidden_dim == 512
    assert cfg.model.moe.aux_loss_coef == 0.01


def test_to_dict_serializa_o_bloco_aninhado():
    cfg = load_config(CONFIGS / "e2_moe_h512.yaml")
    d = cfg.to_dict()
    assert d["model"]["moe"]["n_experts"] == 8
    assert d["run"]["name"] == "e2_moe_h512"


# ---------------------------------------------------------------------------
# Erros de escrita do YAML
# ---------------------------------------------------------------------------


def test_campo_desconhecido_falha(tmp_path):
    bruto = carregar_bruto()
    bruto["model"]["n_camadas"] = 4  # grafia errada de n_layers
    with pytest.raises(ConfigError, match="desconhecidos"):
        load_config(escrever(tmp_path, bruto))


def test_bloco_obrigatorio_ausente_falha(tmp_path):
    bruto = carregar_bruto()
    del bruto["logging"]
    with pytest.raises(ConfigError, match="logging"):
        load_config(escrever(tmp_path, bruto))


def test_campo_obrigatorio_ausente_falha(tmp_path):
    bruto = carregar_bruto()
    del bruto["data"]["corpus_dir"]
    with pytest.raises(ConfigError, match="corpus_dir"):
        load_config(escrever(tmp_path, bruto))


def test_arquivo_inexistente_falha(tmp_path):
    with pytest.raises(ConfigError, match="não encontrada"):
        load_config(tmp_path / "nao_existe.yaml")


# ---------------------------------------------------------------------------
# Regras que quebrariam o treino
# ---------------------------------------------------------------------------


def test_emb_dim_indivisivel_pelas_cabecas_falha(tmp_path):
    bruto = carregar_bruto()
    bruto["model"]["emb_dim"] = 250
    with pytest.raises(ConfigError, match="divisível"):
        load_config(escrever(tmp_path, bruto))


def test_moe_sem_bloco_falha(tmp_path):
    bruto = carregar_bruto()
    bruto["model"]["ffn_type"] = "moe"
    with pytest.raises(ConfigError, match="exige o bloco"):
        load_config(escrever(tmp_path, bruto))


def test_bloco_moe_sem_uso_falha(tmp_path):
    bruto = carregar_bruto()
    bruto["model"]["moe"] = {"n_experts": 8}  # ffn_type continua dense
    with pytest.raises(ConfigError, match="ffn_type = dense"):
        load_config(escrever(tmp_path, bruto))


def test_top_k_maior_que_especialistas_falha(tmp_path):
    bruto = carregar_bruto("e2_moe_h512.yaml")
    bruto["model"]["moe"]["top_k"] = 9
    with pytest.raises(ConfigError, match="top_k"):
        load_config(escrever(tmp_path, bruto))


def test_stride_diferente_do_contexto_falha(tmp_path):
    bruto = carregar_bruto()
    bruto["data"]["stride"] = 128
    with pytest.raises(ConfigError, match="stride"):
        load_config(escrever(tmp_path, bruto))


def test_caminho_absoluto_no_corpo_falha(tmp_path):
    bruto = carregar_bruto()
    bruto["data"]["corpus_dir"] = "/content/drive/MyDrive/LLM_Final/corpus"
    with pytest.raises(ConfigError, match="relativo"):
        load_config(escrever(tmp_path, bruto))


def test_prompts_vazios_falham(tmp_path):
    bruto = carregar_bruto()
    bruto["logging"]["prompts"] = []
    with pytest.raises(ConfigError, match="prompts"):
        load_config(escrever(tmp_path, bruto))


# ---------------------------------------------------------------------------
# Pasta de trabalho
# ---------------------------------------------------------------------------


def test_work_dir_do_argumento_vence_o_yaml():
    cfg = load_config(CONFIGS / "e1_vanilla.yaml", work_dir="/tmp/projeto")
    assert cfg.run.work_dir == "/tmp/projeto"


def test_variavel_de_ambiente_vence_o_yaml(monkeypatch):
    monkeypatch.setenv("PROJECT_WORK_DIR", "/tmp/do_ambiente")
    cfg = load_config(CONFIGS / "e1_vanilla.yaml")
    assert cfg.run.work_dir == "/tmp/do_ambiente"


def test_argumento_vence_a_variavel_de_ambiente(monkeypatch):
    monkeypatch.setenv("PROJECT_WORK_DIR", "/tmp/do_ambiente")
    cfg = load_config(CONFIGS / "e1_vanilla.yaml", work_dir="/tmp/do_argumento")
    assert cfg.run.work_dir == "/tmp/do_argumento"


def test_abspath_junta_work_dir_e_caminho_relativo():
    cfg = load_config(CONFIGS / "e1_vanilla.yaml", work_dir="/tmp/projeto")
    assert cfg.abspath("logs/e1_vanilla.csv") == Path("/tmp/projeto/logs/e1_vanilla.csv")
