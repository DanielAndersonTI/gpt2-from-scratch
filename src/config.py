"""Carga e validação das configurações de experimento.

Regra do projeto: um experimento é um arquivo YAML. Não há valor padrão
escondido no código: se um campo faltar ou sobrar, a carga falha. Assim os
quatro treinos diferem apenas no que está escrito no YAML, e a comparação
continua válida, porque nada muda por acidente.

Os caminhos dentro do YAML são relativos a `run.work_dir`. No Colab, work_dir
aponta para a pasta do Google Drive; na máquina local, para a pasta do projeto.
Basta trocar um campo, ou a variável de ambiente PROJECT_WORK_DIR, para rodar
o mesmo experimento nos dois lugares.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


class ConfigError(ValueError):
    """Erro de configuração, com o caminho do campo problemático na mensagem."""


# ---------------------------------------------------------------------------
# Blocos de configuração
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Identidade do experimento e onde os artefatos ficam."""

    name: str
    work_dir: str
    seed: int = 123
    device: str = "auto"  # auto | cpu | cuda
    deterministic: bool = True


@dataclass(frozen=True)
class DataConfig:
    """De onde vêm os tokens e como as janelas são cortadas."""

    corpus_dir: str
    tokenizer_path: str
    preprocessed_dir: str
    context_length: int = 256
    stride: int = 256  # igual ao contexto = janelas sem sobreposição
    batch_size: int = 128
    num_workers: int = 2
    drop_last: bool = True


@dataclass(frozen=True)
class MoEConfig:
    """Parâmetros da camada de especialistas. Só existe quando ffn_type = moe."""

    n_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 512
    aux_loss_coef: float = 0.01


@dataclass(frozen=True)
class ModelConfig:
    """Arquitetura. Os dois campos que mudam entre experimentos são ffn_type
    (dense ou moe) e pos_emb (learned ou rope)."""

    vocab_size: int = 16000
    context_length: int = 256
    n_layers: int = 4
    n_heads: int = 4
    emb_dim: int = 256
    drop_rate: float = 0.1
    qkv_bias: bool = False
    ffn_type: str = "dense"  # dense | moe
    pos_emb: str = "learned"  # learned | rope
    weight_tying: bool = False
    moe: MoEConfig | None = None
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class TrainConfig:
    """Otimização. O orçamento é fixado em passos, não em épocas, para que os
    quatro treinos consumam exatamente o mesmo número de tokens.

    max_steps = 837 vem da conta: 107.232 janelas de 256 tokens, lote de 128,
    descartando o último lote incompleto. 837 x 128 x 256 = 27.426.816 tokens.
    O notebook do professor imprime 1.675 lotes porque a saída dele foi gerada
    com lote 64; o orçamento que vale é o de tokens.
    """

    max_steps: int = 837
    lr: float = 4e-4
    min_lr_ratio: float = 0.1  # o cosine termina em 10% do pico
    warmup_frac: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every_frac: float = 0.1  # avaliação a cada 10% do treino
    eval_max_batches: int = 0  # 0 = validação inteira; >0 limita para smoke em CPU
    amp: bool = True
    amp_dtype: str = "float16"  # a T4 não tem bf16 nativo
    sdpa: bool = True
    fused_adamw: bool = True
    compile: bool = False
    checkpoint_every: int = 500  # 0 desliga o checkpoint
    checkpoint_dir: str = "checkpoints"
    max_hours: float | None = None  # parada antecipada antes do limite da sessão


@dataclass(frozen=True)
class LogConfig:
    """Onde o CSV é gravado e quais prompts são amostrados durante o treino."""

    csv_path: str
    prompts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    """Configuração completa de um experimento."""

    run: RunConfig
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    logging: LogConfig

    # -- propriedades derivadas ------------------------------------------------

    @property
    def tokens_per_step(self) -> int:
        return self.data.batch_size * self.data.context_length

    @property
    def total_tokens(self) -> int:
        return self.training.max_steps * self.tokens_per_step

    @property
    def warmup_steps(self) -> int:
        return max(1, round(self.training.warmup_frac * self.training.max_steps))

    @property
    def eval_every_steps(self) -> int:
        return max(1, round(self.training.eval_every_frac * self.training.max_steps))

    def abspath(self, relative: str) -> Path:
        """Converte um caminho relativo do YAML em caminho absoluto no disco."""
        return Path(self.run.work_dir) / relative

    def to_dict(self) -> dict[str, Any]:
        """Versão serializável, usada no log e no checkpoint."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Construção a partir do dicionário do YAML
# ---------------------------------------------------------------------------


def _required_fields(cls: type) -> set[str]:
    """Campos sem valor padrão: precisam estar no YAML."""
    return {
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }


def _build(cls: type, dados: Any, onde: str) -> Any:
    """Instancia uma dataclass conferindo campos desconhecidos e ausentes."""
    if not isinstance(dados, dict):
        raise ConfigError(f"{onde}: esperado um mapa de campos, recebido {type(dados).__name__}")

    campos = {f.name for f in dataclasses.fields(cls)}
    extras = set(dados) - campos
    if extras:
        raise ConfigError(f"{onde}: campos desconhecidos: {sorted(extras)}")

    faltando = _required_fields(cls) - set(dados)
    if faltando:
        raise ConfigError(f"{onde}: campos obrigatórios ausentes: {sorted(faltando)}")

    return cls(**dados)


def _parece_absoluto(caminho: str) -> bool:
    """Reconhece caminho absoluto tanto no padrão POSIX quanto no Windows.

    O YAML é escrito no Colab (`/content/...`) e lido também na máquina local,
    então a checagem não pode depender do sistema em que o teste roda. No
    Windows, `Path('/content').is_absolute()` devolve falso, porque falta a
    letra da unidade; por isso as duas convenções são testadas.
    """
    if PurePosixPath(caminho).is_absolute():
        return True
    return bool(PureWindowsPath(caminho).drive) or caminho.startswith("\\")


def _validate_conjunto(cfg: Config) -> None:
    """Regras que envolvem mais de um bloco ou que quebram o treino."""
    m, t, d = cfg.model, cfg.training, cfg.data

    if cfg.run.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError(f"run.device: {cfg.run.device!r} não é auto, cpu ou cuda")

    # A atenção divide a dimensão do embedding entre as cabeças.
    if m.emb_dim % m.n_heads != 0:
        raise ConfigError(
            f"model.emb_dim ({m.emb_dim}) precisa ser divisível por model.n_heads ({m.n_heads})"
        )

    if not 0.0 <= m.drop_rate < 1.0:
        raise ConfigError(f"model.drop_rate: {m.drop_rate} fora de [0, 1)")

    if m.ffn_type not in {"dense", "moe"}:
        raise ConfigError(f"model.ffn_type: {m.ffn_type!r} não é dense nem moe")
    if m.ffn_type == "moe":
        if m.moe is None:
            raise ConfigError("model.ffn_type = moe exige o bloco model.moe")
        if not 1 <= m.moe.top_k <= m.moe.n_experts:
            raise ConfigError(
                f"model.moe.top_k ({m.moe.top_k}) precisa estar entre 1 e n_experts ({m.moe.n_experts})"
            )
        if m.moe.hidden_dim <= 0:
            raise ConfigError("model.moe.hidden_dim precisa ser positivo")
        if m.moe.aux_loss_coef < 0:
            raise ConfigError("model.moe.aux_loss_coef não pode ser negativo")
    elif m.moe is not None:
        # Evita a ilusão de que o bloco MoE está em uso quando não está.
        raise ConfigError("model.moe foi preenchido, mas ffn_type = dense")

    if m.pos_emb not in {"learned", "rope"}:
        raise ConfigError(f"model.pos_emb: {m.pos_emb!r} não é learned nem rope")
    if m.rope_theta <= 0:
        raise ConfigError("model.rope_theta precisa ser positivo")

    # O modelo precisa saber quantas posições existem; o dado precisa cortar
    # janelas do mesmo tamanho. As duas coisas têm de coincidir.
    if m.context_length != d.context_length:
        raise ConfigError(
            f"model.context_length ({m.context_length}) difere de data.context_length "
            f"({d.context_length})"
        )
    if m.vocab_size <= 0:
        raise ConfigError("model.vocab_size precisa ser positivo")

    # Janela com passo maior que o contexto deixaria buracos no texto.
    if not 0 < d.stride <= d.context_length:
        raise ConfigError("data.stride precisa estar entre 1 e data.context_length")
    if d.batch_size <= 0 or d.context_length <= 0:
        raise ConfigError("data.batch_size e data.context_length precisam ser positivos")
    if d.num_workers < 0:
        raise ConfigError("data.num_workers não pode ser negativo")

    if t.max_steps <= 0:
        raise ConfigError("training.max_steps precisa ser positivo")
    if t.lr <= 0:
        raise ConfigError("training.lr precisa ser positivo")
    if not 0.0 <= t.min_lr_ratio <= 1.0:
        raise ConfigError("training.min_lr_ratio precisa estar em [0, 1]")
    if not 0.0 <= t.warmup_frac < 1.0:
        raise ConfigError("training.warmup_frac precisa estar em [0, 1)")
    if not 0.0 < t.eval_every_frac <= 1.0:
        raise ConfigError("training.eval_every_frac precisa estar em (0, 1]")
    if t.eval_max_batches < 0:
        raise ConfigError("training.eval_max_batches não pode ser negativo")
    if t.weight_decay < 0 or t.grad_clip <= 0:
        raise ConfigError("training.weight_decay >= 0 e grad_clip > 0")
    if t.amp_dtype not in {"float16", "bfloat16"}:
        raise ConfigError("training.amp_dtype: use float16 ou bfloat16")
    if t.checkpoint_every < 0:
        raise ConfigError("training.checkpoint_every não pode ser negativo")
    if t.max_hours is not None and t.max_hours <= 0:
        raise ConfigError("training.max_hours precisa ser positivo ou nulo")

    # O treino usa todas as janelas; sobreposição quebraria o orçamento.
    if d.stride != d.context_length:
        raise ConfigError(
            "data.stride precisa ser igual a data.context_length: janelas sobrepostas "
            "fariam o modelo ver mais tokens do que o orçamento de 27,4M"
        )

    if not cfg.logging.prompts or not all(isinstance(p, str) for p in cfg.logging.prompts):
        raise ConfigError("logging.prompts precisa ser uma lista não vazia de textos")

    # Caminhos internos são relativos a work_dir, para o mesmo YAML rodar no
    # Colab e na máquina local.
    for rotulo, caminho in [
        ("data.corpus_dir", d.corpus_dir),
        ("data.tokenizer_path", d.tokenizer_path),
        ("data.preprocessed_dir", d.preprocessed_dir),
        ("training.checkpoint_dir", t.checkpoint_dir),
        ("logging.csv_path", cfg.logging.csv_path),
    ]:
        if _parece_absoluto(caminho):
            raise ConfigError(f"{rotulo}: use caminho relativo a run.work_dir, não {caminho!r}")


def config_from_dict(bruto: dict[str, Any], work_dir: str | None = None) -> Config:
    """Monta a configuração a partir do dicionário do YAML ou do checkpoint.

    O checkpoint grava `cfg.to_dict()`, então esta função é o caminho de volta:
    o avaliador reconstrói o modelo exatamente como ele foi treinado, sem
    precisar do YAML original.
    """
    if not isinstance(bruto, dict):
        raise ConfigError("a configuração precisa ser um mapa de blocos")

    blocos = {"run", "data", "model", "training", "logging"}
    extras = set(bruto) - blocos
    if extras:
        raise ConfigError(f"blocos desconhecidos: {sorted(extras)}")
    faltando = blocos - set(bruto)
    if faltando:
        raise ConfigError(f"blocos obrigatórios ausentes: {sorted(faltando)}")

    run_dados = dict(bruto["run"])
    run_dados["work_dir"] = work_dir or os.environ.get("PROJECT_WORK_DIR") or run_dados.get("work_dir")
    if not run_dados["work_dir"]:
        raise ConfigError("run.work_dir não definido no YAML nem em PROJECT_WORK_DIR")

    # O bloco `moe` é aninhado: sai do mapa do modelo e é construído à parte,
    # para que `ModelConfig` receba sempre um `MoEConfig` já validado.
    dados_moe = bruto["model"].get("moe")
    modelo_dados = dict(bruto["model"])
    modelo_dados.pop("moe", None)
    if dados_moe is not None and not isinstance(dados_moe, dict):
        # No checkpoint, `asdict` já transformou o MoEConfig em dicionário.
        dados_moe = dataclasses.asdict(dados_moe)

    cfg = Config(
        run=_build(RunConfig, run_dados, "run"),
        data=_build(DataConfig, bruto["data"], "data"),
        model=_build(ModelConfig, modelo_dados, "model"),
        training=_build(TrainConfig, bruto["training"], "training"),
        logging=_build(LogConfig, bruto["logging"], "logging"),
    )

    if dados_moe is not None:
        cfg = dataclasses.replace(
            cfg,
            model=dataclasses.replace(
                cfg.model, moe=_build(MoEConfig, dados_moe, "model.moe")
            ),
        )

    _validate_conjunto(cfg)
    return cfg


def load_config(caminho: str | Path, work_dir: str | None = None) -> Config:
    """Lê o YAML e delega a `config_from_dict`.

    Ordem de precedência para a pasta de trabalho:
    1. argumento `work_dir`;
    2. variável de ambiente PROJECT_WORK_DIR;
    3. campo run.work_dir do YAML.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise ConfigError(f"configuração não encontrada: {caminho}")

    with caminho.open(encoding="utf-8") as f:
        bruto = yaml.safe_load(f)

    return config_from_dict(bruto, work_dir=work_dir)
