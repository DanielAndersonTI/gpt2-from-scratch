"""Apoio aos testes de ponta a ponta: corpus sintético e configuração pequena.

O corpus real tem 137 MB e o modelo real tem 11 milhões de parâmetros; nenhum
dos dois cabe em um teste que roda em segundos. Aqui o corpus é gerado, o
tokenizador BPE é treinado na hora e o modelo tem uma camada. O que está sob
teste é o encadeamento das peças, não a qualidade do resultado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from src.config import Config, DataConfig, LogConfig, ModelConfig, RunConfig, TrainConfig
from src.data import SEP, TokenizadorUFCG


def escrever_corpus_sintetico(raiz: Path, n_docs: int = 200) -> int:
    """Cria txt, parquet e tokenizador em `raiz`. Devolve o tamanho do vocabulário."""
    rng = np.random.default_rng(0)
    palavras = [f"palavra{i}" for i in range(80)]
    documentos = [
        " ".join(rng.choice(palavras, size=int(rng.integers(8, 20)))) for _ in range(n_docs)
    ]

    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / "train.txt").write_text(SEP.join(documentos), encoding="utf-8")
    (raiz / "val.txt").write_text(SEP.join(documentos[:40]), encoding="utf-8")

    import pandas as pd

    tabela = pd.DataFrame(
        {"id": range(n_docs), "sigla_centro": ["CCT"] * n_docs, "texto": documentos}
    )
    tabela.to_parquet(raiz / "train.parquet")
    tabela.iloc[:40].to_parquet(raiz / "val.parquet")

    (raiz / "tokenizer_16k").mkdir(exist_ok=True)
    caminho_tok = raiz / "tokenizer_16k" / "tokenizer.json"
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    treinador = trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>"], show_progress=False
    )
    tok.train_from_iterator(documentos, treinador)
    tok.save(str(caminho_tok))

    return TokenizadorUFCG(caminho_tok).n_vocab


def config_sintetica(
    work_dir: Path,
    vocab: int,
    max_steps: int,
    checkpoint_every: int = 4,
) -> Config:
    """Configuração minúscula, em CPU, com o mesmo formato da real."""
    return Config(
        run=RunConfig(name="teste", work_dir=str(work_dir), seed=123, device="cpu", deterministic=True),
        data=DataConfig(
            corpus_dir="corpus",
            tokenizer_path="corpus/tokenizer_16k/tokenizer.json",
            preprocessed_dir="prep",
            context_length=32,
            stride=32,
            batch_size=4,
            num_workers=0,
            drop_last=True,
        ),
        model=ModelConfig(
            vocab_size=vocab,
            context_length=32,
            n_layers=1,
            n_heads=2,
            emb_dim=32,
            drop_rate=0.1,
            ffn_type="dense",
            pos_emb="learned",
        ),
        training=TrainConfig(
            max_steps=max_steps,
            lr=1e-3,
            warmup_frac=0.25,
            eval_every_frac=0.5,
            eval_max_batches=2,
            amp=False,
            sdpa=True,
            fused_adamw=False,
            checkpoint_every=checkpoint_every,
            checkpoint_dir="checkpoints",
        ),
        logging=LogConfig(csv_path="logs/teste.csv", prompts=["palavra1 palavra2"]),
    )
