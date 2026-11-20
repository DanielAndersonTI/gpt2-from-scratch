"""Testes do pipeline de dados.

O que está sendo garantido: as janelas são as mesmas do livro, o
`<|endoftext|>` fecha cada documento, os deslocamentos fecham a conta com o
total de tokens, o dataloader é reproduzível e a junção com os metadados
depende de uma conferência, não de uma suposição.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import (
    SEP,
    Corpus,
    GPTDatasetTokens,
    contar_palavras,
    criar_dataloader,
    ler_documentos,
    localizar_corpus,
    preparar,
    pre_tokenizar,
    verificar_metadados,
)


class TokenizadorFalso:
    """Tokenizador de mentira: cada palavra vale 3 tokens e o eot é 0."""

    def __init__(self, caminho=Path("tokenizer.json"), n_tokens_por_doc=3):
        self.caminho = Path(caminho)
        self.eot_id = 0
        self.n_vocab = 16000
        self.n_tokens_por_doc = n_tokens_por_doc

    def encode_batch(self, textos):
        return [[7] * self.n_tokens_por_doc for _ in textos]


def test_ler_documentos_ignora_vazios(tmp_path):
    caminho = tmp_path / "train.txt"
    caminho.write_text(f"primeiro{SEP}segundo{SEP}   {SEP}terceiro", encoding="utf-8")
    assert ler_documentos(caminho) == ["primeiro", "segundo", "terceiro"]


def test_pre_tokenizar_anexa_eot_e_fecha_os_offsets():
    tokenizer = TokenizadorFalso(n_tokens_por_doc=3)
    pretok = pre_tokenizar(["a", "b", "c"], tokenizer, tamanho_lote=2, progresso=False)

    assert pretok.n_documentos == 3
    assert pretok.n_tokens == 3 * (3 + 1)  # três tokens mais o eot de cada doc
    assert pretok.offsets.tolist() == [0, 4, 8, 12]
    assert pretok.ids.dtype == np.uint16
    # Cada documento termina com o id do eot.
    assert [pretok.ids[i] for i in (3, 7, 11)] == [0, 0, 0]


def test_documento_da_posicao_usa_os_offsets():
    pretok = pre_tokenizar(["a", "b", "c"], TokenizadorFalso(), progresso=False)
    # Documento 0 ocupa [0, 4), o 1 [4, 8), o 2 [8, 12).
    assert pretok.documento_da_posicao(0) == 0
    assert pretok.documento_da_posicao(3) == 0
    assert pretok.documento_da_posicao(4) == 1
    assert pretok.documento_da_posicao(11) == 2


def test_contar_palavras_segue_o_split_do_enunciado():
    assert contar_palavras(["uma duas", "três  quatro"]) == 4


# ---------------------------------------------------------------------------
# Janelas
# ---------------------------------------------------------------------------


class DatasetDoLivro:
    """Cópia fiel do laço do GPTDatasetV1, para comparação."""

    def __init__(self, ids, max_length, stride):
        self.x, self.y = [], []
        for i in range(0, len(ids) - max_length, stride):
            self.x.append(ids[i : i + max_length])
            self.y.append(ids[i + 1 : i + max_length + 1])

    def __len__(self):
        return len(self.x)


@pytest.mark.parametrize("max_length,stride", [(64, 32), (16, 16), (8, 4)])
def test_janelas_iguais_ao_livro(max_length, stride):
    ids = list(range(500))
    livro = DatasetDoLivro(ids, max_length, stride)
    nosso = GPTDatasetTokens(np.asarray(ids, dtype=np.uint16), max_length, stride)

    assert len(livro) == len(nosso)
    for k in range(len(livro)):
        x, y = nosso[k]
        assert torch.equal(x, torch.tensor(livro.x[k]))
        assert torch.equal(y, torch.tensor(livro.y[k]))


def test_alvo_e_a_entrada_deslocada():
    ds = GPTDatasetTokens(np.arange(100, dtype=np.uint16), 16, 16)
    x, y = ds[0]
    assert torch.equal(x[1:], y[:-1])


def test_ultima_janela_incompleta_fica_de_fora():
    # 10 tokens, janela 4, passo 4: range(0, 6, 4) = 0, 4 -> duas janelas.
    ds = GPTDatasetTokens(np.arange(10, dtype=np.uint16), 4, 4)
    assert len(ds) == 2


def test_stride_maior_que_janela_falha():
    with pytest.raises(ValueError):
        GPTDatasetTokens(np.arange(10), 4, 8)


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------


def test_mesma_semente_da_a_mesma_ordem_de_lotes():
    ids = np.arange(5000, dtype=np.uint16)
    ordem = []
    for _ in range(2):
        loader = criar_dataloader(ids, batch_size=8, max_length=16, stride=16,
                                  num_workers=0, seed=123)
        ordem.append(next(iter(loader))[0])
    assert torch.equal(ordem[0], ordem[1])


def test_sementes_diferentes_mudam_a_ordem():
    ids = np.arange(5000, dtype=np.uint16)
    a = next(iter(criar_dataloader(ids, 8, 16, 16, num_workers=0, seed=1)))[0]
    b = next(iter(criar_dataloader(ids, 8, 16, 16, num_workers=0, seed=2)))[0]
    assert not torch.equal(a, b)


def test_lote_tem_o_formato_da_configuracao():
    ids = np.arange(5000, dtype=np.uint16)
    x, y = next(iter(criar_dataloader(ids, 8, 16, 16, num_workers=0)))
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.int64


def test_retomada_pula_os_lotes_ja_vistos_sem_mudar_o_resto():
    """Retomar = recriar o loader com a mesma semente e descartar o prefixo.

    Se a ordem dependesse do estado global, o lote seguinte ao checkpoint seria
    outro e o treino veria dados diferentes dos que veria sem interrupção.
    """
    from itertools import islice

    ids = np.arange(5000, dtype=np.uint16)
    loader = criar_dataloader(ids, 8, 16, 16, num_workers=0, seed=123)
    completos = [x for x, _ in loader]

    retomado = islice(iter(criar_dataloader(ids, 8, 16, 16, num_workers=0, seed=123)), 3, None)
    for k, (x, _) in enumerate(retomado, start=3):
        assert torch.equal(x, completos[k])


# ---------------------------------------------------------------------------
# Localização do corpus e metadados
# ---------------------------------------------------------------------------


def montar_corpus_falso(pasta: Path, com_subpasta: bool, n_docs: int = 3) -> Corpus:
    raiz = pasta / "corpus" if com_subpasta else pasta
    (raiz / "tokenizer_16k").mkdir(parents=True, exist_ok=True)
    (raiz / "tokenizer_16k" / "tokenizer.json").write_text("{}", encoding="utf-8")
    docs = [f"documento {i}" for i in range(n_docs)]
    (raiz / "train.txt").write_text(SEP.join(docs), encoding="utf-8")
    (raiz / "val.txt").write_text(SEP.join(docs), encoding="utf-8")

    import pandas as pd

    tabela = pd.DataFrame(
        {
            "id": list(range(n_docs)),
            "sigla_centro": ["CCT"] * n_docs,
            "texto": docs,
        }
    )
    tabela.to_parquet(raiz / "train.parquet")
    tabela.to_parquet(raiz / "val.parquet")

    return localizar_corpus(pasta)


def test_localiza_corpus_em_subpasta(tmp_path):
    corpus = montar_corpus_falso(tmp_path, com_subpasta=True)
    assert corpus.raiz == tmp_path / "corpus"


def test_localiza_corpus_direto_na_work_dir(tmp_path):
    corpus = montar_corpus_falso(tmp_path, com_subpasta=False)
    assert corpus.raiz == tmp_path


def test_corpus_ausente_falha(tmp_path):
    with pytest.raises(FileNotFoundError):
        localizar_corpus(tmp_path)


def test_metadados_conferem_quando_a_ordem_e_a_do_txt(tmp_path):
    corpus = montar_corpus_falso(tmp_path, com_subpasta=True)
    verificar_metadados(corpus, "train", ler_documentos(corpus.train_txt))


def test_metadados_divergentes_falham(tmp_path):
    corpus = montar_corpus_falso(tmp_path, com_subpasta=True)
    documentos = ler_documentos(corpus.train_txt)
    documentos[0], documentos[1] = documentos[1], documentos[0]
    with pytest.raises(ValueError, match="fora de ordem"):
        verificar_metadados(corpus, "train", documentos)


def test_metadados_com_contagem_diferente_falham(tmp_path):
    corpus = montar_corpus_falso(tmp_path, com_subpasta=True)
    with pytest.raises(ValueError, match="linhas"):
        verificar_metadados(corpus, "train", ["só um"])


# ---------------------------------------------------------------------------
# Cache da pré-tokenização
# ---------------------------------------------------------------------------


def test_preparar_grava_e_reaproveita_o_cache(tmp_path):
    corpus = montar_corpus_falso(tmp_path, com_subpasta=True)
    tokenizer = TokenizadorFalso(corpus.tokenizer)

    primeira = preparar("train", corpus, tokenizer, tmp_path / "prep")
    # O segundo chamado emite "reaproveitando"; os ids precisam ser os mesmos.
    segunda = preparar("train", corpus, tokenizer, tmp_path / "prep")
    assert np.array_equal(primeira.ids, segunda.ids)
    assert np.array_equal(primeira.offsets, segunda.offsets)

    meta = json.loads((tmp_path / "prep" / "train_meta.json").read_text(encoding="utf-8"))
    assert meta["n_documentos"] == 3
    assert meta["n_tokens"] == 12
    assert meta["tokenizer_eot_id"] == 0


def test_forcar_refaz_a_pre_tokenizacao(tmp_path):
    corpus = montar_corpus_falso(tmp_path, com_subpasta=True)
    tokenizer = TokenizadorFalso(corpus.tokenizer)
    preparar("train", corpus, tokenizer, tmp_path / "prep")
    meta1 = (tmp_path / "prep" / "train_meta.json").read_text(encoding="utf-8")

    # Muda o tokenizador: o cache passa a estar inválido.
    (corpus.tokenizer).write_text('{"novo": true}', encoding="utf-8")
    preparar("train", corpus, tokenizer, tmp_path / "prep")
    meta2 = (tmp_path / "prep" / "train_meta.json").read_text(encoding="utf-8")
    assert meta1 != meta2
