"""Corpus, tokenizador, pré-tokenização e janelas de treino.

O fluxo tem quatro etapas, e cada uma deixa um artefato em disco:

1. localizar o corpus e o tokenizador (o mesmo código roda no Colab e aqui);
2. ler os documentos de `train.txt` e `val.txt`, separados por
   `\\n<|endoftext|>\\n`;
3. codificar cada documento e concatenar, guardando o deslocamento de cada um;
4. cortar a sequência em janelas de `context_length` tokens.

A etapa 3 é feita uma vez só. O resultado é um `.npy` com ids em `uint16`, que
cabe em 55 MB para os 27 milhões de tokens de treino. Rodar isso a cada sessão
do Colab custaria minutos e é desnecessário: os ids não mudam.

A ordem dos documentos no `.txt` é a mesma das linhas do `.parquet`, e isso é
verificado, não assumido (veja `verificar_metadados`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

# Separador de documentos nos arquivos .txt do corpus.
SEP = "\n<|endoftext|>\n"
EOT = "<|endoftext|>"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def sha256_arquivo(caminho: str | Path, bloco: int = 1 << 20) -> str:
    """Hash SHA-256 de um arquivo, lido em blocos de 1 MB.

    Serve para provar qual corpus e qual tokenizador geraram os ids salvos.
    Sem isso, um `.npy` antigo pode ser reutilizado depois de o corpus mudar.
    """
    h = hashlib.sha256()
    with Path(caminho).open("rb") as f:
        while pedaco := f.read(bloco):
            h.update(pedaco)
    return h.hexdigest()


def utc_agora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Tokenizador
# ---------------------------------------------------------------------------


class TokenizadorUFCG:
    """Dá ao tokenizador do corpus a interface do `tiktoken` usada no livro.

    O livro chama `encode(texto, allowed_special={...})` e `decode(ids)`. Com
    este envelope, o código do livro roda sem alteração. `allowed_special` é
    aceito e ignorado: no `tokenizer.json` o `<|endoftext|>` já é um token
    especial registrado, então aparece decodificado normalmente.
    """

    def __init__(self, caminho: str | Path) -> None:
        self.caminho = Path(caminho)
        self.tok = Tokenizer.from_file(str(caminho))
        self.eot_id = self.tok.token_to_id(EOT)
        self.n_vocab = self.tok.get_vocab_size()

    def encode(self, texto: str, allowed_special: set[str] | None = None) -> list[int]:
        return self.tok.encode(texto).ids

    def encode_batch(self, textos: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(textos)]

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def __repr__(self) -> str:
        return f"TokenizadorUFCG(vocab={self.n_vocab}, eot_id={self.eot_id})"


# ---------------------------------------------------------------------------
# Localização do corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """Caminhos reais dos arquivos do corpus."""

    raiz: Path
    tokenizer: Path
    train_txt: Path
    val_txt: Path
    train_parquet: Path
    val_parquet: Path

    def txt(self, nome: str) -> Path:
        return self.train_txt if nome == "train" else self.val_txt

    def parquet(self, nome: str) -> Path:
        return self.train_parquet if nome == "train" else self.val_parquet


def localizar_corpus(work_dir: str | Path) -> Corpus:
    """Encontra o corpus tanto em `work_dir/corpus` quanto direto em `work_dir`.

    A pasta do zip não tem diretório de topo: quem descompacta em `corpus/`
    fica com `corpus/train.txt`, e quem descompacta na raiz fica com
    `train.txt`. As duas formas são aceitas, e a primeira é preferida.
    """
    trabalho = Path(work_dir)
    raiz = next(
        (c for c in (trabalho / "corpus", trabalho) if (c / "train.txt").exists()),
        None,
    )
    if raiz is None:
        raise FileNotFoundError(
            f"train.txt não encontrado em {trabalho / 'corpus'} nem em {trabalho}"
        )

    tokenizer = raiz / "tokenizer_16k" / "tokenizer.json"
    if not tokenizer.exists():
        tokenizer = trabalho / "tokenizer_16k" / "tokenizer.json"
    if not tokenizer.exists():
        raise FileNotFoundError(f"tokenizer.json não encontrado a partir de {trabalho}")

    faltando = [
        nome
        for nome in ("val.txt", "train.parquet", "val.parquet")
        if not (raiz / nome).exists()
    ]
    if faltando:
        raise FileNotFoundError(f"arquivos ausentes em {raiz}: {faltando}")

    return Corpus(
        raiz=raiz,
        tokenizer=tokenizer,
        train_txt=raiz / "train.txt",
        val_txt=raiz / "val.txt",
        train_parquet=raiz / "train.parquet",
        val_parquet=raiz / "val.parquet",
    )


# ---------------------------------------------------------------------------
# Leitura e pré-tokenização
# ---------------------------------------------------------------------------


def ler_documentos(txt_path: str | Path) -> list[str]:
    """Divide o arquivo em documentos. Documento vazio não entra.

    O filtro usa `strip()` só para decidir se o documento existe; o texto é
    mantido como está, porque o espaço em branco faz parte dos bytes contados
    no BPB.
    """
    texto = Path(txt_path).read_text(encoding="utf-8")
    return [doc for doc in texto.split(SEP) if doc.strip()]


def contar_palavras(documentos: list[str]) -> int:
    """Total de palavras, na definição do enunciado: `split()` por espaço."""
    return sum(len(doc.split()) for doc in documentos)


@dataclass(frozen=True)
class CorpusPreTokenizado:
    """Ids concatenados e o deslocamento de cada documento."""

    nome: str
    ids: np.ndarray  # uint16, forma (n_tokens,)
    offsets: np.ndarray  # int64, forma (n_docs + 1,)

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[0])

    @property
    def n_documentos(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def documento_da_posicao(self, posicao: int) -> int:
        """Índice do documento que contém a posição de token indicada.

        `searchsorted` faz busca binária: o custo é log(n_docs), não uma
        varredura. A posição do `<|endoftext|>` pertence ao documento que ele
        fecha.
        """
        return int(np.searchsorted(self.offsets, posicao, side="right") - 1)


def pre_tokenizar(
    documentos: list[str],
    tokenizer: TokenizadorUFCG,
    tamanho_lote: int = 1000,
    progresso: bool = True,
) -> CorpusPreTokenizado:
    """Codifica cada documento e anexa o `<|endoftext|>` ao fim.

    Documentos são codificados separadamente de propósito. Codificar o arquivo
    inteiro de uma vez é mais rápido, porém o BPE poderia fundir caracteres
    entre o fim de um documento e o começo do outro, e o deslocamento de cada
    documento deixaria de ser recuperável.
    """
    pedacos: list[np.ndarray] = []
    offsets: list[int] = [0]
    total = 0

    for inicio in range(0, len(documentos), tamanho_lote):
        lote = documentos[inicio : inicio + tamanho_lote]
        for ids in tokenizer.encode_batch(lote):
            pedacos.append(np.asarray(ids + [tokenizer.eot_id], dtype=np.uint16))
            total += len(ids) + 1
            offsets.append(total)
        if progresso:
            print(f"  {min(inicio + tamanho_lote, len(documentos)):,} de {len(documentos):,} documentos")

    return CorpusPreTokenizado(
        nome="",
        ids=np.concatenate(pedacos),
        offsets=np.asarray(offsets, dtype=np.int64),
    )


def preparar(
    nome: str,
    corpus: Corpus,
    tokenizer: TokenizadorUFCG,
    preprocessed_dir: str | Path,
    forcar: bool = False,
) -> CorpusPreTokenizado:
    """Pré-tokeniza e grava `{nome}_ids.npy`, `{nome}_offsets.npy` e `{nome}_meta.json`.

    Se os três arquivos existem e o hash do tokenizador bate, a pré-tokenização
    é pulada. `forcar=True` refaz tudo, para quando o corpus mudar.
    """
    pasta = Path(preprocessed_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_ids = pasta / f"{nome}_ids.npy"
    caminho_offsets = pasta / f"{nome}_offsets.npy"
    caminho_meta = pasta / f"{nome}_meta.json"

    hash_tokenizer = sha256_arquivo(tokenizer.caminho)

    if not forcar and caminho_ids.exists() and caminho_offsets.exists() and caminho_meta.exists():
        meta = json.loads(caminho_meta.read_text(encoding="utf-8"))
        if meta.get("tokenizer_sha256") == hash_tokenizer:
            print(f"{nome}: reaproveitando ids em {caminho_ids}")
            return CorpusPreTokenizado(
                nome=nome,
                ids=np.load(caminho_ids),
                offsets=np.load(caminho_offsets),
            )

    print(f"{nome}: codificando documentos...")
    documentos = ler_documentos(corpus.txt(nome))
    pretok = pre_tokenizar(documentos, tokenizer)
    pretok = CorpusPreTokenizado(nome=nome, ids=pretok.ids, offsets=pretok.offsets)

    np.save(caminho_ids, pretok.ids)
    np.save(caminho_offsets, pretok.offsets)

    meta = {
        "nome": nome,
        "n_documentos": pretok.n_documentos,
        "n_tokens": pretok.n_tokens,
        "n_palavras": contar_palavras(documentos),
        "tokenizer_sha256": hash_tokenizer,
        "tokenizer_vocab": tokenizer.n_vocab,
        "tokenizer_eot_id": tokenizer.eot_id,
        "ids_sha256": sha256_arquivo(caminho_ids),
        "offsets_sha256": sha256_arquivo(caminho_offsets),
        "corpus_txt_sha256": sha256_arquivo(corpus.txt(nome)),
        "criado_em": utc_agora(),
    }
    caminho_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{nome}: {pretok.n_documentos:,} documentos, {pretok.n_tokens:,} tokens")
    return pretok


# ---------------------------------------------------------------------------
# Junção com os metadados
# ---------------------------------------------------------------------------


def carregar_metadados(corpus: Corpus, nome: str):
    """Lê o parquet e acrescenta a coluna `doc_index`, que liga à posição no txt."""
    import pandas as pd

    tabela = pd.read_parquet(corpus.parquet(nome))
    tabela.insert(0, "doc_index", np.arange(len(tabela)))
    return tabela


def verificar_metadados(corpus: Corpus, nome: str, documentos: list[str]) -> None:
    """Confere que a linha *i* do parquet é o documento *i* do txt.

    É essa igualdade que autoriza juntar `sigla_centro`, `curso` e `ano` a uma
    janela pelo índice do documento. Se um dia o corpus for regerado em outra
    ordem, a verificação falha em vez de atribuir metadado errado em silêncio.
    """
    tabela = carregar_metadados(corpus, nome)
    if len(tabela) != len(documentos):
        raise ValueError(
            f"{nome}: parquet com {len(tabela)} linhas e txt com {len(documentos)} documentos"
        )
    textos = tabela["texto"].tolist()
    divergentes = [i for i, (a, b) in enumerate(zip(textos, documentos)) if a != b]
    if divergentes:
        raise ValueError(
            f"{nome}: {len(divergentes)} documentos fora de ordem; "
            f"primeiro índice divergente: {divergentes[0]}"
        )


# ---------------------------------------------------------------------------
# Janelas
# ---------------------------------------------------------------------------


class GPTDatasetTokens(Dataset):
    """Janelas do `GPTDatasetV1` do livro, recortadas sob demanda.

    O original guarda um tensor por janela já no `__init__`, o que para 27
    milhões de tokens ocupa vários GB. Aqui os ids ficam em um único tensor e
    cada janela é um recorte dele. As janelas são idênticas às do livro: o
    teste `test_janelas_iguais_ao_livro` compara as duas implementações.

    O número de janelas segue a fórmula do livro, `len(range(0, N - L, stride))`,
    então os últimos tokens que não completam uma janela ficam de fora.
    """

    def __init__(self, token_ids, max_length: int, stride: int) -> None:
        if max_length <= 0 or stride <= 0:
            raise ValueError("max_length e stride precisam ser positivos")
        if stride > max_length:
            raise ValueError("stride maior que max_length deixaria buracos no texto")
        self.ids = torch.as_tensor(token_ids, dtype=torch.int64)
        self.max_length = max_length
        self.stride = stride
        self.n = len(range(0, len(self.ids) - max_length, stride))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        i = idx * self.stride
        x = self.ids[i : i + self.max_length]
        y = self.ids[i + 1 : i + self.max_length + 1]
        return x, y


def criar_dataloader(
    token_ids,
    batch_size: int = 128,
    max_length: int = 256,
    stride: int = 256,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 2,
    seed: int = 123,
) -> DataLoader:
    """DataLoader no molde do `create_dataloader_v1` do livro.

    `pin_memory` acelera a cópia para a GPU. O `generator` com semente própria
    torna a ordem dos lotes reprodutível, que é o que permite retomar um treino
    interrompido sem repetir nem pular lote.
    """
    dataset = GPTDatasetTokens(token_ids, max_length, stride)
    gerador = torch.Generator()
    gerador.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=gerador,
    )


def semente_da_epoca(base: int, epoca: int) -> int:
    """Semente do embaralhamento de cada época.

    É o que permite recriar a época e pular os lotes já vistos: a sequência é
    a mesma toda vez que a época começa.
    """
    return base + epoca


# ---------------------------------------------------------------------------
# Execução direta: prepara e confere os dados
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Pré-tokeniza o corpus e confere os números.")
    p.add_argument("--work-dir", default=".", help="pasta que contém o corpus")
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--forcar", action="store_true", help="refaz a pré-tokenização")
    p.add_argument("--verificar-parquet", action="store_true", help="confere a ordem dos metadados")
    args = p.parse_args()

    corpus = localizar_corpus(args.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    print(f"Corpus em {corpus.raiz}")
    print(tokenizer)

    pasta = Path(args.preprocessed_dir) if args.preprocessed_dir else Path(args.work_dir) / "preprocessed"

    resumo = {}
    for nome in ("train", "val"):
        pretok = preparar(nome, corpus, tokenizer, pasta, forcar=args.forcar)
        resumo[nome] = pretok

        if args.verificar_parquet:
            documentos = ler_documentos(corpus.txt(nome))
            verificar_metadados(corpus, nome, documentos)
            print(f"{nome}: metadados conferem com o txt documento a documento")

        meta = json.loads((pasta / f"{nome}_meta.json").read_text(encoding="utf-8"))
        fertilidade = pretok.n_tokens / meta["n_palavras"]
        janelas = len(range(0, pretok.n_tokens - 256, 256))
        print(
            f"{nome}: {pretok.n_documentos:,} docs | {pretok.n_tokens:,} tokens | "
            f"{meta['n_palavras']:,} palavras | fertilidade {fertilidade:.3f} | "
            f"{janelas:,} janelas de 256"
        )


if __name__ == "__main__":
    main()
