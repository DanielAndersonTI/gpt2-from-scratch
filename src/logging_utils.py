"""Registro incremental do treino em CSV.

Por que CSV e não só o log de texto do console: se a sessão do Colab cair, o
que está em memória some. Cada linha é escrita e descarregada em disco assim
que o passo termina, então o pior caso é perder o passo em andamento. O arquivo
também é pequeno o bastante para ir para o Git e virar figura do artigo.

Formato: uma linha por evento. O evento é `passo`, `avaliacao` ou `amostra`;
quem analisa filtra pela coluna `evento`.
"""

from __future__ import annotations

import csv
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    """Instante atual em UTC, sem microssegundos, para caber no CSV."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def configure_csv_logger(path, fields: Iterable[str]) -> "CSVLogger":
    """Atalho: cria o logger já com a lista de colunas."""
    return CSVLogger(path, fields)


class CSVLogger:
    """Escreve linhas em CSV, com cabeçalho na criação e flush por linha.

    Uso:
        logger = CSVLogger(caminho, ["passo", "loss"])
        logger.log({"passo": 1, "loss": 2.31})
        logger.close()

    Se o arquivo já existe, ele é continuado e o cabeçalho não é reescrito.
    Isso é o que permite retomar um treino sem duplicar colunas.
    """

    def __init__(self, path: str | Path, fields: Iterable[str]) -> None:
        self.path = Path(path)
        self.fields = list(fields)
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"colunas repetidas no log: {self.fields}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "a" (append): retomada acrescenta linhas em vez de apagar o histórico.
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields, extrasaction="raise")

        # Arquivo novo ou vazio recebe o cabeçalho; arquivo existente não.
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row: dict[str, Any]) -> None:
        """Grava uma linha. Todas as colunas precisam estar presentes.

        Exigir o conjunto completo evita a linha meio preenchida, que depois
        quebra a leitura no pandas sem aviso.
        """
        faltando = set(self.fields) - set(row)
        if faltando:
            raise ValueError(f"linha sem as colunas {sorted(faltando)}")
        self._writer.writerow(row)
        self._file.flush()  # o que está no disco sobrevive à queda da sessão

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class ThroughputMeter:
    """Média móvel de tokens por segundo.

    A primeira medição da GPU inclui a inicialização dos kernels e sai muito
    abaixo do regime. Por isso a média cobre apenas os últimos `window` passos:
    o número reportado é o do regime, não o do aquecimento.
    """

    def __init__(self, window: int = 20) -> None:
        if window <= 0:
            raise ValueError("a janela precisa ser positiva")
        self._tokens: deque[int] = deque(maxlen=window)
        self._segundos: deque[float] = deque(maxlen=window)

    def update(self, n_tokens: int, segundos: float) -> None:
        if segundos <= 0:
            raise ValueError("o intervalo precisa ser positivo")
        self._tokens.append(n_tokens)
        self._segundos.append(segundos)

    def tokens_per_second(self) -> float:
        """Tokens/s na janela. Devolve 0.0 enquanto não houver medição."""
        total = sum(self._segundos)
        return sum(self._tokens) / total if total > 0 else 0.0
