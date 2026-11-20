"""Ajustes comuns aos testes deste repositório.

Coloca a raiz do projeto no `sys.path` para que `import src.config` funcione
sem instalar o pacote, e define o corpus sintético usado nos testes de
ponta a ponta. Assim os testes rodam com `pytest` puro, do jeito que vão rodar
no Colab.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from apoio import escrever_corpus_sintetico  # noqa: E402  (precisa do sys.path acima)


@pytest.fixture(scope="session")
def projeto_sintetico(tmp_path_factory):
    """Corpus e tokenizador de mentira, criados uma vez por sessão."""
    raiz = tmp_path_factory.mktemp("projeto")
    vocab = escrever_corpus_sintetico(raiz / "corpus")
    return raiz, vocab
