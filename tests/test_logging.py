"""Testes do log CSV e da medição de throughput.

O que está sendo garantido: o cabeçalho aparece uma vez só, a retomada
acrescenta linhas sem duplicar coluna, e nenhuma linha sai incompleta.
"""

from __future__ import annotations

import csv

import pytest

from src.logging_utils import CSVLogger, ThroughputMeter, utc_now

COLUNAS = ["evento", "passo", "loss"]


def ler(caminho):
    with caminho.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_cria_cabecalho_e_grava_linha(tmp_path):
    caminho = tmp_path / "log.csv"
    with CSVLogger(caminho, COLUNAS) as logger:
        logger.log({"evento": "passo", "passo": 1, "loss": 9.68})

    linhas = ler(caminho)
    assert len(linhas) == 1
    assert linhas[0]["evento"] == "passo"
    assert float(linhas[0]["loss"]) == 9.68


def test_cabecalho_nao_se_repete_ao_reabrir(tmp_path):
    caminho = tmp_path / "log.csv"
    with CSVLogger(caminho, COLUNAS) as logger:
        logger.log({"evento": "passo", "passo": 1, "loss": 9.6})
    with CSVLogger(caminho, COLUNAS) as logger:
        logger.log({"evento": "passo", "passo": 2, "loss": 8.9})

    linhas = ler(caminho)
    assert len(linhas) == 2
    assert [int(l["passo"]) for l in linhas] == [1, 2]
    # Uma única linha de cabeçalho no arquivo bruto.
    assert caminho.read_text(encoding="utf-8").count("evento,passo,loss") == 1


def test_cria_as_pastas_intermediarias(tmp_path):
    caminho = tmp_path / "logs" / "e1" / "log.csv"
    with CSVLogger(caminho, COLUNAS):
        pass
    assert caminho.exists()


def test_linha_sem_coluna_falha(tmp_path):
    caminho = tmp_path / "log.csv"
    with CSVLogger(caminho, COLUNAS) as logger:
        with pytest.raises(ValueError, match="loss"):
            logger.log({"evento": "passo", "passo": 1})


def test_coluna_extra_falha(tmp_path):
    caminho = tmp_path / "log.csv"
    with CSVLogger(caminho, COLUNAS) as logger:
        with pytest.raises(ValueError):
            logger.log({"evento": "passo", "passo": 1, "loss": 1.0, "bpb": 2.0})


def test_colunas_repetidas_falham(tmp_path):
    with pytest.raises(ValueError, match="repetidas"):
        CSVLogger(tmp_path / "log.csv", ["passo", "passo"])


def test_flush_deixa_a_linha_visivel_no_disco(tmp_path):
    caminho = tmp_path / "log.csv"
    logger = CSVLogger(caminho, COLUNAS)
    logger.log({"evento": "passo", "passo": 1, "loss": 9.0})
    # Sem close(): simula a sessão caindo logo depois do passo.
    assert len(ler(caminho)) == 1
    logger.close()


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def test_throughput_medio_da_janela():
    medidor = ThroughputMeter(window=2)
    medidor.update(1000, 1.0)  # sai da janela depois de dois updates
    medidor.update(2000, 1.0)
    medidor.update(3000, 2.0)
    # Janela = últimos dois: (2000 + 3000) / (1 + 2)
    assert medidor.tokens_per_second() == pytest.approx(1666.6667, rel=1e-4)


def test_throughput_sem_medicao_e_zero():
    assert ThroughputMeter().tokens_per_second() == 0.0


def test_throughput_rejeita_intervalo_invalido():
    medidor = ThroughputMeter()
    with pytest.raises(ValueError):
        medidor.update(100, 0.0)


def test_utc_now_esta_no_formato_iso():
    valor = utc_now()
    assert len(valor) == 19
    assert valor[4] == "-" and valor[10] == "T"
