"""Análise do roteamento do MoE: quem trabalha, em que token e em que centro.

Responde às duas perguntas do relatório (T34 e T35):

1. **Equilíbrio.** Quanto do tráfego cada especialista recebe, por bloco, e
   quão decidido está o roteador. Um roteador colapsado manda quase tudo para
   um especialista e a entropia cai; um roteador difuso divide o tráfego mas
   gasta capacidade à toa.
2. **Especialização.** Se o tráfego se distribui de forma diferente por token
   frequente ou por centro do corpus, o roteador aprendeu algo além da
   frequência. Se for igual em toda parte, ele só aprendeu a repartir.

O script roda sobre uma partição e devolve um JSON com as três tabelas. Ele é
de leitura: não treina, não altera checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.data import (
    carregar_metadados,
    criar_dataloader,
    localizar_corpus,
    preparar,
    TokenizadorUFCG,
)
from src.evaluate import carregar_modelo
from src.logging_utils import utc_now


def _entropia_normalizada(probabilidades: np.ndarray, n_experts: int) -> np.ndarray:
    """Entropia de Shannon por linha, dividida pela entropia máxima, em [0, 1].

    1 significa roteador completamente indeciso, com massa igual em todos os
    especialistas; 0 significa que ele aponta sempre o mesmo. Recebe uma matriz
    (tokens, especialistas) e devolve um valor por token, vetorizado: chamar
    `log` uma vez por token custaria milhões de chamadas em Python.
    """
    p = np.clip(probabilidades, 1e-12, None)
    entropia = -(p * np.log(p)).sum(axis=1)
    return entropia / np.log(n_experts)


def analisar(
    checkpoint: str | Path,
    split: str = "val",
    device: str = "auto",
    max_lotes: int | None = None,
    work_dir: str | Path | None = None,
    saida: str | Path | None = None,
) -> dict:
    """Coleta utilização, entropia, especialização por token e por centro."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dispositivo = torch.device(device)

    model, cfg = carregar_modelo(checkpoint, dispositivo)
    if work_dir is not None:
        cfg = replace(cfg, run=replace(cfg.run, work_dir=str(work_dir)))
    if cfg.model.ffn_type != "moe":
        raise ValueError("este checkpoint não usa MoE; não há roteamento a analisar")

    n_experts = cfg.model.moe.n_experts
    n_blocos = len(model.trf_blocks)

    corpus = localizar_corpus(cfg.run.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    pretokenizado = preparar(split, corpus, tokenizer, cfg.abspath(cfg.data.preprocessed_dir))
    tabela = carregar_metadados(corpus, split)
    centros = tabela["sigla_centro"].tolist()

    loader = criar_dataloader(
        pretokenizado.ids,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.context_length,
        stride=cfg.data.stride,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        seed=cfg.run.seed,
    )

    # Acumuladores. `por_token` tem uma linha por token do vocabulário e uma
    # coluna por especialista: guarda a contagem de despachos top-1.
    despachos = np.zeros((n_blocos, n_experts), dtype=np.int64)
    soma_entropia = np.zeros(n_blocos, dtype=np.float64)
    por_token = np.zeros((n_blocos, cfg.model.vocab_size, n_experts), dtype=np.int32)
    por_centro: list[dict[str, np.ndarray]] = [defaultdict(lambda: np.zeros(n_experts)) for _ in range(n_blocos)]
    n_lotes = 0
    n_tokens_lote = 0

    model.eval()
    with torch.no_grad():
        for k, (x, y) in enumerate(loader):
            if max_lotes is not None and k >= max_lotes:
                break
            x, y = x.to(dispositivo), y.to(dispositivo)
            model(x)

            # Posição de cada alvo no corpus concatenado. O dataset produz
            # janelas que começam em i * stride; o lote k cobre as janelas
            # k * batch_size até k * batch_size + batch_size - 1.
            b, n = y.shape
            inicio = (k * cfg.data.batch_size + np.arange(b)) * cfg.data.stride
            posicoes = inicio[:, None] + np.arange(1, n + 1)[None, :]
            documentos = np.searchsorted(pretokenizado.offsets, posicoes, side="right") - 1
            centros_do_lote = np.array(centros, dtype=object)[documentos.reshape(-1)]

            alvos = y.reshape(-1).cpu().numpy()
            n_tokens_lote += alvos.size
            for i, bloco in enumerate(model.trf_blocks):
                probs, indices = bloco.ff.ultimo_roteamento
                top1 = indices[:, 0].reshape(-1).cpu().numpy()
                p = probs.reshape(-1, n_experts).float().cpu().numpy()

                despachos[i] += np.bincount(top1, minlength=n_experts)
                soma_entropia[i] += float(_entropia_normalizada(p, n_experts).sum())
                np.add.at(por_token[i], (alvos, top1), 1)
                for centro in np.unique(centros_do_lote):
                    mascara = centros_do_lote == centro
                    por_centro[i][str(centro)] += np.bincount(top1[mascara], minlength=n_experts)

            n_lotes += 1

    if n_lotes == 0:
        raise ValueError("nenhum lote avaliado")

    # -- montagem das tabelas ------------------------------------------------
    utilizacao = {
        f"bloco_{i}": (despachos[i] / max(despachos[i].sum(), 1)).round(4).tolist()
        for i in range(n_blocos)
    }
    entropia = {
        f"bloco_{i}": round(float(soma_entropia[i] / n_tokens_lote), 4)
        for i in range(n_blocos)
    }

    # Tokens mais frequentes na partição, com a repartição do bloco 0 e do
    # último bloco. Blocos do meio ficam no JSON completo.
    frequencia = por_token.sum(axis=(0, 2))
    mais_frequentes = np.argsort(frequencia)[::-1][:20]
    tokens = []
    for tid in mais_frequentes:
        total = int(frequencia[tid])
        if total == 0:
            continue
        entrada = {
            "id": int(tid),
            "token": tokenizer.decode([int(tid)]).replace("\n", "\\n"),
            "frequencia": total,
            "por_bloco": {
                f"bloco_{i}": (por_token[i, tid] / total).round(4).tolist()
                for i in range(n_blocos)
            },
        }
        tokens.append(entrada)

    centro_tabela = {}
    for i in range(n_blocos):
        linhas = {}
        for centro, contagem in por_centro[i].items():
            total = contagem.sum()
            if total == 0:
                continue
            fracao = contagem / total
            linhas[centro] = {
                "fracao": fracao.round(4).tolist(),
                "especialista_dominante": int(fracao.argmax()),
                "despachos": int(total),
            }
        centro_tabela[f"bloco_{i}"] = dict(
            sorted(linhas.items(), key=lambda item: -item[1]["despachos"])
        )

    resultado = {
        "experimento": cfg.run.name,
        "split": split,
        "n_experts": n_experts,
        "top_k": cfg.model.moe.top_k,
        "lotes_avaliados": n_lotes,
        "utilizacao_top1": utilizacao,
        "entropia_normalizada": entropia,
        "tokens_frequentes": tokens,
        "por_centro": centro_tabela,
        "horario": utc_now(),
    }

    if saida is not None:
        destino = Path(saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    return resultado


def _imprimir(resultado: dict) -> None:
    """Resumo legível no terminal, para não abrir o JSON a cada olhada."""
    print(f"{resultado['experimento']} | {resultado['lotes_avaliados']} lotes")
    print("\nutilização top-1 por bloco:")
    for bloco, fracoes in resultado["utilizacao_top1"].items():
        maximo = max(fracoes)
        minimo = min(fracoes)
        print(f"  {bloco}: máx {maximo:.3f} mín {minimo:.3f}  {fracoes}")
    print("\nentropia normalizada (1 = indeciso, 0 = sempre o mesmo):")
    for bloco, valor in resultado["entropia_normalizada"].items():
        print(f"  {bloco}: {valor:.3f}")
    print("\n3 tokens mais frequentes, repartição no primeiro e no último bloco:")
    for entrada in resultado["tokens_frequentes"][:3]:
        blocos = list(entrada["por_bloco"])
        print(
            f"  {entrada['token']!r} (id {entrada['id']}, n={entrada['frequencia']}): "
            f"{blocos[0]}={entrada['por_bloco'][blocos[0]]} "
            f"{blocos[-1]}={entrada['por_bloco'][blocos[-1]]}"
        )
    print("\ncentros com mais tráfego (último bloco):")
    ultimo = list(resultado["por_centro"])[-1]
    for centro, dados in list(resultado["por_centro"][ultimo].items())[:5]:
        print(f"  {centro}: dominante {dados['especialista_dominante']} | {dados['despachos']} despachos")


def main() -> None:
    p = argparse.ArgumentParser(description="Analisa o roteamento de um checkpoint MoE.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-lotes", type=int, default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--saida", default=None)
    args = p.parse_args()

    resultado = analisar(
        args.checkpoint,
        split=args.split,
        device=args.device,
        max_lotes=args.max_lotes,
        work_dir=args.work_dir,
        saida=args.saida,
    )
    _imprimir(resultado)
    print("\nJSON completo:\n" + json.dumps(
        {k: v for k, v in resultado.items() if k not in {"tokens_frequentes", "por_centro"}},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
