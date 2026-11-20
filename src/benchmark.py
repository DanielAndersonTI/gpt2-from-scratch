"""Benchmark de custo: quantos tokens por segundo cada otimização entrega.

Mede o mesmo modelo por 200 passos em cinco configurações:

| variante | o que liga                                  |
|----------|---------------------------------------------|
| base     | nada: conta manual de atenção, fp32         |
| amp      | precisão mista fp16 com GradScaler          |
| sdpa     | atenção pelo kernel do PyTorch              |
| fused    | AdamW com atualização fundida               |
| compile  | `torch.compile` no modelo                   |
| todas    | todas as anteriores                         |

Cada variante é medida depois de alguns passos de aquecimento, porque o
primeiro passo na GPU paga a inicialização dos kernels e, no caso do
`compile`, a compilação do grafo. A memória é o pico alocado, não a reservada:
é o número que limita o tamanho do lote.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import Config, load_config
from src.data import criar_dataloader, localizar_corpus, preparar, TokenizadorUFCG
from src.logging_utils import CSVLogger, ThroughputMeter, utc_now
from src.model import GPTModel
from src.seeds import set_seed
from src.train import criar_otimizador, escolher_dispositivo

COLUNAS = [
    "variante",
    "passos",
    "tokens_por_segundo",
    "memoria_pico_mb",
    "segundos",
    "amp",
    "sdpa",
    "fused",
    "compile",
    "horario",
]

VARIANTES = {
    # nome: (amp, sdpa, fused, compile)
    "base": (False, False, False, False),
    "amp": (True, False, False, False),
    "sdpa": (False, True, False, False),
    "fused": (False, False, True, False),
    "compile": (False, False, False, True),
    "todas": (True, True, True, True),
}


def medir(cfg: Config, variante: str, passos: int | None = None, aquecimento: int = 10) -> dict:
    """Roda uma variante e devolve tokens/s, memória de pico e tempo."""
    amp, sdpa, fused, compile_ = VARIANTES[variante]
    device = escolher_dispositivo(cfg)
    set_seed(cfg.run.seed, deterministic=False)

    # -- dados -------------------------------------------------------------
    corpus = localizar_corpus(cfg.run.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    ids = preparar("train", corpus, tokenizer, cfg.abspath(cfg.data.preprocessed_dir)).ids
    loader = criar_dataloader(
        ids,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.context_length,
        stride=cfg.data.stride,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        seed=cfg.run.seed,
    )

    # -- modelo e otimizador ----------------------------------------------
    model: torch.nn.Module = GPTModel(cfg.model, sdpa=sdpa).to(device)
    otimizador = criar_otimizador(model, cfg, usar_fused=fused)
    usar_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    if compile_ and device.type == "cuda":
        model = torch.compile(model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    medidor = ThroughputMeter(window=20)
    alvo = passos or cfg.training.max_steps
    total_executado = 0
    inicio = time.perf_counter()

    model.train()
    iterator = iter(loader)
    for passo in range(alvo + aquecimento):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        otimizador.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=usar_amp):
            logits = model(x)
            perda = F.cross_entropy(logits.flatten(0, 1), y.flatten())

        scaler.scale(perda).backward()
        scaler.unscale_(otimizador)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        scaler.step(otimizador)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # Aquecimento não entra na média: mede-se o regime, não a partida.
        if passo >= aquecimento:
            medidor.update(cfg.tokens_per_step, dt)
            total_executado += 1

    segundos = time.perf_counter() - inicio
    memoria = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    )

    return {
        "variante": variante,
        "passos": total_executado,
        "tokens_por_segundo": medidor.tokens_per_second(),
        "memoria_pico_mb": memoria,
        "segundos": segundos,
        "amp": usar_amp,
        "sdpa": sdpa,
        "fused": fused and torch.cuda.is_available(),
        "compile": compile_,
        "horario": utc_now(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Mede tokens/s e memória por variante.")
    p.add_argument("--config", default="configs/benchmark.yaml")
    p.add_argument("--work-dir", default=None)
    p.add_argument(
        "--variant",
        default="todas-as-variantes",
        choices=[*VARIANTES, "todas-as-variantes"],
        help="uma variante por nome, ou todas-as-variantes para medir as seis",
    )
    p.add_argument("--steps", type=int, default=None, help="passos medidos por variante")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--saida",
        default="logs/benchmark.csv",
        help="CSV do benchmark, relativo a run.work_dir",
    )
    args = p.parse_args()

    cfg = load_config(args.config, work_dir=args.work_dir)
    variantes = (
        list(VARIANTES)
        if args.variant == "todas-as-variantes"
        else [args.variant]
    )

    # Arquivo próprio: o CSV do benchmark tem colunas diferentes do log de
    # treino e não pode dividir o mesmo cabeçalho.
    logger = CSVLogger(cfg.abspath(args.saida), COLUNAS)
    resultados = []
    for variante in variantes:
        print(f"medindo {variante}...")
        resultados.append(medir(cfg, variante, args.steps, args.warmup))
        logger.log(resultados[-1])
    logger.close()

    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
