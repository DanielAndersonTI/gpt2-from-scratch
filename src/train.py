"""Loop de treino do Apêndice D, com retomada e execução autônoma.

O loop é o do livro, com as quatro melhorias que o enunciado exige para a
pós-graduação:

1. aquecimento linear e decaimento cosseno no learning rate;
2. corte de gradiente por norma;
3. AdamW com weight decay só nas matrizes, não em vieses e normas;
4. precisão mista com `GradScaler` (a T4 não tem bf16 nativo).

Além disso, o que faz o treino sobreviver a uma sessão que cai: checkpoint
atômico a cada N passos, log em CSV com `flush` por linha e retomada que recria
a época e pula os lotes já vistos. O orçamento é contado em passos, então um
treino interrompido e retomado consome exatamente os mesmos 27.426.816 tokens
de um treino contínuo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import Config, load_config
from src.data import (
    criar_dataloader,
    localizar_corpus,
    preparar,
    semente_da_epoca,
    TokenizadorUFCG,
)
from src.logging_utils import CSVLogger, ThroughputMeter, utc_now
from src.model import GPTModel
from src.seeds import capture_rng_state, restore_rng_state, set_seed

# Colunas do CSV. Uma linha por evento; evento vazio em coluna que não se aplica.
COLUNAS = [
    "evento",          # passo | avaliacao | fim
    "passo",
    "epoca",
    "loss_treino",
    "loss_val",
    "lr",
    "tokens_por_segundo",
    "memoria_pico_mb",
    "tempo_s",
    "horario",
]


# ---------------------------------------------------------------------------
# Dispositivo, modelo e otimizador
# ---------------------------------------------------------------------------


def escolher_dispositivo(cfg: Config) -> torch.device:
    """Resolve `auto` para cuda se houver GPU, senão cpu."""
    if cfg.run.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.run.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device = cuda, mas nenhuma GPU foi encontrada")
    return torch.device(cfg.run.device)


def descrever_dispositivo(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def criar_otimizador(model: torch.nn.Module, cfg: Config, usar_fused: bool) -> torch.optim.Optimizer:
    """AdamW com dois grupos: matrizes com weight decay, o resto sem.

    Aplicar weight decay em viés e em escala de LayerNorm puxaria esses
    parâmetros para zero sem necessidade. O Apêndice D separa os grupos
    exatamente por isso.
    """
    com_decay, sem_decay = [], []
    for nome, parametro in model.named_parameters():
        if not parametro.requires_grad:
            continue
        if parametro.dim() >= 2:  # matrizes de peso
            com_decay.append(parametro)
        else:  # vieses e ganhos de norma
            sem_decay.append(parametro)

    grupos = [
        {"params": com_decay, "weight_decay": cfg.training.weight_decay},
        {"params": sem_decay, "weight_decay": 0.0},
    ]

    # fused junta as atualizações em um kernel; só existe na CUDA.
    fused = usar_fused and torch.cuda.is_available()
    return torch.optim.AdamW(grupos, lr=cfg.training.lr, betas=(0.9, 0.999), fused=fused)


def lr_do_passo(passo: int, cfg: Config) -> float:
    """Aquecimento linear até `warmup_steps`, depois cosseno até o piso.

    O piso é `lr * min_lr_ratio`; sem ele o passo final ficaria tão pequeno que
    o modelo praticamente pararia de aprender nas últimas centenas de passos.
    """
    pico = cfg.training.lr
    piso = pico * cfg.training.min_lr_ratio
    aquecimento = cfg.warmup_steps

    if passo < aquecimento:
        return pico * (passo + 1) / aquecimento

    progresso = (passo - aquecimento) / max(1, cfg.training.max_steps - aquecimento)
    return piso + 0.5 * (pico - piso) * (1.0 + math.cos(math.pi * min(progresso, 1.0)))


def modelo_bruto(model: torch.nn.Module) -> torch.nn.Module:
    """Devolve o módulo original quando `torch.compile` embrulhou o modelo.

    O módulo compilado guarda o original em `_orig_mod`, e o `state_dict` dele
    tem o prefixo `_orig_mod.` nas chaves. Salvar sempre o original mantém o
    checkpoint legível tanto com quanto sem compilação.
    """
    return getattr(model, "_orig_mod", model)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------


@torch.no_grad()
def avaliar(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    usar_amp: bool = False,
    max_batches: int | None = None,
) -> float:
    """Perda média de validação, em nats por token.

    A validação usa janelas sem sobreposição e sem embaralhar. O valor é a
    média por lote, ponderada igualmente; como todos os lotes têm o mesmo
    tamanho, isso equivale à média por token.

    `max_batches` limita a amostra. Na GPU a validação inteira custa cerca de
    um segundo nos lotes de 128; o limite existe para o teste de fumaça em CPU,
    onde varrer a validação inteira levaria minutos. Quando o limite é usado,
    ele vale para todas as avaliações do experimento, então a curva continua
    comparável dentro dele.
    """
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        if max_batches is not None and n >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=usar_amp):
            logits = model(x)
            perda = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        total += perda.item()
        n += 1
    model.train()
    return total / n if n else float("nan")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def salvar_checkpoint(
    caminho: Path,
    model: torch.nn.Module,
    otimizador: torch.optim.Optimizer,
    scaler,
    passo: int,
    cfg: Config,
) -> None:
    """Grava o checkpoint de forma atômica.

    Escreve em `.tmp` e depois troca o nome com `os.replace`. Se a sessão cair
    no meio da gravação, o checkpoint anterior continua íntegro: nunca fica um
    arquivo pela metade no lugar do último ponto bom.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(".tmp")

    torch.save(
        {
            "modelo": modelo_bruto(model).state_dict(),
            "otimizador": otimizador.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "passo": passo,
            "config": cfg.to_dict(),
            "rng": capture_rng_state(),
            "horario": utc_now(),
        },
        temporario,
    )
    os.replace(temporario, caminho)


def carregar_checkpoint(
    caminho: Path,
    model: torch.nn.Module,
    otimizador: torch.optim.Optimizer,
    scaler,
) -> int:
    """Restaura modelo, otimizador, scaler e geradores. Devolve o passo."""
    checkpoint = torch.load(caminho, map_location="cpu", weights_only=False)
    modelo_bruto(model).load_state_dict(checkpoint["modelo"])
    otimizador.load_state_dict(checkpoint["otimizador"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if "rng" in checkpoint:
        restore_rng_state(checkpoint["rng"])
    return int(checkpoint["passo"])


def caminho_do_checkpoint(cfg: Config) -> Path:
    return cfg.abspath(cfg.training.checkpoint_dir) / f"{cfg.run.name}_ultimo.pt"


# ---------------------------------------------------------------------------
# Amostras de texto
# ---------------------------------------------------------------------------


@torch.no_grad()
def amostrar(
    model: torch.nn.Module,
    tokenizer: TokenizadorUFCG,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[dict]:
    """Gera texto a partir dos prompts fixos do experimento.

    Duas configurações com a mesma loss podem gerar textos bem diferentes; a
    amostra fixa dá evidência qualitativa que a curva não dá. Guardar o texto
    em JSONL, e não no CSV, evita quebra de linha dentro de célula.
    """
    model.eval()
    saidas = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], device=device)
        gerado = idx
        for _ in range(max_new_tokens):
            entrada = gerado[:, -model.cfg.context_length :]
            logits = model(entrada)[:, -1, :]
            if top_k:
                valores, _ = torch.topk(logits, top_k)
                logits = torch.where(
                    logits < valores[:, -1].unsqueeze(-1),
                    torch.full_like(logits, float("-inf")),
                    logits,
                )
            probs = torch.softmax(logits / temperature, dim=-1)
            proximo = torch.multinomial(probs, num_samples=1)
            gerado = torch.cat((gerado, proximo), dim=1)
        saidas.append(
            {
                "prompt": prompt,
                "texto": tokenizer.decode(gerado[0].tolist()),
            }
        )
    model.train()
    return saidas


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------


def treinar(
    cfg: Config,
    max_steps: int | None = None,
    max_horas: float | None = None,
    retomar: bool = True,
    log_every: int = 10,
) -> dict:
    """Executa (ou continua) o treino descrito em `cfg`.

    Devolve um resumo com o passo final, a perda de validação e o caminho do
    checkpoint, que é o que a manhã seguinte precisa saber.
    """
    if max_steps is not None:
        cfg = _com_max_steps(cfg, max_steps)
    limite_horas = max_horas if max_horas is not None else cfg.training.max_hours

    device = escolher_dispositivo(cfg)
    usar_amp = bool(cfg.training.amp and device.type == "cuda")
    set_seed(cfg.run.seed, deterministic=cfg.run.deterministic)

    # -- dados -------------------------------------------------------------
    corpus = localizar_corpus(cfg.run.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    pasta_pre = cfg.abspath(cfg.data.preprocessed_dir)

    ids_treino = preparar("train", corpus, tokenizer, pasta_pre).ids
    ids_val = preparar("val", corpus, tokenizer, pasta_pre).ids

    carregar_lotes = lambda ids, shuffle, drop, semente: criar_dataloader(  # noqa: E731
        ids,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.context_length,
        stride=cfg.data.stride,
        shuffle=shuffle,
        drop_last=drop,
        num_workers=cfg.data.num_workers,
        seed=semente,
    )
    val_loader = carregar_lotes(ids_val, False, False, cfg.run.seed)
    passos_por_epoca = len(
        carregar_lotes(ids_treino, True, cfg.data.drop_last, cfg.run.seed)
    )
    if passos_por_epoca == 0:
        raise RuntimeError("nenhum lote de treino foi formado; confira os dados")

    # -- modelo ------------------------------------------------------------
    model: torch.nn.Module = GPTModel(cfg.model, sdpa=cfg.training.sdpa)
    model.to(device)
    treinaveis = modelo_bruto(model).contar_parametros()
    ativos = modelo_bruto(model).contar_parametros_ativos()

    otimizador = criar_otimizador(model, cfg, cfg.training.fused_adamw)
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    if cfg.training.compile and device.type == "cuda":
        model = torch.compile(model)  # quebra o grafo em kernels fundidos

    # -- retomada ----------------------------------------------------------
    checkpoint = caminho_do_checkpoint(cfg)
    passo_inicial = 0
    if retomar and checkpoint.exists():
        passo_inicial = carregar_checkpoint(checkpoint, model, otimizador, scaler)
        print(f"retomando do checkpoint em {checkpoint} (passo {passo_inicial})")

    # -- log ---------------------------------------------------------------
    logger = CSVLogger(cfg.abspath(cfg.logging.csv_path), COLUNAS)
    caminho_amostras = cfg.abspath(cfg.logging.csv_path).with_name(
        f"{cfg.run.name}_amostras.jsonl"
    )
    caminho_amostras.parent.mkdir(parents=True, exist_ok=True)
    arquivo_amostras = caminho_amostras.open("a", encoding="utf-8")

    medidor = ThroughputMeter(window=20)
    inicio = time.perf_counter()
    passo = passo_inicial
    lr = lr_do_passo(passo, cfg)
    losses_da_janela: list[float] = []
    epoca_atual = passo // passos_por_epoca

    print(
        f"{cfg.run.name}: {descrever_dispositivo(device)} | {treinaveis:,} parâmetros "
        f"({ativos:,} ativos) | {cfg.training.max_steps} passos | "
        f"{cfg.total_tokens:,} tokens | amp={usar_amp} sdpa={cfg.training.sdpa}"
    )

    model.train()
    try:
        while passo < cfg.training.max_steps:
            epoca = passo // passos_por_epoca
            lote_inicial = passo % passos_por_epoca
            if epoca != epoca_atual:
                epoca_atual = epoca
                print(f"-- época {epoca}, a partir do lote {lote_inicial}")

            loader = carregar_lotes(
                ids_treino, True, cfg.data.drop_last, semente_da_epoca(cfg.run.seed, epoca)
            )
            for lote_idx, (x, y) in enumerate(
                islice(loader, lote_inicial, None), start=lote_inicial
            ):
                lr = lr_do_passo(passo, cfg)
                for grupo in otimizador.param_groups:
                    grupo["lr"] = lr

                t0 = time.perf_counter()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                otimizador.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=usar_amp):
                    logits = model(x)
                    perda = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                    aux = modelo_bruto(model).perda_auxiliar()
                    if aux is not None:
                        # O coeficiente da configuração já está aplicado dentro
                        # da camada MoE; somar aqui não pode multiplicar de novo.
                        perda = perda + aux

                scaler.scale(perda).backward()
                scaler.unscale_(otimizador)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                scaler.step(otimizador)
                scaler.update()

                # A GPU é assíncrona: sem sincronizar, o tempo medido seria o de
                # enfileirar o trabalho, não o de executá-lo.
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                medidor.update(cfg.tokens_per_step, dt)

                losses_da_janela.append(perda.item())
                passo += 1

                if passo % log_every == 0 or passo == cfg.training.max_steps:
                    linha = {
                        "evento": "passo",
                        "passo": passo,
                        "epoca": epoca,
                        "loss_treino": sum(losses_da_janela) / len(losses_da_janela),
                        "loss_val": None,
                        "lr": lr,
                        "tokens_por_segundo": medidor.tokens_per_second(),
                        "memoria_pico_mb": _memoria_pico_mb(device),
                        "tempo_s": time.perf_counter() - inicio,
                        "horario": utc_now(),
                    }
                    logger.log(linha)
                    losses_da_janela.clear()
                    print(
                        f"passo {passo}/{cfg.training.max_steps} | "
                        f"loss {linha['loss_treino']:.4f} | lr {lr:.2e} | "
                        f"{linha['tokens_por_segundo']:.0f} tok/s"
                    )

                if cfg.eval_every_steps and passo % cfg.eval_every_steps == 0:
                    loss_val = avaliar(
        model, val_loader, device, usar_amp, cfg.training.eval_max_batches or None
    )
                    logger.log(
                        {
                            "evento": "avaliacao",
                            "passo": passo,
                            "epoca": epoca,
                            "loss_treino": None,  # a média do intervalo já está na linha de passo
                            "loss_val": loss_val,
                            "lr": lr,
                            "tokens_por_segundo": medidor.tokens_per_second(),
                            "memoria_pico_mb": _memoria_pico_mb(device),
                            "tempo_s": time.perf_counter() - inicio,
                            "horario": utc_now(),
                        }
                    )
                    print(f"  validação no passo {passo}: {loss_val:.4f}")

                    for amostra in amostrar(model, tokenizer, cfg.logging.prompts, device):
                        amostra.update({"passo": passo, "experimento": cfg.run.name})
                        arquivo_amostras.write(json.dumps(amostra, ensure_ascii=False) + "\n")
                    arquivo_amostras.flush()

                if cfg.training.checkpoint_every and passo % cfg.training.checkpoint_every == 0:
                    salvar_checkpoint(checkpoint, model, otimizador, scaler, passo, cfg)

                if limite_horas and (time.perf_counter() - inicio) / 3600.0 >= limite_horas:
                    print(f"limite de {limite_horas} h atingido; salvando e saindo")
                    salvar_checkpoint(checkpoint, model, otimizador, scaler, passo, cfg)
                    return _resumo(cfg, passo, None, checkpoint)

                if passo >= cfg.training.max_steps:
                    break
    except KeyboardInterrupt:
        print("interrompido; salvando checkpoint antes de sair")
        salvar_checkpoint(checkpoint, model, otimizador, scaler, passo, cfg)
        return _resumo(cfg, passo, None, checkpoint)
    finally:
        arquivo_amostras.close()

    # -- fecho -------------------------------------------------------------
    loss_val = avaliar(
        model, val_loader, device, usar_amp, cfg.training.eval_max_batches or None
    )
    salvar_checkpoint(checkpoint, model, otimizador, scaler, passo, cfg)
    logger.log(
        {
            "evento": "fim",
            "passo": passo,
            "epoca": passo // passos_por_epoca,
            "loss_treino": None,
            "loss_val": loss_val,
            "lr": lr_do_passo(passo, cfg),
            "tokens_por_segundo": medidor.tokens_per_second(),
            "memoria_pico_mb": _memoria_pico_mb(device),
            "tempo_s": time.perf_counter() - inicio,
            "horario": utc_now(),
        }
    )
    logger.close()

    resumo = _resumo(cfg, passo, loss_val, checkpoint)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return resumo


def _resumo(cfg: Config, passo: int, loss_val: float | None, checkpoint: Path) -> dict:
    return {
        "experimento": cfg.run.name,
        "passos": passo,
        "tokens": passo * cfg.tokens_per_step,
        "loss_val": loss_val,
        "checkpoint": str(checkpoint),
    }


def _memoria_pico_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1e6


def _com_max_steps(cfg: Config, max_steps: int) -> Config:
    """Cópia da configuração com outro orçamento de passos (teste de fumaça)."""
    import dataclasses

    return dataclasses.replace(
        cfg, training=dataclasses.replace(cfg.training, max_steps=max_steps)
    )


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Treina um experimento descrito em YAML.")
    p.add_argument("--config", required=True)
    p.add_argument("--work-dir", default=None, help="sobrescreve run.work_dir")
    p.add_argument("--max-hours", type=float, default=None, help="para antes do limite da sessão")
    p.add_argument("--max-steps", type=int, default=None, help="orçamento menor, para fumaça")
    p.add_argument("--sem-retomar", action="store_true", help="ignora checkpoint existente")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    cfg = load_config(args.config, work_dir=args.work_dir)
    treinar(
        cfg,
        max_steps=args.max_steps,
        max_horas=args.max_hours,
        retomar=not args.sem_retomar,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
