"""Avaliação final: perda, perplexidade e bits por byte.

A perda em nats por token não é comparável entre modelos que usam tokenizadores
diferentes, porque um token pode cobrir mais ou menos texto. Bits por byte
resolve isso: divide a perda total pelo tamanho do texto em bytes.

    BPB = soma das perdas (nats) / (ln 2 x número de bytes)

A contagem de bytes segue a mesma regra do notebook do professor: cada token do
vocabulário é decodificado e medido em UTF-8, e o `<|endoftext|>` conta como um
byte, porque substitui a quebra de linha entre documentos. O teste
`test_bits_por_byte_igual_ao_notebook` confere a igualdade com a função dele.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import config_from_dict
from src.data import criar_dataloader, localizar_corpus, preparar, TokenizadorUFCG
from src.logging_utils import utc_now
from src.model import GPTModel


def bytes_por_token(tokenizer: TokenizadorUFCG) -> torch.Tensor:
    """Bytes UTF-8 de cada id do vocabulário.

    Um token decodificado pode virar mais de um caractere e cada caractere pode
    ocupar mais de um byte, então o tamanho é medido no texto decodificado, não
    no id.
    """
    tamanhos = [len(tokenizer.decode([i]).encode("utf-8")) for i in range(tokenizer.n_vocab)]
    tamanhos[tokenizer.eot_id] = 1  # o separador substitui a quebra de linha
    return torch.tensor(tamanhos, dtype=torch.int64)


@torch.no_grad()
def bits_por_byte(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    tabela_bytes: torch.Tensor,
    max_lotes: int | None = None,
    usar_amp: bool = False,
) -> tuple[float, float, float, int, int]:
    """Devolve (bpb, perda, perplexidade, tokens avaliados, bytes avaliados).

    A perda é somada com `reduction="sum"`, porque o denominador do BPB é a
    soma de bytes, não a média por lote. Os últimos tokens que não completam
    uma janela ficam de fora, e os bytes são contados só para os alvos
    avaliados.
    """
    model.eval()
    soma_loss, soma_bytes, n_tokens = 0.0, 0, 0

    for k, (x, y) in enumerate(loader):
        if max_lotes is not None and k >= max_lotes:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=usar_amp):
            logits = model(x)
            perda = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
        soma_loss += perda.item()
        soma_bytes += int(tabela_bytes[y].sum().item())
        n_tokens += y.numel()

    if n_tokens == 0:
        raise ValueError("nenhum token avaliado")

    perda_media = soma_loss / n_tokens
    return (
        soma_loss / (math.log(2) * soma_bytes),
        perda_media,
        math.exp(perda_media),
        n_tokens,
        soma_bytes,
    )


def carregar_modelo(caminho: str | Path, device: torch.device) -> tuple[torch.nn.Module, object]:
    """Reconstrói o modelo a partir do checkpoint, sem precisar do YAML.

    A configuração inteira viaja dentro do checkpoint. É o que permite avaliar
    um modelo treinado semanas antes sem lembrar em que arquivo ele foi
    definido.
    """
    checkpoint = torch.load(caminho, map_location="cpu", weights_only=False)
    cfg = config_from_dict(checkpoint["config"])
    model = GPTModel(cfg.model, sdpa=cfg.training.sdpa)
    model.load_state_dict(checkpoint["modelo"])
    model.to(device)
    model.eval()
    return model, cfg


def avaliar_checkpoint(
    caminho_checkpoint: str | Path,
    split: str = "val",
    device: str = "auto",
    max_lotes: int | None = None,
    saida: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> dict:
    """Calcula os números finais e, se pedido, grava um JSON.

    `work_dir` tem precedência sobre o do checkpoint. É o que permite avaliar no
    computador local um modelo treinado no Drive do Colab, que grava o caminho
    `/content/drive/...` dentro do checkpoint.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dispositivo = torch.device(device)

    model, cfg = carregar_modelo(caminho_checkpoint, dispositivo)
    if work_dir is not None:
        cfg = replace(cfg, run=replace(cfg.run, work_dir=str(work_dir)))
    corpus = localizar_corpus(cfg.run.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    ids = preparar(split, corpus, tokenizer, cfg.abspath(cfg.data.preprocessed_dir)).ids

    loader = criar_dataloader(
        ids,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.context_length,
        stride=cfg.data.stride,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        seed=cfg.run.seed,
    )

    bpb, perda, ppl, n_tokens, n_bytes = bits_por_byte(
        model, loader, dispositivo, bytes_por_token(tokenizer), max_lotes
    )

    resultado = {
        "experimento": cfg.run.name,
        "split": split,
        "checkpoint": str(caminho_checkpoint),
        "passos_treinados": int(torch.load(caminho_checkpoint, map_location="cpu", weights_only=False)["passo"]),
        "parametros": model.contar_parametros(),
        "parametros_ativos": model.contar_parametros_ativos(),
        "loss": perda,
        "perplexidade": ppl,
        "bpb": bpb,
        "tokens_avaliados": n_tokens,
        "bytes_avaliados": n_bytes,
        "horario": utc_now(),
    }

    if saida is not None:
        destino = Path(saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado


def main() -> None:
    p = argparse.ArgumentParser(description="Avalia um checkpoint: loss, perplexidade e BPB.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-lotes", type=int, default=None)
    p.add_argument(
        "--work-dir",
        default=None,
        help="sobrepõe o diretório de trabalho gravado no checkpoint",
    )
    p.add_argument("--saida", default=None, help="caminho do JSON de resultado")
    args = p.parse_args()

    avaliar_checkpoint(
        args.checkpoint,
        split=args.split,
        device=args.device,
        max_lotes=args.max_lotes,
        saida=args.saida,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    main()
