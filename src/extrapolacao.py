"""Extrapolação de contexto: a mesma avaliação em 256, 512 e 1024 tokens.

O GPT-2 vanilla não passa do comprimento de treino. O embedding posicional tem
uma linha por posição vista no treino (256), e a posição 300 não tem linha. O
RoPE não tem tabela de posições: gira Q e K por um ângulo calculado na hora, a
partir do índice da posição. Nada no mecanismo impede a posição 1.000, e o
teste `test_funciona_alem_do_comprimento_de_treino` garante isso.

Isso não quer dizer que o modelo treinado em 256 tokens saiba usar janelas
maiores. Ele nunca viu uma. O que o script mede é exatamente isso: a perda em
janelas de 256, 512 e 1024, com o mesmo checkpoint, para separar "o mecanismo
aceita" de "o modelo aproveita".

Duas decisões de medição:

- As janelas não se sobrepõem e têm o tamanho do contexto. Sem isso, os tokens
  de uma linha apareceriam em várias janelas da mesma linha, e o número deixaria
  de ser comparável entre contextos.
- Tokens e bytes avaliados mudam com o contexto, porque as pontas que não
  fecham uma janela inteira ficam de fora. As duas colunas aparecem na saída
  para que o BPB de cada linha seja lido com o próprio denominador.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from src.data import criar_dataloader, localizar_corpus, preparar, TokenizadorUFCG
from src.evaluate import bits_por_byte, bytes_por_token, carregar_modelo
from src.logging_utils import utc_now


def lote_para(contexto: int) -> int:
    """Lote que mantém a memória parecida entre contextos.

    256 tokens por 64 janelas é a mesma matriz de atenção que 1024 por 16, então
    a T4 não muda de regime ao aumentar o contexto.
    """
    return max(1, min(64, 16_384 // contexto))


def avaliar_contexto(
    checkpoint: str | Path,
    contexto: int,
    split: str = "val",
    device: str = "auto",
    batch_size: int | None = None,
    max_lotes: int | None = None,
    work_dir: str | Path | None = None,
) -> dict:
    """Perda e BPB de um checkpoint em janelas de `contexto` tokens."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dispositivo = torch.device(device)

    model, cfg = carregar_modelo(checkpoint, dispositivo)
    if work_dir is not None:
        cfg = replace(cfg, run=replace(cfg.run, work_dir=str(work_dir)))

    treinado_em = cfg.model.context_length
    if contexto > treinado_em and cfg.model.pos_emb != "rope":
        # Não é falha do script: é a limitação que a comparação quer medir.
        raise ValueError(
            f"o modelo aprende posições até {treinado_em} tokens; "
            f"{contexto} exigiria RoPE"
        )

    corpus = localizar_corpus(cfg.run.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)
    ids = preparar(split, corpus, tokenizer, cfg.abspath(cfg.data.preprocessed_dir)).ids

    # Mesma conta de `GPTDatasetTokens`: janelas inteiras, de passo igual ao
    # contexto. O lote não pode ser maior que o número de janelas, senão
    # `drop_last` descarta tudo e sobra lote nenhum.
    janelas = len(range(0, len(ids) - contexto, contexto))
    if janelas == 0:
        raise ValueError(
            f"a partição {split} tem {len(ids)} tokens, insuficiente para uma "
            f"janela de {contexto}"
        )
    tamanho_lote = batch_size or min(lote_para(contexto), janelas)

    loader = criar_dataloader(
        ids,
        batch_size=tamanho_lote,
        max_length=contexto,
        stride=contexto,
        shuffle=False,
        drop_last=True,  # só janelas inteiras, para não misturar contextos
        num_workers=0,
        seed=cfg.run.seed,
    )

    bpb, perda, ppl, n_tokens, n_bytes = bits_por_byte(
        model, loader, dispositivo, bytes_por_token(tokenizer), max_lotes
    )

    return {
        "experimento": cfg.run.name,
        "split": split,
        "contexto": contexto,
        "contexto_de_treino": treinado_em,
        "janelas": janelas,
        "tokens_avaliados": n_tokens,
        "bytes_avaliados": n_bytes,
        "loss": perda,
        "perplexidade": ppl,
        "bpb": bpb,
    }


def comparar(
    checkpoint: str | Path,
    contextos: list[int],
    split: str = "val",
    device: str = "auto",
    max_lotes: int | None = None,
    work_dir: str | Path | None = None,
    saida: str | Path | None = None,
) -> dict:
    """Roda todos os contextos e junta num JSON. O que não cabe vira erro anotado."""
    resultados = []
    for contexto in contextos:
        try:
            resultado = avaliar_contexto(
                checkpoint, contexto, split, device, max_lotes=max_lotes, work_dir=work_dir
            )
        except ValueError as erro:
            resultado = {"contexto": contexto, "erro": str(erro)}
        resultados.append(resultado)
        print(
            f"  {contexto:>5} tokens: "
            + (
                f"loss {resultado['loss']:.4f} | bpb {resultado['bpb']:.4f}"
                if "loss" in resultado
                else resultado["erro"]
            )
        )

    saida_json = {
        "checkpoint": str(checkpoint),
        "split": split,
        "contextos": resultados,
        "horario": utc_now(),
    }
    if saida is not None:
        destino = Path(saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(saida_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return saida_json


def _imprimir(comparacao: dict) -> None:
    print(f"\n{comparacao['checkpoint']} | split {comparacao['split']}")
    print(f"{'contexto':>8} {'janelas':>8} {'tokens':>10} {'bytes':>10} {'loss':>8} {'bpb':>8}")
    for linha in comparacao["contextos"]:
        if "erro" in linha:
            print(f"{linha['contexto']:>8} {'':>8} {'':>10} {'':>10}  {linha['erro']}")
            continue
        print(
            f"{linha['contexto']:>8} {linha['janelas']:>8} "
            f"{linha['tokens_avaliados']:>10} {linha['bytes_avaliados']:>10} "
            f"{linha['loss']:>8.4f} {linha['bpb']:>8.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Mede a perda de um checkpoint com contextos de 128, 256, 512..."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--contextos", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-lotes", type=int, default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--saida", default=None)
    args = p.parse_args()

    comparacao = comparar(
        args.checkpoint,
        args.contextos,
        split=args.split,
        device=args.device,
        max_lotes=args.max_lotes,
        work_dir=args.work_dir,
        saida=args.saida,
    )
    _imprimir(comparacao)


if __name__ == "__main__":
    main()
