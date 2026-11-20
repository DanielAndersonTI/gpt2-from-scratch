"""Geração de texto a partir de um checkpoint.

Três estratégias de decodificação, na ordem em que o enunciado pede:

- **gulosa**: sempre o token mais provável. Repetitiva e sem surpresa.
- **temperatura**: divide os logits por T. T < 1 concentra a distribuição,
  T > 1 a achata. T = 1 é a distribuição do modelo.
- **top-k e top-p**: cortam a cauda antes de amostrar. Top-k mantém os k
  tokens mais prováveis; top-p (núcleo) mantém o menor conjunto cuja
  probabilidade acumulada passa de p. Top-p se adapta ao contexto: onde o
  modelo está seguro, sobram poucos tokens; onde está incerto, sobram muitos.
"""

from __future__ import annotations

import argparse
import json

import torch

from src.evaluate import carregar_modelo
from src.data import localizar_corpus, TokenizadorUFCG
from src.logging_utils import utc_now
from src.seeds import set_seed


def filtrar_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zera (com -inf) tudo abaixo do k-ésimo maior logit."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    corte = torch.topk(logits, k).values[..., -1:]
    return logits.masked_fill(logits < corte, float("-inf"))


def filtrar_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Mantém o menor conjunto de tokens com probabilidade acumulada >= p.

    A ordenação é feita nos logits; a soma acumulada é feita nas
    probabilidades, porque é a massa de probabilidade que define o núcleo.
    """
    if p >= 1.0:
        return logits
    ordenados, indices = torch.sort(logits, descending=True, dim=-1)
    acumulada = torch.cumsum(torch.softmax(ordenados, dim=-1), dim=-1)
    # Mantém todos os tokens até aquele que faz a soma acumulada passar de p.
    # A subtração da própria probabilidade garante que esse token entre: sem
    # ela, o primeiro token acima do corte seria descartado e o núcleo ficaria
    # com massa menor que p.
    descartar = acumulada - torch.softmax(ordenados, dim=-1) > p
    ordenados = ordenados.masked_fill(descartar, float("-inf"))
    saida = torch.full_like(logits, float("-inf"))
    return saida.scatter(-1, indices, ordenados)


@torch.no_grad()
def gerar(
    model,
    tokenizer: TokenizadorUFCG,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:
    """Continua o prompt e devolve o texto completo, com o prompt no início.

    O modelo é posto em modo de avaliação durante a geração; sem isso o
    dropout continuaria ativo e duas chamadas com a mesma semente dariam
    textos diferentes, inclusive na decodificação gulosa.
    """
    estava_treinando = model.training
    model.eval()
    try:
        return _gerar(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p)
    finally:
        model.train(estava_treinando)


@torch.no_grad()
def _gerar(
    model,
    tokenizer: TokenizadorUFCG,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)

    for _ in range(max_new_tokens):
        entrada = idx[:, -model.cfg.context_length :]
        logits = model(entrada)[:, -1, :]

        logits = filtrar_top_k(logits, top_k)
        logits = filtrar_top_p(logits, top_p)

        if temperature <= 0:  # decodificação gulosa
            proximo = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            proximo = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, proximo), dim=1)

        # Se o modelo emite o separador de documentos, a amostra acabou.
        if int(proximo) == tokenizer.eot_id:
            break

    return tokenizer.decode(idx[0].tolist())


def main() -> None:
    p = argparse.ArgumentParser(description="Gera texto a partir de um checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", action="append", default=None, help="pode repetir")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--saida", default=None, help="arquivo JSONL com as amostras")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed, deterministic=False)
    model, cfg = carregar_modelo(args.checkpoint, torch.device(device))
    corpus = localizar_corpus(cfg.run.work_dir)
    tokenizer = TokenizadorUFCG(corpus.tokenizer)

    prompts = args.prompt or cfg.logging.prompts
    resultados = []
    for prompt in prompts:
        texto = gerar(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        resultados.append({"prompt": prompt, "texto": texto, "horario": utc_now()})
        print(f"\n>>> {prompt}\n{texto}")

    if args.saida:
        with open(args.saida, "a", encoding="utf-8") as f:
            for linha in resultados:
                f.write(json.dumps(linha, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
