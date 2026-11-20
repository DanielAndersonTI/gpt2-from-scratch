"""O GPT-2 do livro, dirigido por configuração.

A arquitetura é a do capítulo 4 de *Build a Large Language Model (From
Scratch)*, com dois pontos de extensão pedidos pelo enunciado:

- `ffn_type` escolhe a camada feed-forward: `dense` usa a do livro, `moe` usa
  oito especialistas com roteador top-2 (implementado em `src/moe.py`);
- `pos_emb` escolhe a posição: `learned` usa o embedding posicional do livro,
  `rope` gira Q e K (implementado em `src/rope.py`).

Fora desses dois pontos, o modelo é o do livro, incluindo detalhes que afetam
a contagem de parâmetros: `out_proj` tem viés sempre, e `qkv_bias` controla
apenas as projeções de query, key e value. É esse conjunto que dá os
11.414.016 parâmetros da configuração base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


# ---------------------------------------------------------------------------
# Blocos básicos, iguais aos do livro
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """LayerNorm com escala e deslocamento aprendidos.

    Normaliza cada token pelas dimensões do embedding, não pelo lote. Por isso
    funciona com lote de qualquer tamanho e é o que o GPT-2 usa.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        media = x.mean(dim=-1, keepdim=True)
        variancia = x.var(dim=-1, keepdim=True, unbiased=False)
        normalizado = (x - media) / torch.sqrt(variancia + self.eps)
        return self.scale * normalizado + self.shift


class GELU(nn.Module):
    """Ativação suave usada no GPT-2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """Feed-forward densa do livro: 256 -> 1024 -> 256, com GELU."""

    def __init__(self, emb_dim: int, drop_rate: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Atenção causal
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Atenção multi-cabeça causal.

    Com `sdpa=True`, a conta é delegada a
    `torch.nn.functional.scaled_dot_product_attention`, que na GPU escolhe o
    kernel do FlashAttention quando as condições permitem. O resultado é o
    mesmo da conta manual; muda o custo de memória, porque o kernel não
    materializa a matriz completa de atenção.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        sdpa: bool = True,
        rope: nn.Module | None = None,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out precisa ser divisível por num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.sdpa = sdpa
        self.rope = rope  # None no modelo com posição aprendida

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Viés sempre presente, como no livro: qkv_bias controla só q, k, v.
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Máscara triangular superior gravada uma vez, no dispositivo do módulo.
        mascara = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mascara, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_tokens, _ = x.shape

        # (b, n, d) -> (b, n, cabecas, dim_por_cabeca) -> (b, cabecas, n, dim)
        q = self.W_query(x).view(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE: a posição entra girando Q e K, depois da projeção e antes do
        # produto interno. V não gira.
        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.sdpa:
            # is_causal=True aplica a máscara triangular dentro do kernel.
            contexto = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            pontuacoes = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            mascara = self.mask[:n_tokens, :n_tokens]
            pontuacoes = pontuacoes.masked_fill(mascara, float("-inf"))
            pesos = torch.softmax(pontuacoes, dim=-1)
            pesos = self.dropout(pesos)
            contexto = pesos @ v

        contexto = contexto.transpose(1, 2).contiguous().view(b, n_tokens, self.d_out)
        return self.out_proj(contexto)


# ---------------------------------------------------------------------------
# Bloco do transformer
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Atenção, soma residual, feed-forward, soma residual.

    A norma fica antes de cada subcamada (pre-norm), como no GPT-2. O
    feed-forward é a peça que muda entre o baseline e o MoE; o resto do bloco é
    idêntico nas quatro configurações.
    """

    def __init__(self, cfg: ModelConfig, sdpa: bool = True) -> None:
        super().__init__()
        # RoPE só é criada quando a configuração pede. A importação é tardia
        # para que os experimentos com posição aprendida não dependam dela.
        if cfg.pos_emb == "rope":
            from src.rope import RoPE

            rope = RoPE(cfg.emb_dim // cfg.n_heads, cfg.context_length, cfg.rope_theta)
        else:
            rope = None

        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            num_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
            sdpa=sdpa,
            rope=rope,
        )
        self.ff = criar_ffn(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atencao = self.att(self.norm1(x))
        x = x + self.drop_shortcut(atencao)
        feed_forward = self.ff(self.norm2(x))
        return x + self.drop_shortcut(feed_forward)


def criar_ffn(cfg: ModelConfig) -> nn.Module:
    """Escolhe a camada feed-forward conforme `cfg.ffn_type`.

    A importação do MoE é feita aqui dentro, e não no topo do arquivo, para que
    um experimento denso não carregue o código do MoE. Se o módulo ainda não
    existir, o erro diz exatamente o que falta.
    """
    if cfg.ffn_type == "dense":
        return FeedForward(cfg.emb_dim, cfg.drop_rate)
    if cfg.ffn_type == "moe":
        from src.moe import MoEFeedForward  # importação tardia, ver acima

        return MoEFeedForward(cfg.emb_dim, cfg.drop_rate, cfg.moe)
    raise ValueError(f"ffn_type desconhecido: {cfg.ffn_type!r}")


# ---------------------------------------------------------------------------
# O modelo
# ---------------------------------------------------------------------------


class GPTModel(nn.Module):
    """GPT-2 decoder-only: embedding de token, posição, N blocos, cabeça.

    A posição pode ser aprendida (`pos_emb = learned`) ou rotacional
    (`pos_emb = rope`). No segundo caso o embedding posicional deixa de existir
    e a posição entra girando Q e K dentro da atenção.
    """

    def __init__(self, cfg: ModelConfig, sdpa: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = (
            nn.Embedding(cfg.context_length, cfg.emb_dim)
            if cfg.pos_emb == "learned"
            else None
        )
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg, sdpa=sdpa) for _ in range(cfg.n_layers)]
        )
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # Inicialização explícita, igual à do livro, para que a semente controle
        # todos os pesos e a loss inicial seja reproduzível. Vem antes do
        # compartilhamento de pesos: depois dele, reinit na cabeça mexeria no
        # embedding de token, que é o mesmo tensor.
        self.apply(self._init_weights)

        if cfg.weight_tying:
            # Compartilha a matriz de embedding com a cabeça de saída: economiza
            # vocab x emb_dim parâmetros e costuma ajudar em vocabulário grande.
            self.out_head.weight = self.tok_emb.weight

    def _init_weights(self, modulo: nn.Module) -> None:
        if isinstance(modulo, nn.Linear):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                torch.nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, nn.Embedding):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        x = self.tok_emb(idx)

        if self.pos_emb is not None:
            posicoes = torch.arange(seq_len, device=idx.device)
            x = x + self.pos_emb(posicoes)
        # Com RoPE não há embedding posicional: a posição é aplicada dentro da
        # atenção, girando Q e K (ver src/rope.py).

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

    # -- utilidades para o treino ---------------------------------------------

    def perda_auxiliar(self) -> torch.Tensor | None:
        """Soma das perdas de balanceamento dos blocos MoE, ou None no denso.

        O roteador precisa de um empurrão para não mandar todos os tokens ao
        mesmo especialista. Essa perda é somada à perda de linguagem no passo
        de treino; aqui ela só é coletada.
        """
        perdas = [
            bloco.ff.aux_loss
            for bloco in self.trf_blocks
            if getattr(bloco.ff, "aux_loss", None) is not None
        ]
        return torch.stack(perdas).sum() if perdas else None

    def contar_parametros(self) -> int:
        """Total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def contar_parametros_ativos(self) -> int:
        """Parâmetros usados por token. Igual ao total no modelo denso.

        No MoE, os especialistas não escolhidos não são consultados naquele
        token, então o custo por token é o do roteador mais top-k especialistas.
        """
        total = self.contar_parametros()
        for bloco in self.trf_blocks:
            if hasattr(bloco.ff, "parametros_totais") and hasattr(bloco.ff, "parametros_ativos"):
                total -= bloco.ff.parametros_totais() - bloco.ff.parametros_ativos()
        return total


# ---------------------------------------------------------------------------
# Perda e geração
# ---------------------------------------------------------------------------


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Entropia cruzada da previsão do próximo token, média por token.

    `flatten(0, 1)` junta lote e posição: cada token de cada sequência é um
    exemplo independente de classificação entre 16.000 tokens.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


@torch.no_grad()
def calc_loss_loader(
    data_loader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Perda média na validação. `num_batches` limita a amostra."""
    model.eval()
    total = 0.0
    n = 0
    for x, y in data_loader:
        if num_batches is not None and n >= num_batches:
            break
        total += calc_loss_batch(x, y, model, device).item()
        n += 1
    model.train()
    return total / n if n else float("nan")


def gerar_texto_simples(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Decodificação gulosa, com temperatura e top-k opcionais.

    Versão mínima, para conferir o caminho de inferência (T16). A versão de
    entrega, com top-p e os prompts do relatório, fica em `src/generate.py`.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]

        if top_k is not None:
            valores, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < valores[:, -1].unsqueeze(-1),
                                 torch.full_like(logits, float("-inf")), logits)

        if temperature and temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def descrever_parametros(model: GPTModel) -> dict[str, int]:
    """Contagem por bloco, para conferir a tabela do artigo."""
    detalhe = {
        "tok_emb": model.tok_emb.weight.numel(),
        "pos_emb": model.pos_emb.weight.numel() if model.pos_emb is not None else 0,
        "out_head": model.out_head.weight.numel(),
        "final_norm": sum(p.numel() for p in model.final_norm.parameters()),
        "blocos": sum(p.numel() for p in model.trf_blocks.parameters()),
    }
    detalhe["total"] = model.contar_parametros()
    return detalhe


@dataclass(frozen=True)
class ResumoModelo:
    """Números do modelo, para o log e para o relatório."""

    total: int
    ativos: int
    por_bloco: int
