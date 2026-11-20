"""RoPE: codificação de posição por rotação, aplicada a Q e K.

No GPT-2 do livro, a posição entra somada ao embedding do token, como um vetor
aprendido por posição. RoPE faz diferente: não soma nada em lugar nenhum.
Depois de projetar Q e K, gira cada par de dimensões por um ângulo
proporcional à posição do token. O produto interno entre a consulta da posição
`m` e a chave da posição `n` passa a depender só de `m - n`, e não das posições
absolutas. Isso é o que se espera de uma atenção que só precisa saber a
distância entre dois tokens.

A rotação preserva a norma do vetor, porque é uma matriz ortogonal: nenhuma
informação de magnitude se perde, só a fase muda.

Este arquivo é o F7 do plano. O ponto de entrada é `RoPE`, que a atenção usa
recebendo `(q, k)` e devolvendo os dois já girados, com as mesmas formas.

Convenção adotada: metades. O vetor de dimensão `d` é cortado em duas partes
de `d/2`, e a rotação usa o par (primeira metade, segunda metade). A outra
convenção comum, aos pares vizinhos, gira `(x0, x1)`, `(x2, x3)` e assim por
diante; as duas são válidas e não se misturam. Como o resultado é aprendido a
partir do zero, qualquer uma serve, desde que a tabela de frequências combine
com o pareamento usado na rotação.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RoPE"]


class RoPE(nn.Module):
    """Gira Q e K por posição, com frequências geométricas.

    `theta` é a base das frequências. O valor de 10.000 é o do RoPE original;
    bases maiores alargam o período e são a receita usual para estender o
    contexto além do comprimento de treino.

    O módulo não guarda parâmetros treináveis. Ele guarda apenas o vetor de
    frequências inversas, que é constante e por isso não entra no checkpoint.
    """

    def __init__(self, head_dim: int, context_length: int, theta: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "RoPE exige dimensão de cabeça par"

        self.head_dim = head_dim
        self.context_length = context_length

        # Uma frequência por par de dimensões. A progressão é geométrica: os
        # primeiros pares giram rápido e codificam a posição local, os últimos
        # giram devagar e codificam a posição global. Com d = 64, a primeira
        # frequência dá uma volta completa a cada 6 tokens e a última leva
        # dezenas de milhares.
        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (indices / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # -- tabelas --------------------------------------------------------------

    def tabelas(self, n_tokens: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Cossenos e senos para as `n` primeiras posições.

        São recalculados a cada passo direto, e não congelados no
        construtor. A razão é T39: avaliar em 512 ou 1024 tokens exige
        tabelas mais longas do que as do treino, e o RoPE não tem limite
        estrutural para isso. O custo é desprezível, porque a tabela tem
        `n x d/2` números.
        """
        posicoes = torch.arange(n_tokens, device=device, dtype=torch.float32)
        angulos = torch.outer(posicoes, self.inv_freq.to(device))  # (n, d/2)
        return angulos.cos().to(dtype), angulos.sin().to(dtype)

    @staticmethod
    def _meio_girado(x: torch.Tensor) -> torch.Tensor:
        """(-x2, x1): a segunda metade negada, na frente da primeira."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # -- passo direto ---------------------------------------------------------

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gira `q` e `k`, ambos com forma (lote, cabeças, tokens, d_cabeça).

        A rotação é aplicada no mesmo ângulo por cabeça: todas as cabeças usam
        as mesmas frequências, e o que diferencia uma cabeça da outra são as
        projeções aprendidas, não a codificação de posição.
        """
        n_tokens = q.shape[-2]
        cos, sin = self.tabelas(n_tokens, q.device, q.dtype)

        # A tabela tem uma coluna por par; a rotação precisa de uma por
        # dimensão, então o mesmo ângulo é repetido nas duas metades.
        cos = torch.cat((cos, cos), dim=-1).view(1, 1, n_tokens, self.head_dim)
        sin = torch.cat((sin, sin), dim=-1).view(1, 1, n_tokens, self.head_dim)

        # Rotação em forma complexa: x' = x cos + rot(x) sin.
        return (
            q * cos + self._meio_girado(q) * sin,
            k * cos + self._meio_girado(k) * sin,
        )
