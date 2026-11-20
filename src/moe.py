"""Camada feed-forward de Mixture of Experts, com roteador top-k.

A ideia em uma frase: em vez de uma única rede feed-forward por bloco, existem
E especialistas, e cada token consulta apenas os k mais indicados pelo
roteador. O custo por token fica próximo ao de um especialista, enquanto a
capacidade total do modelo cresce com E.

Este arquivo é o F6 do plano. O ponto de entrada é `MoEFeedForward`, que ocupa
o lugar da `FeedForward` densa dentro do bloco do transformer; a interface é a
mesma, `(lote, tokens, emb) -> (lote, tokens, emb)`, mais uma perda auxiliar
que o treino coleta.

Três detalhes decidem se a camada funciona:

1. **Normalização dos pesos do top-k.** As probabilidades dos k escolhidos são
   renormalizadas para somar 1. Sem isso, a escala da saída depende de quão
   confiante o roteador está, e o treino fica instável.
2. **Perda auxiliar de balanceamento.** Sem ela, o roteador colapsa: os tokens
   migram para um ou dois especialistas ainda no começo e os demais nunca
   recebem gradiente. A perda de *switch* pune a concentração.
3. **Despacho por token.** Cada especialista recebe só a fatia de tokens que o
   escolheu. É essa esparsidade que dá a economia; materializar a saída de
   todos os especialistas para todos os tokens anularia o ganho.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import MoEConfig
from src.model import GELU

__all__ = ["Expert", "Router", "MoEFeedForward"]


class Expert(nn.Module):
    """Uma MLP de duas camadas: emb -> hidden -> emb, com GELU.

    A largura `hidden_dim` é o que separa as duas variantes do experimento:
    512 deixa cada especialista maior que a MLP densa e 128 deixa bem menor.
    """

    def __init__(self, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Router(nn.Module):
    """Porteiro: um linear que dá uma pontuação a cada especialista.

    Não tem camada escondida nem ativação, e é de propósito: o roteador precisa
    ser barato, porque roda para todos os tokens. A escolha vem depois, no
    `MoEFeedForward`, por `topk` sobre as probabilidades.
    """

    def __init__(self, emb_dim: int, n_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(emb_dim, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class MoEFeedForward(nn.Module):
    """Feed-forward esparsa: roteador top-k sobre E especialistas.

    A saída de um token é a soma ponderada das saídas dos seus k especialistas,
    com pesos renormalizados. Os especialistas não escolhidos não veem o token,
    nem no passo direto nem no gradiente.

    A perda auxiliar fica em `self.aux_loss` depois de cada passo direto, já
    multiplicada pelo coeficiente da configuração. Ela é None em modo de
    avaliação, onde não há gradiente a corrigir e o valor só atrapalharia o
    log.
    """

    def __init__(self, emb_dim: int, drop_rate: float, cfg: MoEConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.aux_loss_coef = cfg.aux_loss_coef

        self.router = Router(emb_dim, cfg.n_experts)
        self.experts = nn.ModuleList(
            [Expert(emb_dim, cfg.hidden_dim) for _ in range(cfg.n_experts)]
        )
        self.dropout = nn.Dropout(drop_rate)

        # Preenchidos no passo direto e lidos pela análise (T34 e T35). Não são
        # parâmetros nem buffers: mudam a cada lote e não entram no checkpoint.
        self.aux_loss: torch.Tensor | None = None
        self.utilizacao: torch.Tensor | None = None

    # -- passo direto ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_tokens, d = x.shape
        tokens = x.reshape(b * n_tokens, d)

        # 1. Roteamento. As probabilidades somam 1 em cada token; o top-k
        #    escolhe quem trabalha naquele token.
        logits = self.router(tokens)                      # (b*n, E)
        probs = torch.softmax(logits, dim=-1)             # (b*n, E)
        pesos, indices = probs.topk(self.top_k, dim=-1)   # (b*n, k) cada

        # 2. Renormalização: os k pesos voltam a somar 1. Aqui é float32 mesmo
        #    em precisão mista, porque a divisão por um somatório pequeno
        #    amplifica erro numérico em fp16.
        pesos = pesos / pesos.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # 3. Despacho. Para cada especialista, separa-se só a fatia de tokens
        #    que o escolheu. `posicao` é o token, `k-esimo` diz qual dos k
        #    lugares ele ocupa, o que dá o peso certo quando o token consulta
        #    dois especialistas.
        saida = torch.zeros_like(tokens)
        for e, expert in enumerate(self.experts):
            posicao, k_esimo = (indices == e).nonzero(as_tuple=True)
            if posicao.numel() == 0:
                # Especialista ocioso neste lote: não há gradiente para ele
                # neste passo. É exatamente o caso que a perda auxiliar pune.
                continue
            contribuicao = expert(tokens[posicao]) * pesos[posicao, k_esimo].unsqueeze(-1)
            # index_add_ acumula quando o mesmo token aparece para dois
            # especialistas, que é o caso normal do top-2.
            saida.index_add_(0, posicao, contribuicao)

        # 4. Perda auxiliar e contagens para a análise. Sem gradiente: são
        #    diagnóstico, não parte do objetivo. A contagem inclui os k
        #    despachos de cada token, não só o primeiro, porque é essa a carga
        #    que cada especialista realmente executa.
        with torch.no_grad():
            self.utilizacao = torch.bincount(
                indices.reshape(-1), minlength=self.n_experts
            ).float()
        if self.training:
            self.aux_loss = self.aux_loss_coef * self._perda_de_balanceamento(probs, indices)
        else:
            self.aux_loss = None

        # Guarda o roteamento do último lote para a análise (T34 e T35). Sem
        # gradiente e sem histórico: só o último lote fica na memória.
        self.ultimo_roteamento = (probs.detach(), indices.detach())

        return self.dropout(saida).reshape(b, n_tokens, d)

    def _perda_de_balanceamento(
        self, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Perda de *switch*: E vezes o produto entre carga e probabilidade.

        Dois termos, ambos médios sobre os tokens:

        - `carga`: quanto do lote cada especialista de fato recebeu (pelo
          primeiro dos k escolhidos);
        - `probabilidade`: quanto o roteador, em média, deu a cada especialista.

        Antes do deslocamento, todos os tokens no especialista 0 dão E; carga
        perfeitamente uniforme dá 1. O gradiente empurra o roteador a
        espalhar sem obrigar a uniformidade exata, porque especializar um
        pouco é legítimo.

        A subtração de 1 é só deslocamento, não muda o gradiente, e deixa a
        perda valendo zero no balanceamento perfeito. Por Cauchy-Schwarz, a
        soma entre carga e probabilidade nunca cai abaixo de 1/E, então o
        valor nunca fica negativo.
        """
        carga = F.one_hot(indices[:, 0], self.n_experts).float().mean(dim=0)
        probabilidade = probs.mean(dim=0)
        return self.n_experts * torch.sum(carga * probabilidade) - 1.0

    # -- contagem de parâmetros ----------------------------------------------

    def parametros_totais(self) -> int:
        """Parâmetros da camada, incluindo todos os especialistas."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parametros_ativos(self) -> int:
        """Parâmetros consultados por token: roteador mais top-k especialistas.

        É a conta que interessa para o custo computacional. Os especialistas
        fora do top-k existem, ocupam memória e entram na contagem total, mas
        não são executados para aquele token.
        """
        por_expert = sum(p.numel() for p in self.experts[0].parameters())
        roteador = sum(p.numel() for p in self.router.parameters())
        return roteador + self.top_k * por_expert

    def resumo(self) -> dict[str, int]:
        """Números da camada, para o relatório."""
        return {
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "parametros_totais": self.parametros_totais(),
            "parametros_ativos": self.parametros_ativos(),
            "por_expert": sum(p.numel() for p in self.experts[0].parameters()),
        }
