"""Figuras e tabela do roteamento do MoE (T36).

Lê os JSONs de `src/analise_moe.py` e produz o material do artigo:

- `figuras/moe_utilizacao.png`: mapa de calor da fatia de tráfego que cada
  especialista recebe, por bloco, com o valor justo (1/8) no centro da escala.
- `figuras/moe_entropia.png`: entropia normalizada do roteador por bloco.
- `figuras/moe_especializacao.png`: como os tokens mais frequentes se
  repartem entre os especialistas no primeiro bloco.
- `tabelas/roteamento.tex`: tabela booktabs resumindo as duas variantes.

A escala do mapa de utilização é divergente e centrada em 12,5%, para que o
leitor veja de imediato quem recebe mais do que o valor justo, e não só a cor
uniforme de uma escala absoluta.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Paleta categórica validada (azul e laranja) e escala divergente azul-cinza-vermelho.
AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#f0efec"
VERMELHO = "#d03b3b"
TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"
SUPERFICIE = "#ffffff"

DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_cinza_vermelho", ["#104281", "#3987e5", CINZA, "#e66767", "#8f2323"]
)
SEQUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

VARIANTES = [
    ("e2_moe_h512", "E2 · 8 especialistas, h = 512", AZUL),
    ("e3_moe_h128", "E3 · 8 especialistas, h = 128", LARANJA),
]


def _carregar(logs: Path) -> dict[str, dict]:
    dados = {}
    for chave, _, _ in VARIANTES:
        caminho = logs / f"{chave}_roteamento.json"
        dados[chave] = json.loads(caminho.read_text(encoding="utf-8"))
    return dados


def _eixo_limpo(ax) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c8c2")
    ax.tick_params(colors=TEXTO_SEC, labelsize=9)


def _texto_celula(valor: float, vmin: float, vmax: float) -> str:
    """Preto em célula clara, branco em célula escura."""
    meio = (vmin + vmax) / 2
    distancia = abs(valor - meio) / max(abs(vmax - meio), 1e-9)
    return "white" if distancia > 0.55 else TEXTO


def figura_utilizacao(dados: dict[str, dict], destino: Path) -> None:
    fig, eixos = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    justo = 1 / 8
    vmin, vmax = justo - 0.04, justo + 0.04

    for ax, (chave, rotulo, _) in zip(eixos, VARIANTES):
        tabela = dados[chave]["utilizacao_top1"]
        blocos = list(tabela)
        matriz = np.array([tabela[b] for b in blocos])
        imagem = ax.imshow(matriz, cmap=DIVERGENTE, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                ax.text(
                    j, i, f"{matriz[i, j]:.3f}",
                    ha="center", va="center", fontsize=7.5,
                    color=_texto_celula(matriz[i, j], vmin, vmax),
                )
        ax.set_yticks(range(len(blocos)))
        ax.set_yticklabels([b.replace("bloco_", "bloco ") for b in blocos])
        ax.set_title(rotulo, loc="left", fontsize=10, color=TEXTO, pad=6)
        _eixo_limpo(ax)
        ax.set_xticks([])

    eixos[-1].set_xticks(range(8))
    eixos[-1].set_xticklabels([f"e{i}" for i in range(8)], color=TEXTO_SEC, fontsize=9)
    eixos[-1].set_xlabel("especialista", color=TEXTO_SEC, fontsize=9)
    fig.suptitle(
        "Fatia do tráfego top-1 por especialista e por bloco (validação)",
        x=0.02, ha="left", fontsize=11, color=TEXTO,
    )
    barra = fig.colorbar(imagem, ax=eixos, fraction=0.025, pad=0.02)
    barra.set_label("fatia, com 12,5% no branco", color=TEXTO_SEC, fontsize=8)
    barra.ax.tick_params(colors=TEXTO_SEC, labelsize=8)
    barra.outline.set_visible(False)
    fig.savefig(destino, dpi=200, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)


def figura_entropia(dados: dict[str, dict], destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for chave, rotulo, cor in VARIANTES:
        tabela = dados[chave]["entropia_normalizada"]
        valores = list(tabela.values())
        x = np.arange(len(valores))
        ax.plot(
            x, valores, color=cor, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SUPERFICIE, markeredgewidth=2, label=rotulo,
        )
        ax.annotate(
            f"{valores[-1]:.2f}", (x[-1], valores[-1]), textcoords="offset points",
            xytext=(8, 7 if cor == AZUL else -11), va="center", fontsize=9,
            color=TEXTO_SEC,
        )

    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(range(len(dados["e2_moe_h512"]["entropia_normalizada"])))
    ax.set_xticklabels(["bloco 0", "bloco 1", "bloco 2", "bloco 3"], color=TEXTO_SEC)
    ax.set_ylabel("entropia normalizada", color=TEXTO_SEC, fontsize=9)
    ax.grid(axis="y", color="#e4e3de", linewidth=0.8)
    ax.set_axisbelow(True)
    _eixo_limpo(ax)
    ax.set_title(
        "Entropia do roteador por bloco: 1 seria indiferença total, 0 seria colapso",
        loc="left", fontsize=10, color=TEXTO,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=TEXTO_SEC)
    fig.savefig(destino, dpi=200, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)


def _rotulo(token: str) -> str:
    """Rótulo legível do token na figura.

    A quebra de linha não se lê como texto no eixo, então vira a palavra
    `quebra`; o espaço à esquerda dos outros tokens some, para alinhar os
    rótulos. Nenhum token do top-10 é só espaço nem contém outra quebra.
    """
    if token == "\\n":
        return "quebra"
    return token.strip() or "espaço"


def figura_especializacao(dados: dict[str, dict], destino: Path, n_tokens: int = 10) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, (chave, rotulo, _) in zip(eixos, VARIANTES):
        entradas = dados[chave]["tokens_frequentes"][:n_tokens]
        matriz = np.array([e["por_bloco"]["bloco_0"] for e in entradas])
        # Cada linha volta a somar 1 dentro do bloco, para comparar a
        # repartição entre tokens e não o volume de cada token.
        matriz = matriz / matriz.sum(axis=1, keepdims=True)
        imagem = ax.imshow(matriz, cmap=SEQUENCIAL, vmin=0.0, vmax=0.35, aspect="auto")
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                ax.text(
                    j, i, f"{matriz[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if matriz[i, j] > 0.22 else TEXTO,
                )
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"e{i}" for i in range(8)], color=TEXTO_SEC, fontsize=8)
        ax.set_xlabel("especialista", color=TEXTO_SEC, fontsize=9)
        ax.set_title(rotulo, loc="left", fontsize=9.5, color=TEXTO, pad=6)
        _eixo_limpo(ax)
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(entradas), 1), minor=True)
        ax.grid(which="minor", color=SUPERFICIE, linewidth=1.6)
        ax.tick_params(which="minor", length=0)

    eixos[0].set_yticks(range(n_tokens))
    eixos[0].set_yticklabels(
        [_rotulo(e["token"]) for e in dados["e2_moe_h512"]["tokens_frequentes"][:n_tokens]],
        fontsize=8, color=TEXTO_SEC,
    )
    fig.suptitle(
        "Repartição dos 10 tokens mais frequentes no bloco 0 (linhas normalizadas)",
        x=0.02, ha="left", fontsize=10.5, color=TEXTO,
    )
    barra = fig.colorbar(imagem, ax=eixos, fraction=0.03, pad=0.02)
    barra.set_label("fatia do token dentro do bloco", color=TEXTO_SEC, fontsize=8)
    barra.ax.tick_params(colors=TEXTO_SEC, labelsize=8)
    barra.outline.set_visible(False)
    fig.savefig(destino, dpi=200, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)


def _num(valor: float) -> str:
    """Número com três casas e vírgula decimal, como no texto do artigo."""
    return f"{valor:.3f}".replace(".", ",")


def tabela_latex(dados: dict[str, dict], destino: Path) -> None:
    """Tabela booktabs com o resumo numérico das duas variantes."""
    linhas = []
    for chave, rotulo, _ in VARIANTES:
        d = dados[chave]
        fracoes = np.array([v for v in d["utilizacao_top1"].values()])
        entropia_media = sum(d["entropia_normalizada"].values()) / len(d["entropia_normalizada"])
        dominantes = {
            centro: info["especialista_dominante"]
            for centro, info in d["por_centro"]["bloco_3"].items()
        }
        unicos = len(set(dominantes.values()))
        linhas.append(
            (rotulo.split(" · ")[0], fracoes.min(), fracoes.max(), entropia_media, unicos)
        )

    corpo = "\n".join(
        f"{nome} & {_num(minimo)} & {_num(maximo)} & {_num(entropia)} & {centros} \\\\"
        for nome, minimo, maximo, entropia, centros in linhas
    )
    destino.write_text(
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Equilíbrio do roteador nas duas variantes de MoE, validação. "
        "A fatia é a fração dos despachos top-1 por especialista, sobre todos os "
        "blocos; o valor justo é 0,125. A última coluna conta quantos "
        "especialistas distintos dominam algum \\texttt{sigla\\_centro} no último bloco.}\n"
        "\\label{tab:roteamento}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Variante & Fatia mínima & Fatia máxima & Entropia média & Centros com dominante próprio \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def main() -> None:
    raiz = Path(__file__).resolve().parent.parent
    logs = raiz / "logs"
    figuras = raiz / "figuras"
    tabelas = raiz / "tabelas"
    figuras.mkdir(exist_ok=True)
    tabelas.mkdir(exist_ok=True)

    dados = _carregar(logs)
    figura_utilizacao(dados, figuras / "moe_utilizacao.png")
    figura_entropia(dados, figuras / "moe_entropia.png")
    figura_especializacao(dados, figuras / "moe_especializacao.png")
    tabela_latex(dados, tabelas / "roteamento.tex")
    print("figuras em", figuras)
    print("tabela em", tabelas / "roteamento.tex")


if __name__ == "__main__":
    main()
