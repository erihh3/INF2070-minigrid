"""Gráficos de acompanhamento do treino offline (BC, DQN offline, CQL, IQL).

Uso:  python graficos.py                                   (lê resultados_offline_rl.json)
      python graficos.py --entrada outro.json --pasta figs
Gera, na pasta escolhida:
  1_curvas_aprendizado.png  taxa de sucesso ao longo do treino
  2_perdas.png              perdas de treino de cada método
  3_valores_q.png           Q estimado: melhor ação x ações que o especialista nunca usa
  4_concordancia.png        % de estados do especialista em que a política repete a ação dele
  5_acoes.png               ações escolhidas na avaliação final x ações presentes no dataset
  6_sucesso_final.png       comparação final (barras)
"""
import argparse, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CENARIOS = ["expert", "random", "misto"]
TITULO = {"expert": "Especialista", "random": "Aleatório", "misto": "Misto"}
METODOS = ["BC", "DQN offline", "CQL", "IQL"]
COR = {"BC": "#2a78d6", "DQN offline": "#eb6834", "CQL": "#1baf7a", "IQL": "#eda100"}
TINTA, CINZA = "#52514e", "#8a8984"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10, "lines.linewidth": 1.8})


def _logs(df, d, m, chave):
    """Junta a métrica `chave` de todas as sementes: retorna passos, média, mínimo e máximo."""
    series = []
    for log in df[(df.dataset == d) & (df.algo == m)]["log"]:
        pts = [(l["passo"], l[chave]) for l in log if chave in l]
        if pts:
            series.append(pd.Series(dict(pts)))
    if not series:
        return None
    tab = pd.concat(series, axis=1).sort_index()
    return tab.index.values, tab.mean(axis=1).values, tab.min(axis=1).values, tab.max(axis=1).values


def _linha(ax, df, d, m, chave, escala=1.0, **kw):
    r = _logs(df, d, m, chave)
    if r is None:
        return
    x, med, lo, hi = r
    ax.plot(x, med * escala, color=COR[m], label=m, **kw)
    if kw.get("ls", "-") == "-":
        ax.fill_between(x, lo * escala, hi * escala, color=COR[m], alpha=0.15, lw=0)


def _legenda(fig, metodos=METODOS):
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COR[m], label=m) for m in metodos], ncol=len(metodos),
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))


def curvas_aprendizado(df, pasta):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, d in zip(axs, CENARIOS):
        for m in METODOS:
            _linha(ax, df, d, m, "sucesso", escala=100, marker="o", ms=3)
        ax.set_title(TITULO[d]); ax.set_xlabel("passos de gradiente"); ax.set_ylim(-3, 103)
    axs[0].set_ylabel("taxa de sucesso (%)")
    fig.suptitle("Curva de aprendizado (avaliação periódica, 20 episódios; faixa = min–max entre sementes)", y=1.02)
    _legenda(fig); plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(pasta, "1_curvas_aprendizado.png"), dpi=120, bbox_inches="tight"); plt.close(fig)


def perdas(df, pasta):
    componentes = {  # (chave, rótulo, estilo)
        "BC": [("perda_bc", "entropia cruzada", "-")],
        "DQN offline": [("perda_td", "erro TD", "-")],
        "CQL": [("perda_td", "erro TD", "-"), ("penalidade_cql", "penalidade CQL", "--")],
        "IQL": [("perda_td", "erro TD (Q)", "-"), ("perda_v", "expectil (V)", "--"), ("perda_pi", "AWR (π)", ":")],
    }
    fig, axs = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True)
    for i, d in enumerate(CENARIOS):
        for j, m in enumerate(METODOS):
            ax = axs[i, j]
            for chave, rotulo, ls in componentes[m]:
                r = _logs(df, d, m, chave)
                if r is None:
                    continue
                x, med, lo, hi = r
                ax.plot(x, np.abs(med) + 1e-8, color=COR[m] if ls == "-" else TINTA, ls=ls, label=rotulo)
            vals = np.concatenate([np.abs(l.get_ydata()) for l in ax.get_lines()]) if ax.get_lines() else np.array([1.0])
            if vals.max() / max(vals.min(), 1e-12) > 5:   # escala log só quando a perda varia muito
                ax.set_yscale("log")
            if i == 0: ax.set_title(m, color=COR[m], fontweight="bold")
            if j == 0: ax.set_ylabel(f"{TITULO[d]}\nperda")
            if i == 2: ax.set_xlabel("passos de gradiente")
            if i == 0: ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Perdas de treino (média por intervalo, média entre sementes)", y=1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(pasta, "2_perdas.png"), dpi=120, bbox_inches="tight"); plt.close(fig)


def valores_q(df, pasta):
    metodos = ["DQN offline", "CQL", "IQL"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, d in zip(axs, CENARIOS):
        for m in metodos:
            _linha(ax, df, d, m, "q_max")
            _linha(ax, df, d, m, "q_ood", ls=":")
        ax.axhline(1.0, color=TINTA, ls="--", lw=1)
        ax.set_ylim(-0.6, 1.3); ax.set_title(TITULO[d]); ax.set_xlabel("passos de gradiente")
    axs[0].set_ylabel("Q médio nos estados do especialista")
    axs[1].text(axs[1].get_xlim()[1], 1.04, "retorno máximo possível < 1  ", ha="right", color=TINTA, fontsize=9)
    fig.suptitle("Valor estimado  —  linha cheia: max Q   |   pontilhada: max Q nas ações 3–6 (nunca usadas "
                 "pelo especialista; valores abaixo de -0,6 ficam fora da escala)", y=1.02, fontsize=10)
    _legenda(fig, metodos); plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(pasta, "3_valores_q.png"), dpi=120, bbox_inches="tight"); plt.close(fig)


def concordancia(df, pasta):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, d in zip(axs, CENARIOS):
        for m in METODOS:
            _linha(ax, df, d, m, "concordancia", escala=100)
        ax.set_ylim(0, 100); ax.set_title(TITULO[d]); ax.set_xlabel("passos de gradiente")
    axs[0].set_ylabel("concorda com o especialista (%)")
    fig.suptitle("Em quantos estados do especialista a política escolhe a mesma ação que ele", y=1.02)
    _legenda(fig); plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(pasta, "4_concordancia.png"), dpi=120, bbox_inches="tight"); plt.close(fig)


def acoes(df, pasta, acoes_dados):
    grupos = [("girar esq.", [0], "#86b6ef"), ("girar dir.", [1], "#1c5cab"),
              ("frente", [2], "#1baf7a"), ("inúteis (3–6)", [3, 4, 5, 6], "#c9c8c2")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
    linhas = ["dataset"] + METODOS
    for ax, d in zip(axs, CENARIOS):
        for y, nome in enumerate(linhas):
            if nome == "dataset":
                cont = np.array(acoes_dados[d], float)
            else:
                cont = np.sum([np.array(a) for a in df[(df.dataset == d) & (df.algo == nome)]["acoes"]], axis=0)
            frac = cont / cont.sum()
            esq = 0
            for rotulo, idx, cor in grupos:
                v = frac[idx].sum()
                ax.barh(y, v, left=esq, color=cor, edgecolor="white", linewidth=1.5,
                        label=rotulo if (y == 0 and d == "expert") else None)
                if v > 0.09:
                    ax.text(esq + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if cor in ("#1c5cab", "#1baf7a") else "#0b0b0b")
                esq += v
        ax.set_yticks(range(len(linhas)), linhas); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_title(TITULO[d]); ax.grid(False)
        ax.set_xlabel("fração das ações")
    fig.suptitle("Ações escolhidas pela política na avaliação final  ×  ações presentes no dataset", y=1.02)
    fig.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(pasta, "5_acoes.png"), dpi=120, bbox_inches="tight"); plt.close(fig)


def sucesso_final(df, pasta):
    fig, ax = plt.subplots(figsize=(11, 4))
    larg = 0.19
    for j, m in enumerate(METODOS):
        xs = np.arange(len(CENARIOS)) + (j - 1.5) * (larg + 0.01)
        medias = [df[(df.dataset == d) & (df.algo == m)].sucesso.mean() * 100 for d in CENARIOS]
        ax.bar(xs, medias, width=larg, color=COR[m], label=m)
        for i, (x, v) in enumerate(zip(xs, medias)):
            pts = df[(df.dataset == CENARIOS[i]) & (df.algo == m)].sucesso * 100
            ax.scatter([x] * len(pts), pts, s=10, color=TINTA, zorder=3)
            ax.text(x, max(v, pts.max()) + 2, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(3), [TITULO[d] for d in CENARIOS])
    ax.set_ylabel("taxa de sucesso final (%)"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    ax.set_title("Avaliação final (100 episódios; barras = média, pontos = sementes)")
    ax.legend(ncol=4, frameon=False, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(pasta, "6_sucesso_final.png"), dpi=120, bbox_inches="tight"); plt.close(fig)


def gerar(resultados, pasta="figuras", acoes_dados=None):
    os.makedirs(pasta, exist_ok=True)
    df = pd.DataFrame(resultados)
    if acoes_dados is None:
        import orl_minigrid as orl
        acoes_dados = {d: np.bincount(orl.obter_dados(d)["a"], minlength=7).tolist() for d in CENARIOS}
    curvas_aprendizado(df, pasta); perdas(df, pasta); valores_q(df, pasta)
    concordancia(df, pasta); acoes(df, pasta, acoes_dados); sucesso_final(df, pasta)
    print(f"6 gráficos salvos em '{pasta}/'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="resultados_offline_rl.json")
    ap.add_argument("--pasta", default="figuras")
    a = ap.parse_args()
    gerar(json.load(open(a.entrada)), a.pasta)
