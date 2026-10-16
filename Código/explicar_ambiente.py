"""Infográfico do ambiente MiniGrid FourRooms usado nos datasets D4RL/minigrid do Minari.

Uso:  python explicar_ambiente.py            -> salva ambiente_fourrooms.png
      python explicar_ambiente.py --mostrar  -> abre numa janela
"""
import argparse, copy
import numpy as np, matplotlib
import gymnasium as gym, minigrid, minari  # noqa: F401
from minigrid.core.grid import Grid

TILE = 32
AZUL, LARANJA, VERDE, TINTA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
NOMES_OBJ = {0: "não visível", 1: "vazio", 2: "parede", 8: "objetivo", 10: "agente"}


def preparar(seed, passos_ate_visao):
    """Recria o episódio 5 do dataset especialista: mapa inicial e um estado em que o objetivo está à vista."""
    ds = minari.load_dataset("D4RL/minigrid/fourrooms-v0", download=True)
    ep = next(ds.iterate_episodes([5]))
    seed = next(iter(ds.storage.get_episode_metadata([5])))["seed"]
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    env.reset(seed=int(seed))
    inicial = copy.deepcopy(env.unwrapped)
    mapa0 = env.render()
    for a in ep.actions[:passos_ate_visao]:
        obs, *_ = env.step(int(a))
    estado_visao = copy.deepcopy(env.unwrapped)
    mapa_visao = env.render()
    comprimentos = [len(e.actions) for e in ds.iterate_episodes()]
    return dict(env=env, inicial=inicial, mapa0=mapa0, mapa_visao=mapa_visao, obs=obs,
                estado_visao=estado_visao, comprimentos=np.array(comprimentos), seed=seed)


def recorte(u, centro, raio=3):
    """Recorta o render numa janela FIXA em volta de `centro` (assim o deslocamento fica visível)."""
    img = u.get_frame(highlight=True, tile_size=TILE)
    x, y = centro
    x0, y0 = max(0, x - raio), max(0, y - raio)
    return img[y0 * TILE:(y + raio + 1) * TILE, x0 * TILE:(x + raio + 1) * TILE]


def painel_mapa(ax, d):
    u = d["inicial"]
    ax.imshow(d["mapa0"]); ax.axis("off")
    ax.set_title("A. O mundo: grade 19×19 com 4 salas", loc="left", fontweight="bold")
    c = lambda x, y: (x * TILE + TILE / 2, y * TILE + TILE / 2)
    for nome, (x, y) in {"sala 1": (4.5, 4.5), "sala 2": (13.5, 4.5), "sala 3": (4.5, 13.5), "sala 4": (13.5, 13.5)}.items():
        ax.text(*c(x, y), nome, color="white", ha="center", va="center", fontsize=10, alpha=0.8)
    # passagens: células vazias nas paredes do meio (x = 9 ou y = 9)
    n = u.grid.width
    for k in range(1, n - 1):
        for (x, y) in [(9, k), (k, 9)]:
            if u.grid.get(x, y) is None:
                ax.add_patch(matplotlib.patches.Rectangle((x * TILE, y * TILE), TILE, TILE, fill=False,
                                                          ec="#eda100", lw=2.5))
    ax.annotate("passagem estreita", xy=c(*[(9, k) for k in range(1, n - 1) if u.grid.get(9, k) is None][0]),
                xytext=c(12.5, 7.5), color="#eda100", fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#eda100"))
    ax.annotate("robô (triângulo aponta\npara onde ele olha)", xy=c(*u.agent_pos), xytext=c(1.2, 17.7),
                color="white", fontsize=9, arrowprops=dict(arrowstyle="->", color="white"))
    gx, gy = [(x, y) for x in range(n) for y in range(n) if u.grid.get(x, y) and u.grid.get(x, y).type == "goal"][0]
    ax.annotate("objetivo", xy=c(gx, gy), xytext=c(gx - 5, gy + 2.2), color=VERDE, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VERDE))
    ax.text(0, n * TILE + 22, "Posição inicial, direção e objetivo são sorteados a cada episódio.\n"
            "A faixa clara é a área que o robô consegue ver.", fontsize=9, color=CINZA, va="top")


def painel_visao(ax_mapa, ax_pov, ax_num, d):
    ax_mapa.imshow(d["mapa_visao"]); ax_mapa.axis("off")
    ax_mapa.set_title("B. O que o robô observa: só uma janela 7×7", loc="left", fontweight="bold")
    ax_mapa.text(0, d["mapa_visao"].shape[0] + 22, "Mapa real (o robô NÃO recebe isto)", fontsize=9, color=CINZA, va="top")

    img = d["obs"]["image"]
    grade, vis = Grid.decode(img)
    ax_pov.imshow(grade.render(TILE, agent_pos=(3, 6), agent_dir=3, highlight_mask=vis)); ax_pov.axis("off")
    ax_pov.set_title("B (cont.) visão egocêntrica\n(robô sempre embaixo, olhando 'para cima')", fontsize=9)

    obj = img[:, :, 0].T  # linhas = frente/trás, colunas = esquerda/direita
    cores = {0: "#1a1a19", 1: "#e9e8e4", 2: "#9e9d98", 8: VERDE, 10: LARANJA}
    ax_num.set_xlim(-0.5, 6.5); ax_num.set_ylim(6.5, -0.5); ax_num.set_aspect("equal"); ax_num.axis("off")
    for yy in range(7):
        for xx in range(7):
            v = int(obj[yy, xx]) if not (xx == 3 and yy == 6) else 10
            ax_num.add_patch(matplotlib.patches.Rectangle((xx - .5, yy - .5), 1, 1, fc=cores.get(v, "white"), ec="white"))
            ax_num.text(xx, yy, str(int(obj[yy, xx])), ha="center", va="center", fontsize=9,
                        color="white" if v in (0, 8, 10) else TINTA)
    ax_num.set_title("o que entra na rede (canal 'objeto')\n+ cor + estado + direção = 7×7×3 + 1", fontsize=9)
    itens = [f"{k}={v}" for k, v in NOMES_OBJ.items()]
    legenda = "  ".join(itens[:3]) + "\n" + "  ".join(itens[3:])
    ax_num.text(3, 7.3, legenda, ha="center", va="top", fontsize=8, color=CINZA)


def painel_acoes(axs, d):
    nomes = [("0 · girar à esquerda", 0), ("1 · girar à direita", 1), ("2 · andar para frente", 2)]
    u0 = d["estado_visao"]
    centro = tuple(u0.agent_pos)
    axs[0].imshow(recorte(copy.deepcopy(u0), centro)); axs[0].set_title("estado atual", fontsize=9); axs[0].axis("off")
    for ax, (nome, a) in zip(axs[1:4], nomes):
        u = copy.deepcopy(u0); u.step_count = 0
        u.step(a)
        ax.imshow(recorte(u, centro)); ax.set_title(nome, fontsize=9, color=AZUL, fontweight="bold"); ax.axis("off")
    ax = axs[4]; ax.axis("off")
    ax.text(0.5, 0.62, "3 · pegar\n4 · soltar\n5 · alternar\n6 · concluir", ha="center", va="center",
            fontsize=9, color=CINZA, transform=ax.transAxes)
    ax.text(0.5, 0.18, "não fazem nada aqui\n(não há objetos nem portas)\nmas gastam um passo", ha="center",
            va="center", fontsize=8.5, color=LARANJA, transform=ax.transAxes)
    axs[0].text(0, 1.28, "C. As 7 ações (discretas) — efeito de cada uma a partir do mesmo estado",
                transform=axs[0].transAxes, fontweight="bold", fontsize=11)


def painel_recompensa(ax, d):
    t = np.arange(0, 101)
    ax.plot(t, 1 - 0.9 * t / 100, color=AZUL, lw=2)
    ax.set_xlim(0, 105); ax.set_ylim(0, 1.05)
    ax.set_xlabel("passos até chegar ao objetivo"); ax.set_ylabel("recompensa recebida")
    ax.set_title("D. Recompensa: esparsa e maior quanto mais rápido", loc="left", fontweight="bold")
    ax.text(42, 0.78, r"$r = 1 - 0{,}9\cdot\frac{\mathrm{passos}}{100}$" "\nsó no passo em que pisa no objetivo;\n0 em todos os outros",
            fontsize=9, color=TINTA)
    ax.scatter([100], [0], color=LARANJA, zorder=3)
    ax.annotate("100 passos sem chegar:\nepisódio truncado, retorno 0", xy=(100, 0), xytext=(58, 0.12),
                fontsize=8.5, color=LARANJA, arrowprops=dict(arrowstyle="->", color=LARANJA))
    ax2 = ax.twinx(); ax2.grid(False)
    cont, _, _ = ax2.hist(d["comprimentos"], bins=np.arange(0, 40, 2), color=VERDE, alpha=0.3)
    ax2.set_ylim(0, cont.max() * 2.6)        # ocupa só a parte de baixo do gráfico
    ax2.set_yticks([]); ax2.spines[["top", "right"]].set_visible(False)
    ax.text(3, 0.42, "duração dos episódios do especialista\n(1 a 35 passos → recompensa 0,69 a 0,99)",
            color="#13805a", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)


def painel_ciclo(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("E. Ciclo agente–ambiente e o que vai para o dataset", loc="left", fontweight="bold")
    def caixa(x, y, w, h, txt, cor):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=cor, ec="none", alpha=0.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    caixa(0.3, 3.6, 3.0, 1.4, "Política\n(robô)", AZUL)
    caixa(6.6, 3.6, 3.0, 1.4, "Ambiente\nFourRooms", CINZA)
    seta = dict(arrowstyle="-|>", lw=1.8, color=TINTA, mutation_scale=14)
    ax.annotate("", xy=(6.55, 4.6), xytext=(3.35, 4.6), arrowprops=seta)
    ax.text(4.95, 4.8, "ação  a (0–6)", ha="center", fontsize=9)
    ax.annotate("", xy=(3.35, 4.0), xytext=(6.55, 4.0), arrowprops=seta)
    ax.text(4.95, 3.55, "observação s' (7×7×3 + direção)\nrecompensa r, fim?", ha="center", fontsize=8.5, va="top")
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0.3, 0.3), 9.3, 1.9, boxstyle="round,pad=0.08",
                                                   fc="#eda100", ec="none", alpha=0.18))
    ax.text(4.95, 1.75, "Dataset Minari (gravado uma vez, por outra política)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(4.95, 0.95, "(s, a, r, s', terminou?) × 10 mil passos (especialista) ou 1 milhão (aleatória)\n"
            "Offline RL aprende SÓ com isto — sem nunca rodar o ciclo acima durante o treino.",
            ha="center", fontsize=8.5, color=TINTA)
    ax.annotate("", xy=(4.95, 2.3), xytext=(4.95, 3.0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#b07a00", ls="--"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="ambiente_fourrooms.png")
    ap.add_argument("--mostrar", action="store_true")
    args = ap.parse_args()
    if not args.mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10})

    d = preparar(seed=None, passos_ate_visao=15)
    fig = plt.figure(figsize=(17, 15.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 0.55, 0.8], hspace=0.42, wspace=0.12,
                          left=0.03, right=0.97, top=0.93, bottom=0.04)
    painel_mapa(fig.add_subplot(gs[0, 0:3]), d)
    painel_visao(fig.add_subplot(gs[0, 3:5]), fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[1, 5]), d)
    # coluna 6 da linha 0: legenda curta
    ax_leg = fig.add_subplot(gs[0, 5]); ax_leg.axis("off")
    ax_leg.text(0.0, 0.95, "Por que isso importa\npara Offline RL", fontweight="bold", fontsize=10, va="top")
    ax_leg.text(0.0, 0.78,
                "• O especialista que gerou os\n  dados via o mapa inteiro.\n\n"
                "• O agente treinado só vê a\n  janela 7×7: muitas vezes o\n  objetivo está fora de vista.\n\n"
                "• Janelas iguais aparecem em\n  lugares diferentes, com ações\n  ótimas diferentes.\n\n"
                "• Por isso nenhum método sem\n  memória chega a 100%.", fontsize=9, va="top", color=CINZA)
    sub = gs[1, 0:4].subgridspec(1, 5, wspace=0.08)
    painel_acoes([fig.add_subplot(sub[0, k]) for k in range(5)], d)
    painel_recompensa(fig.add_subplot(gs[2, 0:3]), d)
    painel_ciclo(fig.add_subplot(gs[2, 3:6]))
    fig.suptitle("MiniGrid FourRooms — o ambiente dos datasets D4RL/minigrid do Minari", fontsize=15, fontweight="bold")
    fig.savefig(args.saida, dpi=110)
    print(f"infográfico salvo em {args.saida}")
    if args.mostrar:
        plt.show()


if __name__ == "__main__":
    main()
