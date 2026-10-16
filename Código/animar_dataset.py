"""Animação do movimento do robô a partir de um episódio do dataset Minari (MiniGrid FourRooms).

O dataset não grava o mapa nem a posição do agente, mas grava a SEMENTE de cada episódio.
Com ela, o script recria o mesmo mapa (env.reset(seed=...)), reaplica as AÇÕES gravadas e
desenha a trajetória completa. A cada passo, confere se a observação reproduzida é igual à
observação gravada no dataset.

Uso:
  python animar_dataset.py                                   # episódio 0 do dataset especialista
  python animar_dataset.py --dataset random --sucesso        # um episódio aleatório que chegou ao objetivo
  python animar_dataset.py --episodio 5 --fps 3 --saida ep5.gif
  python animar_dataset.py --episodio 5 --mostrar            # abre uma janela em vez de (além de) salvar
"""
import argparse
import numpy as np, matplotlib
import gymnasium as gym, minigrid, minari  # noqa: F401
from minigrid.core.grid import Grid

IDS = {"expert": "D4RL/minigrid/fourrooms-v0", "random": "D4RL/minigrid/fourrooms-random-v0"}
NOMES_ACOES = ["girar à esquerda", "girar à direita", "andar para frente",
               "pegar (sem efeito)", "soltar (sem efeito)", "alternar (sem efeito)", "concluir (sem efeito)"]
TILE = 32  # pixels por célula no render do MiniGrid


def escolher_episodio(ds, args):
    if args.episodio is not None:
        return args.episodio
    if args.sucesso:  # primeiro episódio que terminou no objetivo (útil para o dataset aleatório)
        rng = np.random.default_rng(args.semente_busca)
        for i in rng.permutation(ds.total_episodes):
            ep = next(ds.iterate_episodes([int(i)]))
            if ep.terminations[-1]:          # terminou porque chegou ao objetivo
                return int(i)
    return 0


def reconstruir(ds, idx):
    """Reaplica as ações do episódio no ambiente e coleta tudo o que a animação precisa."""
    ep = next(ds.iterate_episodes([idx]))
    meta = next(iter(ds.storage.get_episode_metadata([idx])))
    seed = meta.get("seed")
    if seed is None:
        raise SystemExit("Este dataset não registrou a semente dos episódios; não dá para recriar o mapa.")

    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=int(seed))
    u = env.unwrapped
    quadros, posicoes, divergencias = [env.render()], [tuple(u.agent_pos)], 0
    divergencias += int(not np.array_equal(obs["image"], ep.observations["image"][0]))
    for t, a in enumerate(ep.actions):
        obs, r, term, trunc, _ = env.step(int(a))
        quadros.append(env.render()); posicoes.append(tuple(u.agent_pos))
        divergencias += int(not np.array_equal(obs["image"], ep.observations["image"][t + 1]))
    env.close()
    return dict(ep=ep, seed=seed, quadros=quadros, posicoes=posicoes, divergencias=divergencias)


def visao_do_agente(img):
    """Desenha a observação 7x7 GRAVADA NO DATASET (o que a rede de fato recebe)."""
    grade, vis = Grid.decode(img)
    return grade.render(TILE, agent_pos=(3, 6), agent_dir=3, highlight_mask=vis)


def animar(dados, args, titulo_ds):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ep, quadros, pos = dados["ep"], dados["quadros"], np.array(dados["posicoes"])
    T = len(ep.actions)
    retorno_acum = np.concatenate([[0.0], np.cumsum(ep.rewards)])
    centros = pos * TILE + TILE / 2

    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 0.62], wspace=0.06, hspace=0.18,
                          left=0.01, right=0.99, top=0.87, bottom=0.02)
    ax_mapa, ax_pov, ax_txt = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for ax in (ax_mapa, ax_pov, ax_txt):
        ax.axis("off")

    im_mapa = ax_mapa.imshow(quadros[0])
    trilha, = ax_mapa.plot([], [], color="#3987e5", lw=3, alpha=0.85, solid_capstyle="round")
    inicio = ax_mapa.scatter(*centros[0], s=90, color="#3987e5", edgecolor="white", zorder=3)
    ax_mapa.set_title("Mapa completo (recriado pela semente) e trajetória", fontsize=11)
    im_pov = ax_pov.imshow(visao_do_agente(ep.observations["image"][0]))
    ax_pov.set_title("Observação do dataset (visão 7×7 do robô)", fontsize=11)
    txt = ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10.5,
                      transform=ax_txt.transAxes)
    fig.suptitle(f"{titulo_ds}  —  episódio {args.idx}  (semente {dados['seed']}, {T} passos)", fontsize=12)

    n_final = max(1, int(args.fps * 1.5))  # segura o último quadro ~1,5 s

    def quadro(k):
        t = min(k, T)
        im_mapa.set_data(quadros[t])
        trilha.set_data(centros[: t + 1, 0], centros[: t + 1, 1])
        im_pov.set_data(visao_do_agente(ep.observations["image"][t]))
        if t == 0:
            linhas = ["passo      : 0 (estado inicial)", "ação       : —", "recompensa : —"]
        else:
            a = int(ep.actions[t - 1])
            linhas = [f"passo      : {t}/{T}", f"ação       : {a} = {NOMES_ACOES[a]}",
                      f"recompensa : {ep.rewards[t - 1]:.3f}"]
        linhas += [f"retorno    : {retorno_acum[t]:.3f}",
                   f"posição    : x={pos[t][0]}, y={pos[t][1]}"]
        if t == T:
            fim = "chegou ao objetivo ✔" if ep.terminations[-1] else (
                  "truncado em 100 passos ✘" if ep.truncations[-1] else "fim")
            linhas.append(f"fim        : {fim}")
        txt.set_text("\n".join(linhas))
        return im_mapa, trilha, im_pov, txt

    anim = FuncAnimation(fig, quadro, frames=T + 1 + n_final, interval=1000 / args.fps, blit=False)
    if args.saida:
        anim.save(args.saida, writer=PillowWriter(fps=args.fps))
        print(f"animação salva em {args.saida}")
    if args.mostrar:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=IDS, default="expert")
    ap.add_argument("--episodio", type=int, default=None, help="índice do episódio (padrão: 0)")
    ap.add_argument("--sucesso", action="store_true", help="sorteia um episódio que chegou ao objetivo")
    ap.add_argument("--semente-busca", type=int, default=0, help="semente do sorteio usado por --sucesso")
    ap.add_argument("--fps", type=float, default=4)
    ap.add_argument("--saida", default=None, help="arquivo .gif (padrão: animacao_<dataset>_ep<N>.gif)")
    ap.add_argument("--mostrar", action="store_true", help="abre a animação numa janela")
    args = ap.parse_args()
    if not args.mostrar:
        matplotlib.use("Agg")

    ds = minari.load_dataset(IDS[args.dataset], download=True)
    args.idx = escolher_episodio(ds, args)
    if args.saida is None and not args.mostrar:
        args.saida = f"animacao_{args.dataset}_ep{args.idx}.gif"

    dados = reconstruir(ds, args.idx)
    T = len(dados["ep"].actions)
    if dados["divergencias"] == 0:
        print(f"episódio {args.idx}: {T} passos reproduzidos; todas as {T + 1} observações batem com o dataset ✔")
    else:
        print(f"atenção: {dados['divergencias']} de {T + 1} observações diferem do dataset "
              "(versão do minigrid diferente da usada para gerar os dados?)")
    animar(dados, args, "Especialista" if args.dataset == "expert" else "Aleatório")


if __name__ == "__main__":
    main()
