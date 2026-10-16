"""Roda a comparação BC x DQN offline x CQL x IQL nos datasets D4RL/minigrid do Minari.

Uso:  python executar.py              (15.000 passos, 2 sementes)
      python executar.py --passos 2000 --sementes 0   (teste rápido)
Gera resultados_offline_rl.json e a pasta figuras/ com os gráficos de acompanhamento.
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import orl_minigrid as orl
import graficos

CENARIOS = ["expert", "random", "misto"]
METODOS = list(orl.ALGORITMOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passos", type=int, default=15_000)
    ap.add_argument("--sementes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--saida", default="resultados_offline_rl.json")
    args = ap.parse_args()

    # 0) baixa os datasets UMA vez, no processo principal, antes do paralelismo.
    #    (se cada processo tentar baixar ao mesmo tempo, o Windows dá PermissionError no cache)
    import minari
    for dataset_id in orl.IDS.values():
        print("verificando/baixando", dataset_id, flush=True)
        minari.load_dataset(dataset_id, download=True)

    # 1) treino em paralelo: um processo por núcleo
    jobs =[(d, m, s, args.passos) for d in CENARIOS for m in METODOS for s in args.sementes]
    jobs.sort(key=lambda j: j[1] != "IQL")  # os mais lentos primeiro
    t0, resultados = time.time(), []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for r in ex.map(orl.rodar_job, jobs):
            resultados.append(r)
            print(f"[{(time.time() - t0) / 60:5.1f} min] {r['dataset']:6s} {r['algo']:12s} "
                  f"seed {r['seed']}: sucesso {r['sucesso']:.0%}  retorno {r['retorno']:.3f}", flush=True)
    json.dump(resultados, open(args.saida, "w"))

    # 2) tabela com score normalizado (0 = política aleatória, 100 = especialista)
    df = pd.DataFrame(resultados)
    ret_aleatoria = orl.avaliar_aleatoria().mean()
    ret_especialista = orl.carregar_transicoes(minari.load_dataset(orl.IDS["expert"]))[1].mean()
    df["score"] = 100 * (df.retorno - ret_aleatoria) / (ret_especialista - ret_aleatoria)
    tabela = (df.groupby(["dataset", "algo"])[["sucesso", "retorno", "score"]].mean()
                .reindex(pd.MultiIndex.from_product([CENARIOS, METODOS])))
    tabela["sucesso"] *= 100
    print("\n", tabela.round(2).to_string())

    # 3) gráficos de acompanhamento (também podem ser refeitos depois com: python graficos.py)
    graficos.gerar(resultados, pasta="figuras")


if __name__ == "__main__":
    main()
