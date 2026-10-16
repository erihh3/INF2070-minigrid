"""Offline RL em MiniGrid FourRooms: BC, DQN offline, CQL e IQL (ações discretas)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
import gymnasium as gym, minigrid  # noqa: F401  (registra os ambientes MiniGrid)

N_ACOES = 7
N_OBJ, N_COR, N_EST = 11, 6, 3
DIM_ENTRADA = 7 * 7 * (N_OBJ + N_COR + N_EST) + 4


# ----------------------------------------------------------------------------- dados
def carregar_transicoes(dataset):
    """Converte um MinariDataset em arrays compactos (imagem em uint8)."""
    img, dir_, a, r, img2, dir2, d, retornos, tamanhos = [], [], [], [], [], [], [], [], []
    for ep in dataset.iterate_episodes():
        o = ep.observations
        img.append(o["image"][:-1]); img2.append(o["image"][1:])
        dir_.append(o["direction"][:-1]); dir2.append(o["direction"][1:])
        a.append(ep.actions); r.append(ep.rewards)
        d.append(ep.terminations.astype(np.float32))  # truncation não é terminal
        retornos.append(ep.rewards.sum()); tamanhos.append(len(ep.actions))
    cat = np.concatenate
    dados = dict(img=cat(img), dir=cat(dir_).astype(np.int64), a=cat(a).astype(np.int64),
                 r=cat(r).astype(np.float32), img2=cat(img2), dir2=cat(dir2).astype(np.int64),
                 d=cat(d))
    return dados, np.array(retornos), np.array(tamanhos)


def juntar(*conjuntos):
    return {k: np.concatenate([c[k] for c in conjuntos]) for k in conjuntos[0]}


def codificar(img, direcao):
    """one-hot de (objeto, cor, estado) em cada célula da visão 7x7 + direção."""
    img = torch.as_tensor(img, dtype=torch.long)
    partes = [F.one_hot(img[..., 0].clamp(max=N_OBJ - 1), N_OBJ),
              F.one_hot(img[..., 1].clamp(max=N_COR - 1), N_COR),
              F.one_hot(img[..., 2].clamp(max=N_EST - 1), N_EST)]
    x = torch.cat(partes, -1).flatten(1).float()
    return torch.cat([x, F.one_hot(torch.as_tensor(direcao, dtype=torch.long), 4).float()], 1)


class Amostrador:
    def __init__(self, dados, seed=0):
        self.dados, self.n = dados, len(dados["a"])
        self.rng = np.random.default_rng(seed)

    def __call__(self, bs=256):
        i = self.rng.integers(self.n, size=bs)
        D = self.dados
        return (codificar(D["img"][i], D["dir"][i]), torch.as_tensor(D["a"][i]),
                torch.as_tensor(D["r"][i]), codificar(D["img2"][i], D["dir2"][i]),
                torch.as_tensor(D["d"][i]))


def mlp(saida=N_ACOES, h=256):
    return nn.Sequential(nn.Linear(DIM_ENTRADA, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                         nn.Linear(h, saida))


def atualizar_alvo(rede, alvo, tau=0.005):
    with torch.no_grad():
        for p, pa in zip(rede.parameters(), alvo.parameters()):
            pa.mul_(1 - tau).add_(tau * p)


# ----------------------------------------------------------------------------- registro
class Registro:
    """Acumula métricas de treino e, periodicamente, avalia a política no ambiente.

    A cada `log_cada` passos grava a média das métricas do intervalo e o Q nos estados de monitoramento.
    A cada `avaliar_cada` passos roda `n_aval` episódios para a curva de aprendizado."""

    def __init__(self, passos, s_monitor=None, a_monitor=None, log_cada=None, avaliar_cada=None, n_aval=20):
        self.log_cada = log_cada or max(1, passos // 30)
        self.avaliar_cada = avaliar_cada or max(1, passos // 10)
        self.s_monitor, self.a_monitor, self.n_aval = s_monitor, a_monitor, n_aval
        self.acum, self.linhas = {}, []

    def __call__(self, t, politica, q_fn=None, **metricas):
        for k, v in metricas.items():
            self.acum.setdefault(k, []).append(float(v))
        if t % self.log_cada and t % self.avaliar_cada:
            return
        linha = {"passo": t, **{k: float(np.mean(v)) for k, v in self.acum.items()}}
        self.acum = {}
        if self.s_monitor is not None:
            with torch.no_grad():
                if q_fn is not None:
                    qm = q_fn(self.s_monitor)
                    linha["q_max"] = qm.max(1).values.mean().item()
                    linha["q_ood"] = qm[:, 3:].max(1).values.mean().item()   # ações que o especialista nunca usa
                # concordância: % de estados do especialista em que a política escolhe a ação do especialista
                linha["concordancia"] = (politica(self.s_monitor).argmax(1) == self.a_monitor).float().mean().item()
        if t % self.avaliar_cada == 0:
            rets, suc, _ = avaliar(politica, n=self.n_aval, seed=50_000)   # sementes diferentes da avaliação final
            linha["sucesso"], linha["retorno"] = float(suc.mean()), float(rets.mean())
        self.linhas.append(linha)


# ----------------------------------------------------------------------------- algoritmos
def treinar_bc(dados, passos=20000, bs=256, seed=0, lr=1e-3, registro=None, **_):
    torch.manual_seed(seed); amostra = Amostrador(dados, seed)
    pi = mlp(); opt = torch.optim.Adam(pi.parameters(), lr)
    for t in range(1, passos + 1):
        s, a, *_ = amostra(bs)
        logits = pi(s)
        perda = F.cross_entropy(logits, a)
        opt.zero_grad(); perda.backward(); opt.step()
        if registro:
            registro(t, pi, perda_bc=perda.item(),
                     acuracia_treino=(logits.argmax(1) == a).float().mean().item())
    return pi


def treinar_q(dados, alpha=0.0, passos=20000, bs=256, gamma=0.99, seed=0, lr=3e-4, registro=None, **_):
    """alpha = 0 -> DQN offline (Double DQN); alpha > 0 -> CQL(H) discreto."""
    torch.manual_seed(seed); amostra = Amostrador(dados, seed)
    q, q_alvo = mlp(), mlp(); q_alvo.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr)
    for t in range(1, passos + 1):
        s, a, r, s2, d = amostra(bs)
        with torch.no_grad():
            a2 = q(s2).argmax(1, keepdim=True)
            y = r + gamma * (1 - d) * q_alvo(s2).gather(1, a2).squeeze(1)
        q_s = q(s); q_sa = q_s.gather(1, a[:, None]).squeeze(1)
        perda_td = F.mse_loss(q_sa, y)
        penal = (torch.logsumexp(q_s, 1) - q_sa).mean()
        perda = perda_td + alpha * penal if alpha > 0 else perda_td
        opt.zero_grad(); perda.backward(); opt.step(); atualizar_alvo(q, q_alvo)
        if registro:
            extra = {"penalidade_cql": penal.item()} if alpha > 0 else {}
            registro(t, q, q_fn=q, perda_td=perda_td.item(), **extra)
    return q


def treinar_iql(dados, passos=20000, bs=256, gamma=0.99, seed=0, lr=3e-4, tau_expectile=0.7,
                beta=10.0, registro=None, **_):
    """IQL: V por regressão de expectil sobre Q(s, a_dados); Q por TD com V(s');
    política extraída por regressão ponderada pela vantagem (AWR). Nunca consulta ações OOD."""
    torch.manual_seed(seed); amostra = Amostrador(dados, seed)
    q1, q2, v, pi = mlp(), mlp(), mlp(1), mlp()
    q1a, q2a = mlp(), mlp(); q1a.load_state_dict(q1.state_dict()); q2a.load_state_dict(q2.state_dict())
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr)
    opt_v = torch.optim.Adam(v.parameters(), lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr)
    q_min = lambda s: torch.min(q1(s), q2(s))
    for t in range(1, passos + 1):
        s, a, r, s2, d = amostra(bs)
        # 1) V(s) ~ expectil tau de Q_alvo(s, a) com a vindo dos DADOS
        with torch.no_grad():
            q_alvo = torch.min(q1a(s), q2a(s)).gather(1, a[:, None]).squeeze(1)
        u = q_alvo - v(s).squeeze(1)
        peso = torch.abs(tau_expectile - (u < 0).float())
        perda_v = (peso * u ** 2).mean()
        opt_v.zero_grad(); perda_v.backward(); opt_v.step()
        # 2) Q(s, a) <- r + gamma V(s')   (sem max sobre ações!)
        with torch.no_grad():
            y = r + gamma * (1 - d) * v(s2).squeeze(1)
        perda_q = (F.mse_loss(q1(s).gather(1, a[:, None]).squeeze(1), y) +
                   F.mse_loss(q2(s).gather(1, a[:, None]).squeeze(1), y)) / 2
        opt_q.zero_grad(); perda_q.backward(); opt_q.step()
        atualizar_alvo(q1, q1a); atualizar_alvo(q2, q2a)
        # 3) política: BC ponderado por exp(beta * vantagem)
        with torch.no_grad():
            adv = q_alvo - v(s).squeeze(1)
            w = torch.exp(beta * adv).clamp(max=100.0)
        perda_pi = (w * F.cross_entropy(pi(s), a, reduction="none")).mean()
        opt_pi.zero_grad(); perda_pi.backward(); opt_pi.step()
        if registro:
            registro(t, pi, q_fn=q_min, perda_td=perda_q.item(), perda_v=perda_v.item(),
                     perda_pi=perda_pi.item(), peso_awr=w.mean().item())
    return pi


# ----------------------------------------------------------------------------- avaliação
def avaliar(rede, n=100, seed=10_000, env_id="MiniGrid-FourRooms-v0"):
    """Avaliação online (só possível porque temos o simulador). Política gulosa.
    Retorna retornos, sucessos e a contagem de ações escolhidas."""
    env = gym.make(env_id); rets, suc, acoes = [], [], np.zeros(N_ACOES)
    for k in range(n):
        o, _ = env.reset(seed=seed + k); done, R = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(rede(codificar(o["image"][None], [o["direction"]])).argmax())
            acoes[a] += 1
            o, r, term, trunc, _ = env.step(a); R += r; done = term or trunc
        rets.append(R); suc.append(float(term))
    return np.array(rets), np.array(suc), acoes


def avaliar_aleatoria(n=100, seed=10_000, env_id="MiniGrid-FourRooms-v0"):
    env = gym.make(env_id); env.action_space.seed(0); rets = []
    for k in range(n):
        env.reset(seed=seed + k); done, R = False, 0.0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample()); R += r; done = term or trunc
        rets.append(R)
    return np.array(rets)


# ----------------------------------------------------------------------------- experimentos
IDS = {"expert": "D4RL/minigrid/fourrooms-v0", "random": "D4RL/minigrid/fourrooms-random-v0"}
_CACHE = {}


def obter_dados(nome):
    """'expert', 'random' ou 'misto' (= expert + random). Carregado uma vez por processo."""
    import minari
    if nome not in _CACHE:
        if nome == "misto":
            _CACHE[nome] = juntar(obter_dados("expert"), obter_dados("random"))
        else:
            _CACHE[nome] = carregar_transicoes(minari.load_dataset(IDS[nome], download=True))[0]
    return _CACHE[nome]


ALGORITMOS = {
    "BC":          (treinar_bc,  {}),
    "DQN offline": (treinar_q,   {"alpha": 0.0}),
    "CQL":         (treinar_q,   {"alpha": 0.5}),
    "IQL":         (treinar_iql, {"tau_expectile": 0.7, "beta": 3.0}),
}


def rodar_job(args):
    """Treina um algoritmo em um dataset com uma semente, registrando a evolução, e avalia a política final."""
    import time
    nome_ds, algo, seed, passos = args
    dados = obter_dados(nome_ds)
    exp = obter_dados("expert")
    idx = np.random.default_rng(123).choice(len(exp["a"]), 1000, replace=False)
    registro = Registro(passos, s_monitor=codificar(exp["img"][idx], exp["dir"][idx]),
                        a_monitor=torch.as_tensor(exp["a"][idx]))
    fn, kw = ALGORITMOS[algo]
    t0 = time.time()
    politica = fn(dados, passos=passos, seed=seed, registro=registro, **kw)
    rets, suc, acoes = avaliar(politica)
    return dict(dataset=nome_ds, algo=algo, seed=seed, retorno=float(rets.mean()), sucesso=float(suc.mean()),
                acoes=acoes.tolist(), log=registro.linhas, segundos=time.time() - t0)
