import numpy as np

from src.mlp import QNet


class _Mem:
    # ring buffer of transitions — 20k covers ~a few full train passes on daily bars
    def __init__(self, cap, dim, seed=None):
        self.cap = int(cap)
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((self.cap, self.dim), np.float64)
        self.s2 = np.zeros((self.cap, self.dim), np.float64)
        self.a = np.zeros((self.cap,), np.int64)
        self.r = np.zeros((self.cap,), np.float64)
        self.d = np.zeros((self.cap,), np.float64)
        self.i = 0
        self.n = 0

    def __len__(self):
        return self.n

    def add(self, s, a, r, s2, done):
        j = self.i
        self.s[j] = s
        self.a[j] = a
        self.r[j] = r
        self.s2[j] = s2
        self.d[j] = 1.0 if done else 0.0
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def batch(self, bs):
        idx = self.rng.choice(self.n, size=bs, replace=False)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class SpxAgent:
    def __init__(
        self,
        n_in,
        n_act=3,
        hiddens=(64, 64),
        lr=1e-3,
        gamma=0.99,
        eps0=1.0,
        eps_lo=0.05,
        eps_mul=0.995,
        mem_cap=10000,
        bs=64,
        sync_n=100,
        wd=1e-4,
        seed=42,
    ):
        self.n_in = n_in
        self.n_act = n_act
        self.gamma = gamma
        self.eps = eps0
        self.eps_lo = eps_lo
        self.eps_mul = eps_mul
        self.bs = bs
        self.sync_n = sync_n
        self.ticks = 0
        self.rng = np.random.default_rng(seed)

        layers = [n_in] + list(hiddens) + [n_act]
        self.net = QNet(layers, lr=lr, wd=wd, how="he", seed=seed)
        self.tgt = QNet(layers, lr=lr, wd=wd, how="he", seed=seed)
        # hard-copy weights into target once at start
        self.tgt.W = [w.copy() for w in self.net.W]
        self.tgt.b = [bb.copy() for bb in self.net.b]
        self.buf = _Mem(mem_cap, n_in, seed=seed)

    def act(self, obs, explore=True):
        if explore and self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_act))
        qv = self.net.out(obs.reshape(1, -1))[0]
        return int(np.argmax(qv))

    def bump_eps(self):
        self.eps = max(self.eps_lo, self.eps * self.eps_mul)

    def remember(self, s, a, r, s2, done):
        self.buf.add(s, a, r, s2, done)

    def update(self):
        if len(self.buf) < self.bs:
            return None
        S, A, R, S2, D = self.buf.batch(self.bs)
        qhat, hs, zs = self.net.fwd(S)

        # double: pick a* from online, score with tgt (vanilla dqn was too optimistic on choppy weeks)
        a_star = np.argmax(self.net.out(S2), axis=1)
        q_next = self.tgt.out(S2)
        ii = np.arange(self.bs)
        y = R + self.gamma * q_next[ii, a_star] * (1.0 - D)

        ymat = qhat.copy()
        ymat[ii, A] = y

        g = (2.0 / self.bs) * (qhat - ymat)
        nrm = np.linalg.norm(g)
        if nrm > 1.0:
            g = g / nrm  # saw one nan run without this

        self.net.back(hs, zs, g)
        td_err = float(np.mean((qhat - ymat) ** 2))
        self.ticks += 1

        if self.ticks % self.sync_n == 0:
            self.tgt.W = [w.copy() for w in self.net.W]
            self.tgt.b = [bb.copy() for bb in self.net.b]

        return td_err
