import numpy as np


class QNet:
    def __init__(self, dims, lr=1e-3, wd=0.0, how="he", seed=None):
        self.dims = list(dims)
        self.lr = lr
        self.wd = wd
        self.how = how
        self.rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for i in range(len(self.dims) - 1):
            nin, nout = self.dims[i], self.dims[i + 1]
            if how == "xavier":
                s = np.sqrt(2.0 / (nin + nout))
            else:
                s = np.sqrt(2.0 / nin)  # he — stuck with this after xavier blew up once
            self.W.append(self.rng.normal(0.0, s, (nin, nout)).astype(np.float64))
            self.b.append(np.zeros((1, nout), dtype=np.float64))

    def fwd(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        hs = [x.astype(np.float64)]
        zs = []
        cur = hs[0]
        for i in range(len(self.W)):
            z = cur @ self.W[i] + self.b[i]
            zs.append(z)
            if i < len(self.W) - 1:
                cur = np.maximum(0.0, z)
            else:
                cur = z  # raw Qs, can be neg
            hs.append(cur)
        return hs[-1], hs, zs

    def out(self, x):
        y, _, _ = self.fwd(x)
        return y

    def back(self, hs, zs, g):
        m = hs[0].shape[0]
        d = g.astype(np.float64)
        for i in reversed(range(len(self.W))):
            dW = (hs[i].T @ d) / m
            db = np.mean(d, axis=0, keepdims=True)
            if self.wd > 0:
                dW = dW + self.wd * self.W[i]
            if i > 0:
                d = (d @ self.W[i].T) * (zs[i - 1] > 0).astype(np.float64)
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def dump(self):
        return {
            "weights": [w.copy() for w in self.W],
            "biases": [bb.copy() for bb in self.b],
            "layer_sizes": list(self.dims),
            "learning_rate": self.lr,
            "l2_reg": self.wd,
            "init": self.how,
        }

    def load(self, p):
        self.W = [w.copy() for w in p["weights"]]
        self.b = [bb.copy() for bb in p["biases"]]
        self.dims = list(p["layer_sizes"])
        self.lr = float(p["learning_rate"])
        self.wd = float(p["l2_reg"])
        self.how = p.get("init", "he")
