import numpy as np
import pandas as pd

from src.features import FEATS


class Book:
    # 0=sit, 1=go long all cash, 2=flatten
    def __init__(self, df, cash0=10000.0, fee=0.001, r_mult=100.0, cols=None):
        self.df = df.reset_index(drop=True)
        self.cash0 = float(cash0)
        self.fee = float(fee)  # 10bps — 50bps made the agent freeze forever
        self.r_mult = float(r_mult)
        self.cols = list(cols or FEATS)
        self.px = self.df["Close"].astype(float).to_numpy()
        self.T = len(self.df) - 1

        self.cash = self.cash0
        self.sh = 0.0
        self.long_ = 0
        self.t = 0
        self.curve = []

    def n_feats(self):
        return len(self.cols)

    def _nav(self, p):
        return self.cash + self.sh * p

    def _x(self):
        row = self.df.iloc[self.t].copy()
        p = float(self.px[self.t])
        nav = max(self._nav(p), 1e-8)
        row["position"] = float(self.long_)
        row["cash_ratio"] = float(self.cash / nav)
        return row[self.cols].astype(float).to_numpy()

    def reset(self):
        self.cash = self.cash0
        self.sh = 0.0
        self.long_ = 0
        self.t = 0
        self.curve = [self.cash0]
        return self._x()

    def step(self, a):
        p = float(self.px[self.t])
        nav0 = self._nav(p)
        a = int(a)

        # used to slap a -0.01 on illegal buys; agent just never bought again
        if a == 1 and self.long_ == 0:
            self.sh = (self.cash * (1.0 - self.fee)) / p
            self.cash = 0.0
            self.long_ = 1
        elif a == 2 and self.long_ == 1:
            self.cash = self.sh * p * (1.0 - self.fee)
            self.sh = 0.0
            self.long_ = 0

        self.t += 1
        p2 = float(self.px[self.t])
        nav1 = self._nav(p2)
        # % change * 100 so TD targets aren't ~1e-4
        rew = ((nav1 - nav0) / max(nav0, 1e-8)) * self.r_mult
        self.curve.append(nav1)

        done = self.t >= self.T
        nxt = np.zeros(self.n_feats(), dtype=np.float64) if done else self._x()
        return nxt, float(rew), done, {"nav": nav1, "px": p2, "long": self.long_, "a": a}

    def bh(self):
        p0 = float(self.px[0])
        p1 = float(self.px[self.T])
        sh = (self.cash0 * (1.0 - self.fee)) / p0
        fin = sh * p1 * (1.0 - self.fee)
        ret = (fin / self.cash0) - 1.0
        eq = ((self.px[: self.T + 1] / p0) * self.cash0).astype(float)
        dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
        sh_r = 0.0
        if len(dr) > 1 and np.std(dr) > 1e-12:
            sh_r = float(np.mean(dr) / np.std(dr) * np.sqrt(252))
        peak = np.maximum.accumulate(eq)
        mdd = float(((eq - peak) / np.maximum(peak, 1e-8)).min())
        return {"final_value": float(fin), "total_return": float(ret), "sharpe": sh_r, "max_drawdown": mdd, "equity_curve": eq}
