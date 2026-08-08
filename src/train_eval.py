import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dqn import SpxAgent
from src.environment import Book


def rollout(agent, book, explore=False):
    s = book.reset()
    done = False
    rs = []
    acts = []
    while not done:
        a = agent.act(s, explore=explore)
        s2, r, done, info = book.step(a)
        rs.append(r)
        acts.append(a)
        s = s2

    eq = np.array(book.curve, dtype=float)
    tot = (eq[-1] / eq[0]) - 1.0
    dr = np.diff(eq) / eq[:-1]
    sharpe = 0.0
    if len(dr) > 1 and np.std(dr) > 1e-12:
        sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(252))
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / np.maximum(peak, 1e-8)).min())
    return {
        "final_value": float(eq[-1]),
        "total_return": float(tot),
        "avg_reward": float(np.mean(rs)) if rs else 0.0,
        "cumulative_reward": float(np.sum(rs)),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "equity_curve": eq,
        "actions": acts,
        "action_counts": {"hold": acts.count(0), "buy": acts.count(1), "sell": acts.count(2)},
        "n_steps": len(acts),
    }


def run_eps(agent, train_book, val_book=None, n_eps=30, cb=None):
    hist = {
        "episode_reward": [],
        "episode_return": [],
        "epsilon": [],
        "loss": [],
        "val_return": [],
        "best_val_return": None,
        "best_params": None,
    }
    best = -1e18

    for ep in range(1, n_eps + 1):
        s = train_book.reset()
        done = False
        ep_r = 0.0
        losses = []
        t = 0
        while not done:
            a = agent.act(s, explore=True)
            s2, r, done, info = train_book.step(a)
            agent.remember(s, a, r, s2, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            s = s2
            ep_r += r
            t += 1

        agent.bump_eps()
        eq = train_book.curve
        ep_ret = (eq[-1] / eq[0]) - 1.0
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        hist["episode_reward"].append(ep_r)
        hist["episode_return"].append(ep_ret)
        hist["epsilon"].append(agent.eps)
        hist["loss"].append(avg_loss)

        val_ret = None
        if val_book is not None and (ep % 5 == 0 or ep == n_eps):
            old = agent.eps
            agent.eps = 0.0
            m = rollout(agent, val_book, explore=False)
            agent.eps = old
            val_ret = m["total_return"]
            hist["val_return"].append({"episode": ep, "return": val_ret})
            if val_ret > best:
                best = val_ret
                hist["best_val_return"] = best
                hist["best_params"] = agent.net.dump()

        if cb is not None:
            cb({"episode": ep, "reward": ep_r, "return": ep_ret, "epsilon": agent.eps, "loss": avg_loss, "val_return": val_ret, "steps": t})

    return hist


COLS = [
    "experiment_number", "timestamp", "mode", "parameters", "dataset_size",
    "train_test_split", "episodes", "train_return", "test_return",
    "train_sharpe", "test_sharpe", "test_max_drawdown", "buy_hold_return",
    "final_test_value", "notes",
]


def append_exp(csv_path, record):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLS).writeheader()
    with csv_path.open("r", newline="") as f:
        n = len(list(csv.DictReader(f))) + 1
    row = {k: "" for k in COLS}
    row.update(record)
    row["experiment_number"] = n
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if isinstance(row.get("parameters"), dict):
        row["parameters"] = json.dumps(row["parameters"])
    with csv_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLS).writerow(row)
    return n


def make_figs(hist, tr_m, te_m, bh, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(hist["episode_return"], color="#1f77b4")
    ax[0, 0].set_title("Training Episode Return")
    ax[0, 0].set_xlabel("Episode"); ax[0, 0].set_ylabel("Return"); ax[0, 0].grid(True, alpha=0.3)
    ax[0, 1].plot(hist["episode_reward"], color="#ff7f0e")
    ax[0, 1].set_title("Training Episode Cumulative Reward")
    ax[0, 1].set_xlabel("Episode"); ax[0, 1].grid(True, alpha=0.3)
    ax[1, 0].plot(hist["loss"], color="#2ca02c")
    ax[1, 0].set_title("Average TD Loss per Episode")
    ax[1, 0].set_xlabel("Episode"); ax[1, 0].set_ylabel("MSE"); ax[1, 0].grid(True, alpha=0.3)
    ax[1, 1].plot(hist["epsilon"], color="#d62728")
    ax[1, 1].set_title("eps")
    ax[1, 1].set_xlabel("Episode"); ax[1, 1].set_ylabel("eps"); ax[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "training_curves.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    paths["training_curves"] = p1

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(te_m["equity_curve"], label="DQN Agent", color="#1f77b4", linewidth=1.8)
    bh_eq = bh["equity_curve"]
    n = min(len(te_m["equity_curve"]), len(bh_eq))
    ax.plot(bh_eq[:n], label="Buy & Hold", color="#7f7f7f", linestyle="--", linewidth=1.5)
    ax.set_title("Test Set Equity Curve"); ax.set_xlabel("Trading Day"); ax.set_ylabel("Portfolio Value ($)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "test_equity_curve.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    paths["test_equity"] = p2

    counts = te_m["action_counts"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labs = ["Hold", "Buy", "Sell"]
    vals = [counts["hold"], counts["buy"], counts["sell"]]
    ax.bar(labs, vals, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax.set_title("Test-Set Action Distribution"); ax.set_ylabel("Count")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    p3 = out_dir / "action_distribution.png"
    fig.savefig(p3, dpi=150); plt.close(fig)
    paths["actions"] = p3

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    tbl = [
        ["Metric", "DQN (Train)", "DQN (Test)", "Buy & Hold (Test)"],
        ["Total Return", "%.2f%%" % (tr_m["total_return"] * 100), "%.2f%%" % (te_m["total_return"] * 100), "%.2f%%" % (bh["total_return"] * 100)],
        ["Sharpe Ratio", "%.3f" % tr_m["sharpe"], "%.3f" % te_m["sharpe"], "%.3f" % bh["sharpe"]],
        ["Max Drawdown", "%.2f%%" % (tr_m["max_drawdown"] * 100), "%.2f%%" % (te_m["max_drawdown"] * 100), "%.2f%%" % (bh["max_drawdown"] * 100)],
        ["Final Value ($)", "%.2f" % tr_m["final_value"], "%.2f" % te_m["final_value"], "%.2f" % bh["final_value"]],
    ]
    t = ax.table(cellText=tbl, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.2, 1.4)
    ax.set_title("Performance Summary", pad=12)
    fig.tight_layout()
    p4 = out_dir / "metrics_table.png"
    fig.savefig(p4, dpi=150); plt.close(fig)
    paths["metrics"] = p4

    summary = {
        "train": {k: v for k, v in tr_m.items() if k not in ("equity_curve", "actions")},
        "test": {k: v for k, v in te_m.items() if k not in ("equity_curve", "actions")},
        "buy_hold": {"final_value": bh["final_value"], "total_return": bh["total_return"], "sharpe": bh["sharpe"], "max_drawdown": bh["max_drawdown"]},
    }
    sp = out_dir / "metrics_summary.json"
    with sp.open("w") as f:
        json.dump(summary, f, indent=2)
    paths["summary"] = sp
    return paths


def rewrite_log(csv_path, txt_path):
    df = pd.read_csv(csv_path)
    lines = ["EXPERIMENT LOG — Deep Q-Network Algorithmic Trading Agent", "=" * 88,
             "%-6s%-48s%-34s" % ("Exp#", "Parameters", "Results"), "-" * 88]
    for _, row in df.iterrows():
        params = str(row.get("parameters", ""))
        if len(params) > 46:
            params = params[:43] + "..."
        results = "TrainRet=%s | TestRet=%s" % (row.get("train_return", ""), row.get("test_return", ""))
        lines.append("%-6d%-48s%-34s" % (int(row["experiment_number"]), params, results))
        lines.append("      Split=%s  N=%s  Episodes=%s" % (row.get("train_test_split", ""), row.get("dataset_size", ""), row.get("episodes", "")))
        lines.append("      Sharpe(test)=%s  MaxDD(test)=%s  B&H=%s" % (row.get("test_sharpe", ""), row.get("test_max_drawdown", ""), row.get("buy_hold_return", "")))
        if row.get("notes"):
            lines.append("      Notes: %s" % row.get("notes"))
        lines.append("")
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines))
