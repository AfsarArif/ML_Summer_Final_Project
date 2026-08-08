import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqn import SpxAgent
from src.environment import Book
from src.features import get_spx
from src.train_eval import append_exp, make_figs, rewrite_log, rollout, run_eps


def parse_args():
    p = argparse.ArgumentParser(description="spx dqn trader")
    p.add_argument("--mode", choices=["prepare", "train", "all"], default="all")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--episodes", type=int, default=40)  # 20 was thin, 60 just slower
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.97)
    p.add_argument("--target-update", type=int, default=200)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--initial-cash", type=float, default=10000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--notes", type=str, default="")
    return p.parse_args()


def prep(args):
    print("Preparing S&P 500 dataset...")
    tr, te = get_spx(args.data_dir, force=args.force_download)
    print("  Train rows:", len(tr))
    print("  Test rows: ", len(te))
    print("  Saved under:", args.data_dir)
    return tr, te


def go_train(args):
    tr, te = prep(args)
    train_book = Book(tr, cash0=args.initial_cash, fee=args.transaction_cost)
    test_book = Book(te, cash0=args.initial_cash, fee=args.transaction_cost)

    agent = SpxAgent(
        n_in=train_book.n_feats(),
        n_act=3,
        hiddens=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        eps0=args.epsilon_start,
        eps_lo=args.epsilon_end,
        eps_mul=args.epsilon_decay,
        mem_cap=args.buffer_size,
        bs=args.batch_size,
        sync_n=args.target_update,
        wd=args.l2,
        seed=args.seed,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print("Training DQN | state_dim=%d | episodes=%d | hidden=%s" % (train_book.n_feats(), args.episodes, args.hidden))

    def on_ep(info):
        if info["episode"] % 5 == 0 or info["episode"] == 1:
            val = info["val_return"]
            vs = "%.2f%%" % (val * 100) if val is not None else "n/a"
            print("  Episode %3d | return=%7.2f%% | loss=%.5f | eps=%.3f | val_ret=%s" % (
                info["episode"], info["return"] * 100, info["loss"], info["epsilon"], vs))

    hist = run_eps(agent, train_book, val_book=test_book, n_eps=args.episodes, cb=on_ep)

    if hist.get("best_params") is not None:
        agent.net.load(hist["best_params"])
        agent.tgt.W = [w.copy() for w in agent.net.W]
        agent.tgt.b = [bb.copy() for bb in agent.net.b]
        print("Restored best validation weights (val_return=%s)" % hist.get("best_val_return"))

    agent.eps = 0.0
    tr_m = rollout(agent, train_book, explore=False)
    te_m = rollout(agent, test_book, explore=False)
    bh = test_book.bh()

    paths = make_figs(hist, tr_m, te_m, bh, args.results_dir)
    print("Saved figures:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    params = {
        "algorithm": "DQN",
        "network": "MLP%s" % str(tuple([train_book.n_feats()] + list(args.hidden) + [3])),
        "lr": args.lr, "gamma": args.gamma, "batch_size": args.batch_size,
        "buffer": args.buffer_size, "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end, "epsilon_decay": args.epsilon_decay,
        "target_update": args.target_update, "l2": args.l2,
        "transaction_cost": args.transaction_cost, "seed": args.seed,
        "error_function": "MSE (Bellman TD)",
    }
    exp_no = append_exp(args.log_dir / "experiments.csv", {
        "mode": "train",
        "parameters": params,
        "dataset_size": len(tr) + len(te),
        "train_test_split": "80:20 chronological",
        "episodes": args.episodes,
        "train_return": "%.2f%%" % (tr_m["total_return"] * 100),
        "test_return": "%.2f%%" % (te_m["total_return"] * 100),
        "train_sharpe": "%.3f" % tr_m["sharpe"],
        "test_sharpe": "%.3f" % te_m["sharpe"],
        "test_max_drawdown": "%.2f%%" % (te_m["max_drawdown"] * 100),
        "buy_hold_return": "%.2f%%" % (bh["total_return"] * 100),
        "final_test_value": "%.2f" % te_m["final_value"],
        "notes": args.notes or "Primary DQN training run",
    })
    rewrite_log(args.log_dir / "experiments.csv", args.log_dir / "experiments.log")
    print("Logged experiment #%d -> %s" % (exp_no, args.log_dir / "experiments.log"))

    print("\n=== Results Summary ===")
    print("Train return: %.2f%%" % (tr_m["total_return"] * 100))
    print("Test return:  %.2f%%" % (te_m["total_return"] * 100))
    print("Buy&Hold:     %.2f%%" % (bh["total_return"] * 100))
    print("Test Sharpe:  %.3f" % te_m["sharpe"])
    print("Test MaxDD:   %.2f%%" % (te_m["max_drawdown"] * 100))
    print("B&H Sharpe:   %.3f" % bh["sharpe"])
    print("B&H MaxDD:    %.2f%%" % (bh["max_drawdown"] * 100))

    serializable = {
        "episode_return": hist["episode_return"],
        "episode_reward": hist["episode_reward"],
        "epsilon": hist["epsilon"],
        "loss": [None if (isinstance(x, float) and np.isnan(x)) else x for x in hist["loss"]],
        "val_return": hist["val_return"],
        "best_val_return": hist.get("best_val_return"),
    }
    (args.results_dir / "training_history.json").write_text(json.dumps(serializable, indent=2))
    return agent, tr_m, te_m, bh


def main():
    args = parse_args()
    np.random.seed(args.seed)
    if args.mode == "prepare":
        prep(args)
    elif args.mode == "train":
        go_train(args)
    elif args.mode == "all":
        prep(args)
        go_train(args)
    else:
        raise ValueError("bad mode " + str(args.mode))


if __name__ == "__main__":
    main()
