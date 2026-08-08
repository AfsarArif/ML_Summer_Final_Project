# DQN Trading Agent — Submission

**Author:** Mohamed Afsar Harsath Arif (NetID: mxh210050)  
**Topic:** Deep Reinforcement Learning (Double DQN on S&P 500)

## Contents

| Item | Path |
|------|------|
| IEEE report | `Final_Report_DQN_Trading.pdf` |
| Code | `main.py`, `src/` |
| Dataset | `data/` |
| Results / figures | `results/` |
| Experiment log | `logs/experiments.log` |
| Run instructions | this `README.md` |

## Dataset

Publicly hosted on GitHub:  
https://github.com/AfsarArif/ML_Summer_Final_Project/tree/main/data

S&P 500 daily OHLCV (Yahoo Finance `^GSPC`, 2010+). Files in `data/`:

- `sp500_raw.csv` — raw prices
- `sp500_train.csv` / `sp500_test.csv` — 80/20 chronological split
- `sp500_processed.csv` — combined with `split` column


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --mode prepare
python main.py --mode train --episodes 60
```

Or all at once:

```bash
python main.py --mode all --episodes 60
```

