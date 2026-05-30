# AGENTS.md

## Overview
PyTorch MARL algorithms (IQL, QMIX, VDN, COMA, QTRAN-base/alt, MAVEN, CommNet, G2ANet) trained on SMAC (StarCraft II). Pure research code — no tests, no CI, no lint.

## Quick start
```
pip install -r requirements.txt
python main.py --map=3m --alg=qmix
```

Evaluation only: `--evaluate=True --load_model=True`

## Architecture

```
main.py          → runs 8 independent trials by default, creates StarCraft2Env + Runner
runner.py        → training loop, evaluation, plotting
agent/agent.py   → Agents (standard) vs CommAgents (commnet/g2anet); dispatches to policy/
policy/          → per-algorithm learn() logic (vdn, qmix, coma, iql, etc.)
network/         → neural network modules (base_net, qmix_net, commnet, g2anet, etc.)
common/          → arguments, rollout, replay_buffer, utils
result/          → output plots (.png) and metrics (.npy) per alg/map
model/           → saved model checkpoints per alg/map
```

## Algorithm string dispatch (fragile, understand before modifying)

The `--alg` string controls dispatch at **three points** in `agent/agent.py`, `runner.py`, and `main.py`:

1. **Contains `commnet` or `g2anet`** → uses `CommAgents` + `CommRolloutWorker` (multi-agent communication). These must be composed with a training algorithm, e.g. `--alg=reinforce+commnet` or `--alg=central_v+g2anet`.

2. **Contains `coma`, `central_v`, or `reinforce`** → on-policy (no replay buffer), uses `get_coma_args()` / `get_centralv_args()` / `get_reinforce_args()`, and softmax action selection with epsilon-greedy.

3. **Everything else** (`vdn`, `iql`, `qmix`, `qtran_alt`, `qtran_base`, `maven`) → off-policy with replay buffer, uses `get_mixer_args()`. These are mixable value-function methods.

## Key non-obvious details

- `main.py` runs **8 independent trials** in a loop (`for i in range(8)`). The first evaluation result is printed and the loop breaks on `--evaluate=True`. For a single training run, edit the loop.
- `--alg` must be an **exact string match or substring match** depending on the dispatch point. Valid values are listed as a comment in `common/arguments.py:18-20`.
- On-policy algos (COMA, CentralV, Reinforce) train on the full episode rollout. Off-policy algos collect episodes into a ReplayBuffer and sample mini-batches.
- Models are auto-saved every `save_cycle` steps (default 5000) to `model/<alg>/<map>/`.
- `--replay_dir` must be an **absolute path** for replay saving to work.
- CUDA is off by default (`--cuda=False`).
- SMAC/PySC2 must be installed — SMAC depends on a local StarCraft II installation.
- Some comments in `agent/agent.py`, `common/rollout.py`, and `common/utils.py` are in Chinese.
- DyMA-CL lives in a separate project and its `--alg` value is not wired up here.
