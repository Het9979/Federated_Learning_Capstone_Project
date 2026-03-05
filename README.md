````md
# FedRL — Federated Reinforcement Learning on a Rehab Simulation (DQN / Dueling-DQN)

FedRL is a **federated reinforcement learning** prototype that trains a shared policy across multiple heterogeneous clients **without centralizing trajectories**.  
The project simulates a **robot-assisted rehabilitation** setting where each client represents a different “site” (e.g., noisier outcomes, faster fatigue accumulation, slower motivation recovery). Each client trains locally with **DQN-style learning** (experience replay + target network), and the server aggregates model updates using **FedAvg**.

> **Why this matters:** In privacy-constrained domains (healthcare, mobile, edge), you often can’t pool raw interaction data. FedRL shows an end-to-end pipeline for “learn together, keep data local.”

---

## ✨ Key Features

- **Rehab simulation environment** with patient dynamics:
  - latent skill progression, fatigue accumulation, motivation adaptation
  - 6D state representation and discrete “difficulty level” actions
- **RL training pipeline (DQN-style):**
  - experience replay buffer
  - target network updates
  - epsilon-greedy exploration
- **Federated Learning (FedAvg):**
  - multiple non-IID clients (`A/B/C/D`) train locally
  - server aggregates weights using weighted average (by local steps)
- **Baselines:**
  - random policy, greedy heuristics, and an **oracle expert** (hidden-state access)
- **Experiment suites:**
  - feasibility sweeps (patients × architectures × hyperparams × reward scenarios)
  - privacy–utility tradeoff via noisy updates
  - scalability experiments (vary number of clients)
- **Evaluation:**
  - reports **median episodic reward** (robust to stochastic variance)

---

## 🧠 System Overview

### Architecture (high level)

```mermaid
flowchart TB
  subgraph Client["Client (variant A/B/C/D)"]
    E[RehabEnv] --> T[Trajectory (s,a,r,s',done)]
    T --> RB[Replay Buffer]
    RB --> TR[DQN Trainer]
    TR --> LM[Local Model Weights]
  end

  subgraph Server["Federated Server"]
    AGG[FedAvg Aggregation] --> GM[Global Model Weights]
  end

  GM -->|broadcast| LM
  LM -->|upload weights + steps| AGG

  GM --> EV[Evaluation: median episodic reward]
  EV --> OUT[CSV Results + Plots]
````

---

## 📁 Repository Structure

> **Important:** Scripts use local imports like `from env import RehabEnv`, so run them **from inside the `FedRL/` folder**.

```
FedRL/
  config.py                 # all hyperparameters + experiment grids
  env.py                    # Patient + RehabEnv simulator
  model.py                  # DQN + DuelingDQN networks
  buffer.py                 # replay buffer (uniform)
  trainer.py                # DQN training step + target updates
  federated.py              # FLClient local training + FLServer FedAvg
  evaluate.py               # evaluation loop (median episodic reward)
  baselines.py              # random/greedy/oracle baselines
  feasibility.py            # feasibility sweep runner + CSV logging
  experiments.py            # extra experiments (privacy/scalability/efficiency)
  plots.py                  # plotting utilities
  plots_results.py          # example plotting script
  sumtree.py                # advanced PER + n-step + Double DQN (standalone path)
  main.py                   # demo runner (see "Known Issues" below)
  feasibility_results.csv   # example results (generated output)
```

---

## ⚙️ Setup

### Requirements

* Python 3.9+ (recommended)
* `torch`, `numpy`, `tqdm`, `matplotlib`

### Install

```bash
pip install numpy tqdm matplotlib torch
```

> For GPU installs of PyTorch, follow the official PyTorch install instructions for your CUDA version.

---

## 🚀 Quickstart

### 1) Run a small federated training run (recommended entrypoint)

This uses the most consistent experiment path (`feasibility.py` uses **DuelingDQN** end-to-end):

```bash
cd FedRL
python - << 'PY'
from feasibility import run_custom_fed

# Small run for a fast sanity-check
model, history = run_custom_fed(
    rounds=3,
    local_episodes=20,
    eval_episodes=50,
    patients=1000,
    hidden_dims=(128, 256),
)

print("History (median reward per round):", history)
PY
```

### 2) Run the full feasibility sweep (writes CSV continuously)

```bash
cd FedRL
python feasibility.py
```

Outputs:

* `feasibility_results.csv` (and a `_backup.csv` if the file is locked)

### 3) Run additional experiments

```bash
cd FedRL
python - << 'PY'
import experiments

# Examples (uncomment to run)
# experiments.hyperparameter_experiments()
# experiments.reward_sensitivity_experiment()
# experiments.scalability_experiment()
# experiments.privacy_experiment()

print("See experiments.py for available experiment runners.")
PY
```

---

## 🧪 Environment, State, Actions, Reward

### State (6D)

`[skill, avg_effort, fatigue, motivation, t_norm, remaining_norm]`

* Values are clipped/normalized to keep learning stable.
* Time features are normalized to `[0,1]`.

### Action space

Discrete difficulty levels:

* `action ∈ {0, 1, ..., NUM_ROBOTS-1}`
* mapped to difficulty via: `difficulty = action / NUM_ROBOTS`

### Reward

Reward is shaped to balance success with human factors:

* encourage success
* penalize fatigue
* encourage motivation

Weights are controlled by:

```py
REWARD_WEIGHTS = (w1, w2, w3)
```

---

## 📊 Evaluation

Primary metric:

* **Median episodic reward** over `EVAL_EPISODES`

Why median?

* Robust to stochasticity (noise + varied patient profiles)

---

## 🔬 Results (Example)

A sample `feasibility_results.csv` is included in this repo. It records runs across:

* patient counts
* hidden layer sizes
* learning rate / batch size / local episodes
* reward-weight scenarios

To reproduce (or extend) results:

```bash
cd FedRL
python feasibility.py
```

---

## 🧩 Configuration

All knobs live in `config.py`, including:

* environment sizes (`NUM_PATIENTS`, `TIMESTEPS_PER_SESSION`, `NUM_ROBOTS`)
* RL hyperparams (`GAMMA`, `LR`, `BATCH_SIZE`, epsilon schedule, etc.)
* FL settings (`FED_ROUNDS`, `CLIENT_VARIANTS`)
* experiment grids (`PATIENT_COUNTS`, `ARCHITECTURES`, `HYPERPARAMS`, `REWARD_SCENARIOS`)

---

## ⚠️ Known Issues / Notes (Read This)

This repo is intentionally “research-prototype style” and includes a few rough edges:

1. **Model mismatch in `main.py` and `experiments.py`**

* `FLClient` currently uses **DuelingDQN**
* `main.py` / `experiments.py` instantiate **DQN**
* This can cause `state_dict` key mismatches during `load_state_dict()`

✅ Recommended: use `feasibility.py` / `run_custom_fed()` as the main entrypoint, or unify models across scripts.

2. **Episode truncation config**

* `config.MAX_LOCAL_STEPS` is set/used in feasibility scripts, but `env.py` currently uses `TIMESTEPS_PER_SESSION` for termination.
* If you want true truncation control, wire `MAX_LOCAL_STEPS` into `RehabEnv`.

3. **Duplicate `FLServer` definition**

* `federated.py` defines `FLServer` twice (same FedAvg logic).
  ✅ Safe improvement: remove the duplicate definition.

4. **Baseline evaluation batching**

* `baselines.evaluate_local_model()` calls the model on an unbatched tensor (may break dueling / dropout behavior).
  ✅ Safe improvement: add `.unsqueeze(0)` and set `model.eval()` during evaluation.

---

## 🛠️ Roadmap / Improvements (Great for extending or publishing)

* Integrate `sumtree.py` into the main training path:

  * Prioritized replay (PER), n-step returns, Double DQN, gradient clipping
* Add cross-variant evaluation (A/B/C/D) + worst-client metric
* Upgrade FedAvg → **FedProx** or **SCAFFOLD** for non-IID stability
* Replace mutable global config with a `dataclass Config` passed explicitly
* Add a `requirements.txt` + CLI (`argparse`) for reproducible runs

---

## 📜 License

Add a license if you plan to open-source publicly (MIT/Apache-2.0 are common).

---

## 🙌 Acknowledgements

Built as a learning + research prototype to explore the intersection of:

* federated learning
* reinforcement learning
* non-IID client heterogeneity
* privacy-utility tradeoffs
