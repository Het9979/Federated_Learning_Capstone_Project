import copy
import time
import csv
import numpy as np
import torch
from baselines import (
    evaluate_local_model,
    immediate_best,
    random_policy,
    run_expert_baseline,
    two_best,
)
import config
from model import DQN
from federated import FLClient, FLServer
from evaluate import evaluate_model
from baselines import *
from plots import plot_rewards, plot_bar


# -------------------------------------------------
# Utility
# -------------------------------------------------
def save_results(filename, headers, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# -------------------------------------------------
# 1️⃣ FEDERATED TRAINING CORE
# -------------------------------------------------
def run_federated(rounds=config.FED_ROUNDS, noise_std=0.0):
    clients = [FLClient(v) for v in config.CLIENT_VARIANTS]
    global_model = DQN().to(config.DEVICE)
    server = FLServer(global_model)

    history = []

    for r in range(rounds):
        wts, sizes = [], []

        for client in clients:
            client.set_weights(global_model.state_dict())
            size = client.train_local()
            weights = client.get_weights()

            # 🔒 Differential Privacy Noise
            if noise_std > 0:
                for k in weights:
                    weights[k] += noise_std * torch.randn_like(weights[k])

            wts.append(weights)
            sizes.append(size)

        server.aggregate(wts, sizes)
        score = evaluate_model(global_model)
        history.append(score)
        print(f"[Round {r}] Global Reward: {score:.3f}")

    return global_model, clients, history


# -------------------------------------------------
# 2️⃣ BASELINE COMPARISON (Section 3.5)
# -------------------------------------------------
def run_baselines(global_model, clients):
    results = []

    results.append(["Federated", evaluate_model(global_model)])
    for i, v in enumerate(config.CLIENT_VARIANTS):
        results.append([f"Local_{v}", evaluate_local_model(clients[i].model, v)])

    results.append(["Expert", run_expert_baseline()])
    results.append(["ImmediateBest", immediate_best()])
    results.append(["TwoBest", two_best()])
    results.append(["Random", random_policy()])

    save_results("baseline_results.csv", ["Model", "MedianReward"], results)
    plot_bar(results, "Baseline Comparison")

    return results


# -------------------------------------------------
# 3️⃣ HYPERPARAMETER SWEEP (3.6.1)
# -------------------------------------------------
def hyperparameter_experiments():
    rows = []
    for lr in [1e-4, 5e-4, 1e-3]:
        config.LR = lr
        model, _, hist = run_federated(rounds=10)
        rows.append([lr, hist[-1]])
    save_results("hyperparam_lr.csv", ["LR", "FinalReward"], rows)
    return rows


# -------------------------------------------------
# 4️⃣ REWARD SENSITIVITY (3.6.2)
# -------------------------------------------------
def reward_sensitivity_experiment():
    rows = []
    original = copy.deepcopy(config)

    # Skill-focused
    config.REWARD_WEIGHTS = (2.0, -0.3, 0.3)
    model, _, _ = run_federated(rounds=10)
    rows.append(["SkillFocus", evaluate_model(model)])

    # Fatigue-focused
    config.REWARD_WEIGHTS = (1.0, -1.0, 0.2)
    model, _, _ = run_federated(rounds=10)
    rows.append(["FatigueFocus", evaluate_model(model)])

    # Motivation-focused
    config.REWARD_WEIGHTS = (1.0, -0.2, 1.0)
    model, _, _ = run_federated(rounds=10)
    rows.append(["MotivationFocus", evaluate_model(model)])

    save_results("reward_sensitivity.csv", ["Scenario", "Reward"], rows)
    return rows


# -------------------------------------------------
# 5️⃣ SCALABILITY (3.6.3)
# -------------------------------------------------
def scalability_experiment():
    rows = []
    for n in [2, 4, 8]:
        config.CLIENT_VARIANTS = ["A", "B", "C", "D"][:n]
        model, _, _ = run_federated(rounds=10)
        rows.append([n, evaluate_model(model)])
    save_results("scalability.csv", ["NumGyms", "Reward"], rows)
    return rows


# -------------------------------------------------
# 6️⃣ TRAINING EFFICIENCY (3.6.4)
# -------------------------------------------------
def efficiency_experiment():
    start = time.time()
    run_federated(rounds=10)
    fed_time = time.time() - start

    # Centralized baseline (single client with all data)
    start = time.time()
    client = FLClient("A")
    client.train_local(episodes=200)
    cent_time = time.time() - start

    rows = [["Federated", fed_time], ["Centralized", cent_time]]
    save_results("efficiency.csv", ["Mode", "Seconds"], rows)
    return rows


# -------------------------------------------------
# 7️⃣ PRIVACY VS ACCURACY (3.6.6)
# -------------------------------------------------
def privacy_experiment():
    rows = []
    for noise in [0.0, 0.01, 0.05]:
        model, _, _ = run_federated(rounds=10, noise_std=noise)
        rows.append([noise, evaluate_model(model)])
    save_results("privacy_tradeoff.csv", ["NoiseSTD", "Reward"], rows)
    return rows
