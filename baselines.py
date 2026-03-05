import numpy as np
import torch
from env import RehabEnv
import config


# --------------------------------------------------
# 1️⃣ LOCAL SUBMODEL EVALUATION
# --------------------------------------------------
def evaluate_local_model(model, variant, episodes=200):
    env = RehabEnv(variant)
    scores = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0
        while not done:
            with torch.no_grad():
                a = torch.argmax(model(torch.FloatTensor(s).to(config.DEVICE))).item()
            s, r, done = env.step(a)
            total += r
        scores.append(total)
    return np.median(scores)


# --------------------------------------------------
# 2️⃣ EXPERT ORACLE (Perfect Knowledge)
# --------------------------------------------------
def expert_policy(env):
    """
    Oracle uses hidden patient state to pick optimal difficulty.
    """
    best_reward = -1e9
    best_action = 0
    for a in range(config.ACTION_DIM):
        difficulty = a / config.ACTION_DIM
        # Simulate one-step reward using hidden states
        success = env.patient.skill - difficulty
        fatigue_penalty = env.patient.fatigue
        motivation_bonus = env.patient.motivation
        reward = 1.5 * (success - 0.5) - 0.5 * fatigue_penalty + 0.5 * motivation_bonus
        if reward > best_reward:
            best_reward = reward
            best_action = a
    return best_action


def run_expert_baseline(episodes=200):
    env = RehabEnv("A")
    scores = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0
        while not done:
            a = expert_policy(env)
            s, r, done = env.step(a)
            total += r
        scores.append(total)
    return np.median(scores)


# --------------------------------------------------
# 3️⃣ STATIC BASELINES
# --------------------------------------------------


def random_policy():
    env = RehabEnv("A")
    scores = []
    for _ in range(200):
        s = env.reset()
        done = False
        total = 0
        while not done:
            a = np.random.randint(config.ACTION_DIM)
            s, r, done = env.step(a)
            total += r
        scores.append(total)
    return np.median(scores)


def immediate_best():
    """Greedy: choose robot closest to patient skill"""
    env = RehabEnv("A")
    scores = []
    for _ in range(200):
        s = env.reset()
        done = False
        total = 0
        while not done:
            skill = env.patient.skill
            diffs = [
                abs(skill - (a / config.ACTION_DIM)) for a in range(config.ACTION_DIM)
            ]
            a = np.argmin(diffs)
            s, r, done = env.step(a)
            total += r
        scores.append(total)
    return np.median(scores)


def two_best():
    """Alternate between two closest difficulty levels"""
    env = RehabEnv("A")
    scores = []
    toggle = 0
    for _ in range(200):
        s = env.reset()
        done = False
        total = 0
        while not done:
            skill = env.patient.skill
            diffs = np.argsort(
                [abs(skill - (a / config.ACTION_DIM)) for a in range(config.ACTION_DIM)]
            )
            a = diffs[toggle % 2]
            toggle += 1
            s, r, done = env.step(a)
            total += r
        scores.append(total)
    return np.median(scores)
