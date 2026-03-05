import time
import numpy as np
import config


def log_training_progress(round_idx, reward):
    print(f"[Round {round_idx}] Median Reward: {reward:.3f}")


def hyperparameter_sweep(param_name, values, train_fn):
    print(f"\n🔍 Hyperparameter Sweep: {param_name}")
    results = {}
    for v in values:
        print(f"Testing {param_name} = {v}")
        results[v] = train_fn(v)
    return results


def reward_sensitivity_test(modifier_fn, train_fn):
    print("\n🎯 Reward Sensitivity Test")
    return train_fn(modifier_fn)


def scalability_test(gym_counts, train_fn):
    print("\n📈 Scalability Test")
    results = {}
    for g in gym_counts:
        print(f"Testing with {g} gyms")
        results[g] = train_fn(g)
    return results


def measure_training_time(train_fn):
    start = time.time()
    train_fn()
    end = time.time()
    return end - start
