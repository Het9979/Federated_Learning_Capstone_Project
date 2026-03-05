import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Example data from your latest run
# -----------------------------
fed_reward = 0.913

baselines = {
    "Expert Oracle": 0.778,
    "Random": -5.689,
    "Immediate Best": -7.056,
    "Two Best": -6.750,
}

local_models = {"Local A": 0.989, "Local B": 0.680, "Local C": 0.737, "Local D": 0.713}

# Simulated federated reward progression over 20 rounds (example)
fed_progression = [
    0.447,
    0.480,
    0.502,
    0.520,
    0.540,
    0.560,
    0.580,
    0.600,
    0.625,
    0.640,
    0.660,
    0.680,
    0.700,
    0.730,
    0.760,
    0.780,
    0.810,
    0.840,
    0.870,
    0.913,
]


# -----------------------------
# 1️⃣ Bar chart: final rewards
# -----------------------------
def plot_final_rewards():
    labels = ["Federated Global"] + list(local_models.keys()) + list(baselines.keys())
    rewards = [fed_reward] + list(local_models.values()) + list(baselines.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, rewards, color=["blue"] + ["green"] * 4 + ["red"] * 4)

    plt.ylabel("Median Reward")
    plt.title("Final Rewards: Federated, Local Models & Baselines")
    plt.xticks(rotation=45)
    plt.ylim(min(rewards) - 1, max(rewards) + 0.5)

    # Show value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


# -----------------------------
# 2️⃣ Line chart: federated reward progression
# -----------------------------
def plot_fed_progression():
    rounds = np.arange(1, len(fed_progression) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fed_progression, marker="o", color="blue")
    plt.title("Federated Model Reward Progression Across Rounds")
    plt.xlabel("Federated Round")
    plt.ylabel("Median Reward")
    plt.grid(True)
    plt.ylim(min(fed_progression) - 0.1, max(fed_progression) + 0.1)

    # Show value labels at each point
    for x, y in zip(rounds, fed_progression):
        plt.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Run plots
# -----------------------------
if __name__ == "__main__":
    plot_final_rewards()
    plot_fed_progression()
