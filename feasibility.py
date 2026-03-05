import copy
import csv
import torch
from tqdm import tqdm
import config
from model import DuelingDQN

# Assumes the FLClient/FLServer classes are in federated.py
from federated import FLClient, FLServer
from evaluate import evaluate_model
import os


# -----------------------------
# Utility to save results (Robust)
# -----------------------------
def save_results(filename, headers, rows):
    """
    Saves results to CSV. If the file is locked (e.g. open in Excel),
    it saves to a backup file instead of crashing.
    """
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
    except PermissionError:
        print(f"\n⚠️  WARNING: Could not write to '{filename}'. Is it open in Excel?")

        # Create a unique backup filename
        backup_name = filename.replace(".csv", "_backup.csv")
        print(f"   Saving to '{backup_name}' instead to prevent crashing...\n")

        with open(backup_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)


# -----------------------------
# Run federated experiment with custom settings
# -----------------------------
def run_custom_fed(
    hidden_dims=(128, 256),
    patients=1000,
    hyperparams=None,
    reward=None,
    rounds=5,
    max_env_steps=100,  # Controls environment truncation (max steps per episode)
    local_episodes=20,  # Controls how many episodes the client trains per round
    eval_episodes=20,
):
    # Update config dynamically
    config.HIDDEN_1, config.HIDDEN_2 = hidden_dims
    config.NUM_PATIENTS = patients

    if hyperparams:
        config.LR = hyperparams["LR"]
        config.BATCH_SIZE = hyperparams["BATCH_SIZE"]
        # Note: LOCAL_EPOCHS from config is passed as local_episodes argument

    if reward:
        config.REWARD_WEIGHTS = (reward["w1"], reward["w2"], reward["w3"])

    config.EVAL_EPISODES = eval_episodes
    # Used by RehabEnv to determine "done" state (truncation)
    config.MAX_LOCAL_STEPS = max_env_steps

    # Initialize clients and global model
    clients = [FLClient(v) for v in config.CLIENT_VARIANTS]
    global_model = DuelingDQN().to(config.DEVICE)
    server = FLServer(global_model)

    history = []

    round_pbar = tqdm(range(rounds), desc="Fed Rounds", position=0)

    for r in round_pbar:
        wts, sizes = [], []
        for client in clients:
            client.set_weights(global_model.state_dict())

            # UPDATED: Call train_local with 'episodes'
            size = client.train_local(episodes=local_episodes)

            weights = client.get_weights()
            wts.append(weights)
            sizes.append(size)

        # Aggregate federated weights
        server.aggregate(wts, sizes)

        # Evaluate efficiently
        score = evaluate_model(global_model, episodes=eval_episodes)
        history.append(score)
        round_pbar.set_postfix({"Global Reward": f"{score:.3f}"})

    return global_model, history


# -----------------------------
# Full feasibility sweep
# -----------------------------
def feasibility_sweep():
    results = []

    # Loop over patient counts, architectures, hyperparams, reward functions
    for patients in config.PATIENT_COUNTS:
        for arch in config.ARCHITECTURES:
            for hp in config.HYPERPARAMS:
                for rw in config.REWARD_SCENARIOS:
                    print(
                        f"\nRunning: Patients={patients}, Arch={arch}, LR={hp['LR']}, "
                        f"Batch={hp['BATCH_SIZE']}, Epochs={hp['LOCAL_EPOCHS']}, Reward={rw}"
                    )

                    # UPDATED: Pass hp['LOCAL_EPOCHS'] to local_episodes
                    model, history = run_custom_fed(
                        hidden_dims=arch,
                        patients=patients,
                        hyperparams=hp,
                        reward=rw,
                        # ---------------------------------------------------------
                        # FIXED: Hardcoded to 50 so it runs long enough
                        # Change this number if you want more/fewer rounds
                        rounds=config.FED_ROUNDS,
                        # ---------------------------------------------------------
                        max_env_steps=config.MAX_LOCAL_STEPS,
                        local_episodes=hp["LOCAL_EPOCHS"],
                        eval_episodes=config.EVAL_EPISODES,
                    )

                    final_reward = history[-1]
                    results.append(
                        [
                            patients,
                            arch[0],
                            arch[1],
                            hp["LR"],
                            hp["BATCH_SIZE"],
                            hp["LOCAL_EPOCHS"],
                            rw["w1"],
                            rw["w2"],
                            rw["w3"],
                            final_reward,
                        ]
                    )

                    # Save after each sweep to avoid losing results
                    save_results(
                        "feasibility_results.csv",
                        [
                            "Patients",
                            "H1",
                            "H2",
                            "LR",
                            "Batch",
                            "Epochs",
                            "W1",
                            "W2",
                            "W3",
                            "FinalReward",
                        ],
                        results,
                    )
    print("✅ Feasibility study complete. Results saved to feasibility_results.csv")
    return results


# -----------------------------
# Run if script is executed
# -----------------------------
if __name__ == "__main__":
    feasibility_sweep()
