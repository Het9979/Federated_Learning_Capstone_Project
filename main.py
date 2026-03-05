from baselines import (
    evaluate_local_model,
    immediate_best,
    random_policy,
    run_expert_baseline,
    two_best,
)
from federated import FLClient, FLServer
from model import DQN
from evaluate import evaluate_model
from baselines import *
from analysis import *
import config
from plots import plot_rewards


def run_federated():
    clients = [FLClient(v) for v in config.CLIENT_VARIANTS]
    global_model = DQN()
    server = FLServer(global_model)

    history = []

    for r in range(config.FED_ROUNDS):
        print(f"\n=== Federated Round {r} ===")
        wts, sizes = [], []
        for client in clients:
            client.set_weights(global_model.state_dict())
            size = client.train_local()
            wts.append(client.get_weights())
            sizes.append(size)

        server.aggregate(wts, sizes)
        score = evaluate_model(global_model)
        history.append(score)
        print(f"Global Median Reward: {score:.3f}")

    plot_rewards(history)
    return global_model, clients


if __name__ == "__main__":
    global_model, clients = run_federated()

    print("\n===== BASELINES =====")
    print("Expert Oracle:", run_expert_baseline())
    print("Random:", random_policy())
    print("Immediate Best:", immediate_best())
    print("Two Best:", two_best())

    print("\n===== LOCAL MODELS =====")
    for i, c in enumerate(config.CLIENT_VARIANTS):
        score = evaluate_local_model(clients[i].model, c)
        print(f"Local Model {c}: {score}")
