import matplotlib.pyplot as plt


def plot_rewards(history):
    plt.figure()
    plt.plot(history, marker="o")
    plt.title("Federated Training")
    plt.xlabel("Round")
    plt.ylabel("Median Reward")
    plt.grid()
    plt.show()


def plot_bar(results, title):
    labels = [r[0] for r in results]
    values = [r[1] for r in results]
    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel("Median Reward")
    plt.tight_layout()
    plt.show()
