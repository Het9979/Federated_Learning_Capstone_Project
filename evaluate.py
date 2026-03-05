import numpy as np
import torch
from env import RehabEnv
import config


def evaluate_model(model, episodes=config.EVAL_EPISODES):
    scores = []
    env = RehabEnv("A")

    model.eval()  # evaluation mode
    with torch.no_grad():
        for _ in range(episodes):
            s = env.reset()
            done = False
            total = 0

            while not done:
                state_tensor = (
                    torch.FloatTensor(s)
                    .unsqueeze(0)  # comment for vanilla DQN uncomment for dueling DQN
                    .to(config.DEVICE)
                )  # 🔥 FIX
                q_vals = model(state_tensor)
                a = torch.argmax(q_vals).item()

                s, r, done = env.step(a)
                total += r

            scores.append(total)

    model.train()  # switch back to training mode
    return np.median(scores)
