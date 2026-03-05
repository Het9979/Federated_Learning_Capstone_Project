import copy
import torch
import numpy as np
from tqdm import tqdm
from env import RehabEnv
from model import DuelingDQN
from trainer import DQNTrainer
import config


class FLClient:
    # NOW ACCEPTS HIDDEN_DIMS
    def __init__(self, variant, hidden_dims=(128, 256)):
        self.env = RehabEnv(variant)

        # Pass the dimensions to the model
        self.model = DuelingDQN().to(config.DEVICE)
        self.trainer = DQNTrainer(self.model)

        # --- Exploration Strategy ---
        self.epsilon = config.EPS_START
        self.epsilon_min = config.EPS_END
        self.epsilon_decay = config.EPS_DECAY

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def train_local(self, episodes=50):
        total_steps_in_round = 0

        pbar = tqdm(range(episodes), desc=f"[Client {self.env.variant}]", leave=False)

        for ep in pbar:
            s = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(config.ACTION_DIM)
                else:
                    with torch.no_grad():
                        s_tensor = torch.FloatTensor(s).unsqueeze(0).to(config.DEVICE)
                        q = self.model(s_tensor)
                        a = torch.argmax(q).item()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                ns, r, done = self.env.step(a)
                self.trainer.buffer.add(s, a, r, ns, done)
                self.trainer.train_step()

                s = ns
                ep_reward += r
                total_steps_in_round += 1

            pbar.set_postfix({"R": f"{ep_reward:.2f}", "Eps": f"{self.epsilon:.2f}"})

        self.trainer.update_target()
        return total_steps_in_round


class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_weights, client_sizes):
        new_w = copy.deepcopy(client_weights[0])
        total_data_points = sum(client_sizes)
        for key in new_w.keys():
            new_w[key] = sum(
                client_weights[i][key] * (client_sizes[i] / total_data_points)
                for i in range(len(client_weights))
            )
        self.global_model.load_state_dict(new_w)


class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_weights, client_sizes):
        # Initialize with the first client's weights structure
        new_w = copy.deepcopy(client_weights[0])

        total_data_points = sum(client_sizes)

        # Weighted Average Aggregation (FedAvg)
        for key in new_w.keys():
            new_w[key] = sum(
                client_weights[i][key] * (client_sizes[i] / total_data_points)
                for i in range(len(client_weights))
            )

        self.global_model.load_state_dict(new_w)
