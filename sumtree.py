import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
from collections import deque


# --------------------------------------
# 1. SumTree for Efficient Priority Sampling
# --------------------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree array stores priorities
        self.tree = np.zeros(2 * capacity - 1)
        # Data array stores actual transitions
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]


# --------------------------------------
# 2. Prioritized Replay Buffer with N-Step Logic
# --------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_step=3, gamma=0.99):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.epsilon = 0.01  # Small constant to ensure non-zero priority

        # N-step learning
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, state, action, reward, next_state, done):
        # Store raw transition in temporary n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step:
            return  # Wait until we have N steps

        # Calculate N-step return
        R, next_s, d = self._get_n_step_info()
        currentState, currentAction = self.n_step_buffer[0][:2]

        # Store processed transition with max priority (so it gets trained on at least once)
        max_p = np.max(self.tree.tree[-self.capacity :])
        if max_p == 0:
            max_p = 1.0

        self.tree.add(max_p, (currentState, currentAction, R, next_s, d))

    def _get_n_step_info(self):
        # Calculate discounted return over N steps
        R, next_s, done = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, transition in enumerate(reversed(self.n_step_buffer)):
            r, n_s, d = transition[2], transition[3], transition[4]
            R = r + self.gamma * R * (1 - d)
            if d:  # If episode ended inside the window
                next_s = n_s
                done = True
        return R, next_s, done

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        # Stratified sampling based on priorities
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # Calculate Importance Sampling (IS) weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # Normalize weights

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(idxs),
            torch.FloatTensor(is_weights).to(config.DEVICE),
        )

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.count


# --------------------------------------
# 3. DQNTrainer (Handles Double DQN & PER Updates)
# --------------------------------------
class DQNTrainer:
    def __init__(self, model):
        self.model = model
        self.target_model = type(model)()  # Clone architecture
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(config.DEVICE)
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)

        # Initialize Prioritized Buffer with N-step
        self.buffer = PrioritizedReplayBuffer(
            capacity=(
                config.BUFFER_CAPACITY if hasattr(config, "BUFFER_CAPACITY") else 10000
            ),
            n_step=3,  # 3-step returns
            gamma=config.GAMMA,
        )

        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.n_step_gamma = config.GAMMA**3  # Discount factor for N-step
        self.beta = 0.4  # Importance sampling starting value

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        # Anneal beta towards 1.0 (for unbiased updates at end of training)
        self.beta = min(1.0, self.beta + 1e-4)

        # Sample from Prioritized Buffer
        states, actions, rewards, next_states, dones, idxs, weights = (
            self.buffer.sample(self.batch_size, self.beta)
        )

        state_batch = torch.FloatTensor(states).to(config.DEVICE)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(config.DEVICE)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(config.DEVICE)
        next_state_batch = torch.FloatTensor(next_states).to(config.DEVICE)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(config.DEVICE)

        # --- Double DQN Logic ---
        # 1. Select best action using Online Model
        with torch.no_grad():
            next_actions = self.model(next_state_batch).argmax(1, keepdim=True)
            # 2. Evaluate that action using Target Model
            next_q_values = self.target_model(next_state_batch).gather(1, next_actions)

        # Compute Target Q (using N-step gamma)
        target_q = reward_batch + (1 - done_batch) * self.n_step_gamma * next_q_values

        # Compute Current Q
        current_q = self.model(state_batch).gather(1, action_batch)

        # Compute TD Errors (Absolute difference for priority update)
        errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        self.buffer.update_priorities(idxs, errors)

        # Compute Loss: MSE weighted by Importance Sampling weights
        loss = (weights * (current_q - target_q).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping (stabilizes training)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
