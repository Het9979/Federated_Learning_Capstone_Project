import torch
import torch.nn.functional as F
from buffer import ReplayBuffer
import config


class DQNTrainer:
    def __init__(self, model):
        self.model = model
        self.target = type(model)().to(config.DEVICE)
        self.target.load_state_dict(model.state_dict())

        self.opt = torch.optim.Adam(model.parameters(), lr=config.LR)
        self.buffer = ReplayBuffer(config.BUFFER_SIZE)

    def train_step(self):
        if len(self.buffer) < config.BATCH_SIZE:
            return

        s, a, r, ns, d = self.buffer.sample(config.BATCH_SIZE)

        s = torch.FloatTensor(s).to(config.DEVICE)
        a = torch.LongTensor(a).to(config.DEVICE)
        r = torch.FloatTensor(r).to(config.DEVICE)
        ns = torch.FloatTensor(ns).to(config.DEVICE)
        d = torch.FloatTensor(d).to(config.DEVICE)

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target(ns).max(1)[0]
        target = r + config.GAMMA * next_q * (1 - d)

        loss = F.mse_loss(q, target.detach())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
