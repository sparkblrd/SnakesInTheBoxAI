import os
import torch
import torch.nn as nn
import torch.optim as optim


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx]

            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


def save_checkpoint(model, trainer, n_games, record, file_name="checkpoint.pth"):
    folder_path = "./model"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "n_games": n_games,
        "record": record
    }

    torch.save(checkpoint, file_path)


def load_checkpoint(model, trainer, file_name="checkpoint.pth"):
    file_path = os.path.join("./model", file_name)

    if not os.path.exists(file_path):
        print("No checkpoint found, training from zero.")
        return 0, 0

    checkpoint = torch.load(file_path)

    model.load_state_dict(checkpoint["model_state"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

    n_games = checkpoint.get("n_games", 0)
    record = checkpoint.get("record", 0)

    model.train()

    print("Checkpoint loaded:", file_path)
    print("Loaded games:", n_games, "Loaded record:", record)

    return n_games, record