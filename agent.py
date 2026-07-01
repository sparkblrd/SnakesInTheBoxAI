import random
import torch
from collections import deque

from model import LinearQNet, QTrainer, save_checkpoint, load_checkpoint
from game import SnakeGameAI, SIZE

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.record = 0

        self.n_games, self.record = load_checkpoint(
            self.model,
            self.trainer
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def get_state(self, game):
        head_x, head_y = game.snake[-1]
        
        point_l = (head_x - SIZE, head_y)
        point_r = (head_x + SIZE, head_y)
        point_u = (head_x, head_y - SIZE)
        point_d = (head_x, head_y + SIZE)

        dir_l = game.direction == "LEFT"
        dir_r = game.direction == "RIGHT"
        dir_u = game.direction == "UP"
        dir_d = game.direction == "DOWN"

        state = [
            # danger straight
            (dir_r and game.isCollision(point_r)) or
            (dir_l and game.isCollision(point_l)) or
            (dir_u and game.isCollision(point_u)) or
            (dir_d and game.isCollision(point_d)),

            # danger right
            (dir_u and game.isCollision(point_r)) or
            (dir_d and game.isCollision(point_l)) or
            (dir_l and game.isCollision(point_u)) or
            (dir_r and game.isCollision(point_d)),

            # danger left
            (dir_d and game.isCollision(point_r)) or
            (dir_u and game.isCollision(point_l)) or
            (dir_r and game.isCollision(point_u)) or
            (dir_l and game.isCollision(point_d)),

            # direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.apple[0] < head_x,
            game.apple[0] > head_x,
            game.apple[1] < head_y,
            game.apple[1] > head_y
        ]

        return [int(x) for x in state]

    def get_action(self, state):
        self.epsilon = max(5, 80 - self.n_games)

        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    game = SnakeGameAI()
    agent = Agent()

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.playStep(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        agent.remember(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        if done:
            agent.train_long_memory()

            game.reset()
            agent.n_games += 1

            if score > agent.record:
                agent.record = score

            save_checkpoint(
                agent.model,
                agent.trainer,
                agent.n_games,
                agent.record
            )

            print(
                "Game:", agent.n_games,
                "Score:", score,
                "Record:", agent.record,
                "Epsilon:", agent.epsilon
            )
            
if __name__ == "__main__":
    train()