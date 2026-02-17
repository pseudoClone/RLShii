import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# DQN Model
# --------------------------------------------------
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

# --------------------------------------------------
# Replay Memory
# --------------------------------------------------
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# --------------------------------------------------
# FrozenLake DQL
# --------------------------------------------------
class FrozenLakeDQL:
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32

    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['L', 'D', 'R', 'U']

    # --------------------------------------------------
    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        x = torch.zeros(num_states, device=device)
        x[state] = 1.0
        return x

    # --------------------------------------------------
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target = torch.tensor([reward], device=device)
            else:
                with torch.no_grad():
                    target = torch.tensor(
                        reward + self.discount_factor_g *
                        target_dqn(
                            self.state_to_dqn_input(new_state, num_states)
                        ).max(),
                        device=device
                    )

            current_q = policy_dqn(
                self.state_to_dqn_input(state, num_states)
            )
            current_q_list.append(current_q)

            target_q = target_dqn(
                self.state_to_dqn_input(state, num_states)
            ).clone().detach()

            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(
            torch.stack(current_q_list),
            torch.stack(target_q_list)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --------------------------------------------------
    def train(self, episodes, render=False, is_slippery=False):
        env = gym.make(
            'FrozenLake-v1',
            map_name="4x4",
            is_slippery=is_slippery,
            render_mode='human' if render else None
        )

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1.0
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_states, num_states, num_actions).to(device)
        target_dqn = DQN(num_states, num_states, num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), lr=self.learning_rate_a
        )

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(
                            self.state_to_dqn_input(state, num_states)
                        ).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

            if reward == 1:
                rewards_per_episode[i] = 1

            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()

        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

        plt.figure(figsize=(10, 4))
        sum_rewards = np.array([
            np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
            for x in range(episodes)
        ])

        plt.subplot(1, 2, 1)
        plt.plot(sum_rewards)
        plt.title("Rewards (100-episode window)")

        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")

        plt.savefig("frozen_lake_dql.png")

    # --------------------------------------------------
    def test(self, episodes, is_slippery=False):
        env = gym.make(
            'FrozenLake-v1',
            map_name="4x4",
            is_slippery=is_slippery,
            render_mode='human'
        )

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_states, num_actions).to(device)
        policy_dqn.load_state_dict(
            torch.load("frozen_lake_dql.pt", map_location=device)
        )
        policy_dqn.eval()

        for _ in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(
                        self.state_to_dqn_input(state, num_states)
                    ).argmax().item()

                state, _, terminated, truncated, _ = env.step(action)

        env.close()

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    frozen_lake = FrozenLakeDQL()
    frozen_lake.train(1000, is_slippery=False)
    frozen_lake.test(10, is_slippery=False)