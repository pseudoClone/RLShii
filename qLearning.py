import gymnasium as gym
import numpy as np
import time

# --------------------
# Hyperparameters
# --------------------
alpha = 0.8        # learning rate
gamma = 0.95       # discount factor
epsilon = 1.0      # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 10000

# --------------------
# Training environment (NO rendering)
# --------------------
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

state_space = env.observation_space.n
action_space = env.action_space.n

Q = np.zeros((state_space, action_space))

# --------------------
# Training loop
# --------------------
for ep in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env.close()

# --------------------
# Testing environment (WITH human rendering)
# --------------------
test_env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="human"
)

state, _ = test_env.reset()
done = False

print("Testing trained agent...\n")
time.sleep(1)

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    time.sleep(0.5)

print("\nReward:", reward)
test_env.close()