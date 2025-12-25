import random

class GridWorld:
        def __init__(self, size=5):
                self.size = size
                self.agent_pos = (0,0)
                self.target_pos = (4,4)
                self.actions = ["up", "down", "left", "right"]

        def step(self, action):
                x, y = self.agent_pos
                if action == "up":
                        y = max(0, y-1)
                if action == "down":
                        y = min(self.size - 1, y + 1)
                if action == "left":
                        x = max(0, x - 1)
                if action == "right":
                        x = min(self.size - 1, x + 1)
                
                self.agent_pos = (x, y)
                
                reward = 0
                if self.agent_pos == self.target_pos:
                        reward = 1
                return reward

gridworldInstance = GridWorld()
for _ in range(10):
        action = random.choice(gridworldInstance.actions)
        reward = gridworldInstance.step(action=action)
        print(f"Action: {action} || Reward: {reward}")