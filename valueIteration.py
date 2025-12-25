import random
import numpy as np

class Grid():
        def __init__(self, size=5):
                self.size = size
                self.agent_pos = (0, 0)
                self.target_pos = (4,4)
                self.actions = ["up", "down", "left", "right"]
                self.gamma = 0.9 # Windy Lab's value. This is for improving long term return
                self.epsilon = 1e-6 # Convergence

        def step(self, state, action):
                x, y  = state
                if action == "up":
                        y = max(0, y-1)
                if action == "down":
                        y = min(self.size - 1, y + 1)
                if action == "left":
                        x = max(0, x - 1)
                if action == "right":
                        x = min(self.size - 1, x + 1)
                next_state = (x, y)
                if next_state == self.target_pos:
                        return next_state, 1
                return next_state, 0
        
        def value_iteration(self):
                value_table = np.zeros((self.size, self.size)) # Create a value table
                isConverged = False

                while not isConverged:
                        new_value_table = np.copy(value_table)
                        '''
                        FYI, cells are states
                        Create a new table for updated values of value_table of states/cells
                        '''
                        for i in range(self.size):
                                for j in range(self.size):
                                        state = (i, j)
                                        max_value = float('-inf')
                                        for action in self.actions:
                                                next_state, reward = self.step(state, action)
                                                value = reward + self.gamma * value_table[next_state[0], next_state[1]]
                                                '''
                                                v_{s1} = r_{s1} + γ*v_{next_state}
                                                Bellman simplified
                                                '''
                                                max_value = max(max_value, value)
                                        new_value_table[i, j] = max_value
                        if np.max(np.abs(new_value_table - value_table)) < self.epsilon:
                                isConverged = True
                        value_table = new_value_table
                return value_table


def main():
        gridWorld = Grid()
        value_table = gridWorld.value_iteration()
        print(value_table)

if __name__ == "__main__":
        main()