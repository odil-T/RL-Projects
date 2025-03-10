"""
TODO:
1. add models and their metric charts to separate folders
2. add minimum epsilon
3. save classes to utils
4. fix initialization of same location of blobs
5. add play again? feature
5. save to github 'reinforcement learning' repo subfolder of some sort. add readme, installation instructions
"""

import pickle
import datetime
import cv2
import numpy as np
from PIL import Image


BOARD_SIZE = 10
HM_EPISODES = 50000
SHOW_EVERY = 3000

epsilon = 0.95
EPSILON_DECAY = 0.999
EPSILON_GRACE_PERIOD = 10000
LEARNING_RATE = 0.1
DISCOUNT = 0.95


class Environment:
    def __init__(self, BOARD_SIZE, FOOD_REWARD=1000, ENEMY_PENALTY=-10000, MOVE_PENALTY=-10):
        self.BOARD_SIZE = BOARD_SIZE
        self.FOOD_REWARD = FOOD_REWARD
        self.ENEMY_PENALTY = ENEMY_PENALTY
        self.MOVE_PENALTY = MOVE_PENALTY
        self.BLOB_COLORS = {
            'player': (255, 175, 0),
            'food': (100, 255, 100),
            'enemy': (0, 0, 255)
        }

    def reset(self):
        self.done = False
        self.player = Blob(self.BOARD_SIZE)
        self.food = Blob(self.BOARD_SIZE)
        self.enemy = Blob(self.BOARD_SIZE)

    def render(self):
        board_array = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, 3), np.uint8)
        board_array[self.player.y, self.player.x] = self.BLOB_COLORS['player']
        board_array[self.food.y, self.food.x] = self.BLOB_COLORS['food']
        board_array[self.enemy.y, self.enemy.x] = self.BLOB_COLORS['enemy']
        image = Image.fromarray(board_array)
        image = np.array(image.resize((300, 300)))

        cv2.imshow('Game', image)
        cv2.waitKey(10)

    def get_state(self):
        return self.player - self.food, self.player - self.enemy

    def step(self, action):
        self.player.move(action)
        new_state = self.get_state()

        if self.player.x == self.food.x and self.player.y == self.food.y:
            reward = self.FOOD_REWARD
            self.done = True
        elif self.player.x == self.enemy.x and self.player.y == self.enemy.y:
            reward = self.ENEMY_PENALTY
            self.done = True
        else:
            reward = self.MOVE_PENALTY

        return new_state, reward, self.done


class Blob():
    def __init__(self, BOARD_SIZE):
        self.BOARD_SIZE = BOARD_SIZE
        self.x = np.random.randint(0, BOARD_SIZE)
        self.y = np.random.randint(0, BOARD_SIZE)

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def move(self, choice):
        if choice == 0:
            self.x += 1
        elif choice == 1:
            self.x -= 1
        elif choice == 2:
            self.y += 1
        elif choice == 3:
            self.y -= 1

        if self.x >= self.BOARD_SIZE:
            self.x = self.BOARD_SIZE - 1
        if self.x < 0:
            self.x = 0
        if self.y >= self.BOARD_SIZE:
            self.y = self.BOARD_SIZE - 1
        if self.y < 0:
            self.y = 0


# Initialize the Q-Table
q_table = {}
for x1 in range(-BOARD_SIZE + 1, BOARD_SIZE):
    for y1 in range(-BOARD_SIZE + 1, BOARD_SIZE):
        for x2 in range(-BOARD_SIZE + 1, BOARD_SIZE):
            for y2 in range(-BOARD_SIZE + 1, BOARD_SIZE):
                q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-2, 0) for i in range(4)]


env = Environment(BOARD_SIZE)
episode_rewards = []
show = False

for episode in range(HM_EPISODES):
    if not episode % SHOW_EVERY:
        show = True
        print(f'epsilon: {epsilon} | mean_cumulative_episode_reward: {np.mean(episode_rewards)}')

    env.reset()
    episode_reward = 0

    for i in range(100):
        if show:
            env.render()

        state = env.get_state()

        if epsilon < np.random.uniform():
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, 4)

        new_state, reward, done = env.step(action)

        # Obtaining Q-values for training the Q-Table
        current_q = q_table[state][action]
        max_future_q = np.max(q_table[new_state])
        if reward == env.FOOD_REWARD:
            new_q = env.FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + max_future_q * DISCOUNT)

        q_table[state][action] = new_q
        episode_reward += reward

        if done:
            break

    if episode > EPSILON_GRACE_PERIOD:
        epsilon *= EPSILON_DECAY

    episode_rewards.append(episode_reward)

    if show:
        cv2.destroyAllWindows()
        show = False

# Saving the Q-Table
filename = f'{datetime.datetime.now()}'.replace(':', '-')
with open(filename, 'wb') as f:
    pickle.dump(q_table, f)
