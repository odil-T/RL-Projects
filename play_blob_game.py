import pickle
import cv2
import numpy as np
from PIL import Image


BOARD_SIZE = 10
epsilon = 0.2

BUTTON2ACTION = {c: i for i, c in zip(range(4), 'dasw')}


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
            return True
        if self.x < 0:
            self.x = 0
            return True
        if self.y >= self.BOARD_SIZE:
            self.y = self.BOARD_SIZE - 1
            return True
        if self.y < 0:
            self.y = 0
            return True

        return False


q_table_file = '2025-02-27 22-03-43.478093'
with open(q_table_file, 'rb') as f:
    q_table = pickle.load(f)

env = Environment(BOARD_SIZE)
done = False
env.reset()

while True:
    env.render()

    if done:
        cv2.waitKey(1000)
        break

    # Enemy's (user) turn
    key = chr(cv2.waitKey(0))
    if key == 'q':
        break
    elif key in 'wasd':
        is_OOB = env.enemy.move(BUTTON2ACTION[key])
        if is_OOB:
            continue

    # check here if you are on top of player. break if so

    state = env.get_state()

    if epsilon < np.random.uniform():
        action = np.argmax(q_table[state])
    else:
        action = np.random.randint(0, 4)

    _, _, done = env.step(action)


cv2.destroyAllWindows()