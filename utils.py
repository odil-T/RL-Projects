import cv2
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image


StateType = Tuple[Tuple[int, int], Tuple[int, int]]
ActionType = List[float]
QTableType = Dict[StateType, ActionType]


class Environment:
    def __init__(
            self,
            BOARD_SIZE: int,
            FOOD_REWARD: int = 1000,
            ENEMY_PENALTY: int = -10000,
            MOVE_PENALTY: int = -10
    ):

        self.BOARD_SIZE = BOARD_SIZE
        self.FOOD_REWARD = FOOD_REWARD
        self.ENEMY_PENALTY = ENEMY_PENALTY
        self.MOVE_PENALTY = MOVE_PENALTY
        self.BLOB_COLORS = {
            'player': (255, 175, 0),
            'food': (100, 255, 100),
            'enemy': (0, 0, 255)
        }

    def reset(self) -> None:
        self.done = False
        blacklisted_coords = []

        self.player = Blob(self.BOARD_SIZE, blacklisted_coords)
        blacklisted_coords.append((self.player.x, self.player.y))

        self.food = Blob(self.BOARD_SIZE, blacklisted_coords)
        blacklisted_coords.append((self.food.x, self.food.y))

        self.enemy = Blob(self.BOARD_SIZE, [(self.player.x, self.player.y), blacklisted_coords])

    def render(self) -> None:
        board_array = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, 3), np.uint8)
        board_array[self.player.y, self.player.x] = self.BLOB_COLORS['player']
        board_array[self.food.y, self.food.x] = self.BLOB_COLORS['food']
        board_array[self.enemy.y, self.enemy.x] = self.BLOB_COLORS['enemy']

        image = Image.fromarray(board_array)
        image = np.array(image.resize((300, 300)))

        cv2.imshow('Game', image)
        cv2.waitKey(10)

    def get_state(self) -> StateType:
        return self.player - self.food, self.player - self.enemy

    def step(self, action: int) -> Tuple[StateType, int, bool]:
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
    def __init__(self, BOARD_SIZE: int, blacklisted_coords: List[Tuple[int, int]]):
        self.BOARD_SIZE = BOARD_SIZE
        self.ACTION_TO_DISPLACEMENT = {  # action choice to x, y displacement mapping
            0: (1, 0),
            1: (-1, 0),
            2: (0, 1),
            3: (0, -1),
            4: (1, 1),
            5: (-1, 1),
            6: (1, -1),
            7: (-1, -1),
            8: (0, 0)
        }

        while True:
            self.x = np.random.randint(0, BOARD_SIZE)
            self.y = np.random.randint(0, BOARD_SIZE)

            if (self.x, self.y) not in blacklisted_coords:
                break

    def __sub__(self, other: 'Blob') -> Tuple[int, int]:
        return self.x - other.x, self.y - other.y

    def move(self, choice: int) -> bool:
        x_d, y_d = self.ACTION_TO_DISPLACEMENT[choice]
        self.x += x_d
        self.y += y_d

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
