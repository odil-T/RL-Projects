import os.path
import pickle
import datetime
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import Environment, QTableType
from typing import Union, Tuple, List


BOARD_SIZE = 10

HM_EPISODES = 500000
SHOW_EVERY = 3000

epsilon = 0.95
EPSILON_DECAY = 0.999
EPSILON_GRACE_PERIOD = 5000
EPSILON_MINIMUM = 0.15

LEARNING_RATE = 0.1
DISCOUNT = 0.95

q_table_start = None  # specify Q-Table file to resume training
results_folder_path = 'results'


def initialize_q_table(q_table_start: Union[None, str]) -> QTableType:
    """Initializes the Q-Table, either from scratch or from a saved file."""

    if q_table_start is None:
        q_table = {}
        for x1 in range(-BOARD_SIZE + 1, BOARD_SIZE):
            for y1 in range(-BOARD_SIZE + 1, BOARD_SIZE):
                for x2 in range(-BOARD_SIZE + 1, BOARD_SIZE):
                    for y2 in range(-BOARD_SIZE + 1, BOARD_SIZE):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-2, 0) for i in range(4)]
    else:
        print('Loading saved Q-Table to resume training...')
        with open(q_table_start, 'rb') as f:
            q_table = pickle.load(f)

    return q_table


def create_results_folder(results_folder_path: str) -> None:
    if not os.path.exists(results_folder_path):
        print('Creating the results folder...')
        os.makedirs(results_folder_path)


def train_agent(q_table: QTableType, env: 'Environment', epsilon: float) -> Tuple[QTableType, List[float]]:
    """Trains the agent for a specified number of episodes."""

    show = False
    episode_rewards = []

    for episode in range(HM_EPISODES):
        if not episode % SHOW_EVERY:
            show = True
            print(f'Episode: {episode} | epsilon: {epsilon} | mean_cumulative_episode_reward: {np.mean(episode_rewards)}')

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

        if episode > EPSILON_GRACE_PERIOD and epsilon > EPSILON_MINIMUM:
            epsilon *= EPSILON_DECAY
        if epsilon < EPSILON_MINIMUM:
            epsilon = EPSILON_MINIMUM

        episode_rewards.append(episode_reward)

        if show:
            cv2.destroyAllWindows()
            show = False

    return q_table, episode_rewards


def save_results(q_table: QTableType, episode_rewards: List[float]) -> None:
    """Saves the trained Q-table and training metrics."""

    current_time = f'{datetime.datetime.now()}'.replace(':', '-')
    folder_name = f'{results_folder_path}/experiment ' + current_time
    model_filename = f'{folder_name}/model {current_time}'
    chart_filename = f'{folder_name}/metrics {current_time}.png'

    print('Saving the Q-Table and training metrics...')
    os.makedirs(folder_name)

    with open(model_filename, 'wb') as f:
        pickle.dump(q_table, f)

    plt.figure(figsize=(10, 6))
    plt.plot(range(HM_EPISODES), episode_rewards)
    plt.title('Episode Reward Change Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig(chart_filename)
    plt.close()


if __name__ == '__main__':
    q_table = initialize_q_table(q_table_start)
    create_results_folder(results_folder_path)

    env = Environment(BOARD_SIZE)
    q_table, episode_rewards = train_agent(q_table, env, epsilon)

    save_results(q_table, episode_rewards)