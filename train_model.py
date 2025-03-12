"""
TODO:
1. add type hints
2. divide the file into main functions and place it in name == main. ask gemini what the best way to divide our code
5. add play again? feature
5. save to github 'reinforcement learning' repo subfolder of some sort. add readme, installation instructions
"""
import os.path
import pickle
import datetime
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import Environment


BOARD_SIZE = 10

HM_EPISODES = 50000
SHOW_EVERY = 3000

epsilon = 0.95
EPSILON_DECAY = 0.999
EPSILON_GRACE_PERIOD = 5000
EPSILON_MINIMUM = 0.15

LEARNING_RATE = 0.1
DISCOUNT = 0.95

q_table_start = None  # specify Q-Table file to resume training
results_folder = 'results'

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

if not os.path.exists(results_folder):
    print('Creating the results folder...')
    os.makedirs(results_folder)


env = Environment(BOARD_SIZE)
show = False
episode_rewards = []

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

    if episode > EPSILON_GRACE_PERIOD and epsilon > EPSILON_MINIMUM:
        epsilon *= EPSILON_DECAY
    if epsilon < EPSILON_MINIMUM:
        epsilon = EPSILON_MINIMUM

    episode_rewards.append(episode_reward)

    if show:
        cv2.destroyAllWindows()
        show = False

# Saving the Q-Table and chart
current_time = f'{datetime.datetime.now()}'.replace(':', '-')
folder_name = f'{results_folder}/experiment ' + current_time
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