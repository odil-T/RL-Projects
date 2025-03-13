import pickle
import cv2
import numpy as np
from utils import Environment


BOARD_SIZE = 10
epsilon = 0.2

BUTTON2ACTION = {c: i for i, c in zip(range(4), 'dasw')}


q_table_file = 'results/experiment 2025-03-13 23-02-52.692282/model 2025-03-13 23-02-52.692282'
with open(q_table_file, 'rb') as f:
    q_table = pickle.load(f)

env = Environment(BOARD_SIZE)
done = False
env.reset()

while True:
    while True:
        if done:
            cv2.waitKey(1000)
            break

        env.render()

        # Enemy's (user) turn
        key = chr(cv2.waitKey(0))
        if key == 'q':
            break
        elif key in 'wasd':
            is_OOB = env.enemy.move(BUTTON2ACTION[key])
            if is_OOB:
                continue
        else:
            continue

        state = env.get_state()

        # When the user gets the agent  # TODO done value is being overwritten downstream
        if state[1] == (0, 0):
            print('got him!')
            done = True

        if epsilon < np.random.uniform():
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, 4)

        _, _, done = env.step(action)

    user_input = input('Play again? (Y/n): ').strip().lower()
    if user_input == 'y' or user_input == '':
        pass
    elif user_input == 'n':
        break
    else:
        print("Please enter 'y' or 'n'.")


cv2.destroyAllWindows()