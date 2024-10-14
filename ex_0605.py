from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

@dataclass
class Environment:
    WORLD_WIDTH = 10
    WORLD_HEIGHT = 7
    START = [3, 0]
    GOAL = [3, 7]
    WIND = [0,0,0,1,1,1,2,2,1,0]
    ACTIONS = [0,1,2,3]

@dataclass
class Config:
    ALPHA = 0.5
    GAMMA = 1
    EPSILON = 0.1

def step(state, action, env=Environment()):
    x, y = state
    reward = -1
    if action == 0: #up
        next_state = [max(x - 1 - env.WIND[y], 0), y]
    elif action == 1: #down
        next_state = [max(min(x + 1 - env.WIND[y], env.WORLD_HEIGHT - 1), 0), y]
    elif action == 2: #right
        next_state = [max(x - env.WIND[y], 0), min(y + 1, env.WORLD_WIDTH - 1)]
    elif action == 3: #left
        next_state = [max(x - env.WIND[y], 0), max(y - 1, 0)]
    return reward, next_state

def select_action(state, q_value, env=Environment(), conf=Config()):
    if random.uniform(0,1) < conf.EPSILON:
        return np.random.choice(env.ACTIONS)
    else:
        values = q_value[state[0],state[1],:]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

def run_episode(q, env=Environment(), conf=Config()):
    state = env.START
    action = select_action(state, q)
    time = 0
    while state != env.GOAL:
        reward, next_state = step(state, action)
        next_action = select_action(next_state , q)
        TD_tgt = reward + conf.GAMMA*q[next_state[0], next_state[1], next_action]
        q[state[0], state[1], action] += conf.ALPHA * (TD_tgt - q[state[0], state[1], action])
        state, action = next_state, next_action
        time += 1
    return time

def main(env=Environment()):
    q_value = np.zeros(((env.WORLD_HEIGHT, env.WORLD_WIDTH, len(env.ACTIONS))))
    max_epi = 500
    
    steps=[]
    for _ in tqdm(range(max_epi)):
        steps.append(run_episode(q_value))
    steps = np.add.accumulate(steps)
    
    plt.plot(steps, np.arange(1, len(steps)+1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    main()