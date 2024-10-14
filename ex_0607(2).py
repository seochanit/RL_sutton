import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

class Environment:
    STATE_A: int = 0
    STATE_B: int = 1
    STATE_TERMINAL: int = 2
    ACTION_A_RIGHT: int = 0
    ACTION_A_LEFT: int = 1
    ACTION_B = range(0,10)
    STATE_ACTIONS = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTION_B]
    TRANSITIONS = [[STATE_TERMINAL, STATE_B], [STATE_TERMINAL] * len(ACTION_B)]
    START = STATE_A

class Config:
    ALPHA: float = 0.1
    GAMMA: int = 1
    epsilon: float = 0.1

def step(state: int, action: int, env = Environment()) -> Tuple[float, int]:
    reward = 0 if state == env.STATE_A else np.random.normal(-0.1, 1)
    next_state = env.TRANSITIONS[state][action]
    return reward, next_state

def select_action(state: int, q_value, env=Environment(), conf=Config()):
    if random.uniform(0, 1) < conf.epsilon:
        return np.random.choice(env.STATE_ACTIONS[state])
    else:
        values = q_value[state]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

def run_episode(q1, q2=None, env = Environment(), conf = Config()):
    state = env.START
    
    left_cnt = 0
    while state != env.STATE_TERMINAL:
        if q2 is None: # q_learning
            action = select_action(state, q1)
        else: # dbl_q_learning
            sum_values = [q_value1 + q_value2 for q_value1, q_value2 in zip(q1, q2)]
            action = select_action(state, sum_values)
        
        if state == env.STATE_A and action == env.ACTION_A_LEFT:
            left_cnt += 1
        
        reward, next_state = step(state, action)
        
        if q2 is None:
            active_q = q1
            target = reward if next_state == env.STATE_TERMINAL else reward + conf.GAMMA * np.max(q1[next_state])
        else:
            if random.uniform(0, 1) < 0.5:
                
                active_q = q1
                target_q = q2
            else:
                active_q = q2
                target_q = q1
            best_action = np.random.choice([action for action, value in enumerate(active_q[next_state]) if value == np.max(active_q[next_state])])
            target = reward if next_state == env.STATE_TERMINAL else reward + conf.GAMMA * target_q[next_state][best_action]
        active_q[state][action] += conf.ALPHA * (target - active_q[state][action])
        state = next_state
    return left_cnt

def run_batches(num_run, num_epi):
    q_left_cnt = np.zeros((num_run, num_epi))
    dbl_left_cnt = np.zeros((num_run, num_epi))
    
    for run_i in tqdm(range(num_run)):
        env = Environment()
        q = [np.zeros(2), np.zeros(len(env.ACTION_B)), np.zeros(1)]
        q1 = [np.zeros(2), np.zeros(len(env.ACTION_B)), np.zeros(1)]
        q2 = [np.zeros(2), np.zeros(len(env.ACTION_B)), np.zeros(1)]
        for epi_i in range(num_epi):
            q_left_cnt[run_i, epi_i] = run_episode(q)
            dbl_left_cnt[run_i, epi_i] = run_episode(q1, q2)
    return q_left_cnt.mean(axis=0), dbl_left_cnt.mean(axis=0)

def plot_chart(q_left_percent, dbl_left_percent, num_epi):
    plt.figure(figsize=(16, 9))
    plt.plot(q_left_percent * 100, label='Q Learning', color='red')
    plt.plot(dbl_left_percent * 100, label='Double Q Learning', color='green')
    plt.plot(np.ones(num_epi) * 5, label='Optimal', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Percentage of left actions at A')
    plt.ylim([0, 100])  
    plt.xlim([1, 300])
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    num_epi, num_run = 300, 10_000
    q_left, dbl_left = run_batches(num_run, num_epi)
    plot_chart(q_left, dbl_left, num_epi)