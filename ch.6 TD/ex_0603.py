import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

TRUE_VALUES = np.array([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0])
VALUES = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])

class Environment:
    states = np.arange(7) # left_terminal, A, B, C, D, E, right_terminal
    start = 3
    actions = ['left', 'right']
    terminal = [0,6]

class Config:
    ALPHA = 0.01
    GAMMA = 1

def step(state, env=Environment()):
    s_now = state
    if np.random.choice(env.actions) == 'left':
        s_next = s_now - 1
        reward = 0
    else:
        s_next = s_now + 1
        reward = 0 if s_next != 6 else 1
    return s_next, reward

def MC(env=Environment()):
    s_now = env.start
    history = [s_now]
    rewards = [0]
    while True:
        s_next, reward = step(s_now)
        history.append(s_next)
        rewards.append(reward)
        if s_next in env.terminal:
            break
        s_now = s_next
    return history, rewards

def TD(env=Environment()):
    s_now = env.start
    history = [s_now]
    rewards = [0]
    while True:
        s_next, reward = step(s_now)
        history.append(s_next)
        rewards.append(reward)
        if s_next in env.terminal:
            break
        s_now = s_next
    return history, rewards

def batch_update(method, num_epi, num_run, conf=Config()):
    total_error = np.zeros(num_epi)
    for run_i in tqdm(range(num_run)):
        errors = []
        values = VALUES.copy()
        for epi_i in range(num_epi):
            traject=[]
            rewards=[]
            if method == 'TD':
                history, reward = TD()
            else:
                history, reward = MC()
            traject.append(history)
            rewards.append(reward)
            while True:
                new_values = values.copy()
                for traj_, reward_ in zip(traject, rewards):
                    for i in range(0, len(traj_)-1):
                        if method == 'TD':
                            values[traj_[i]] += conf.ALPHA*(reward_[i] + conf.GAMMA * values[traj_[i+1]] - values[traj_[i]])
                        else:
                            values[traj_[i]] += conf.ALPHA*(reward_[i] - values[traj_[i]])
                if np.sum(np.abs(new_values - values)) < 1e-2:
                    break
                values = new_values
            errors.append(np.sqrt(np.sum(np.power(values - TRUE_VALUES, 2)) / 5.0))
        total_error += np.asarray(errors)
    total_error /= num_run
    return total_error

def figure():
    epi = 100
    run = 100
    td_error = batch_update('TD', epi, run)
    mc_error = batch_update('MC', epi, run)
    
    plt.plot(td_error, label='TD')
    plt.plot(mc_error, label='MC')
    plt.title("Batch Training")
    plt.xlabel('Walks/Episodes')
    plt.ylabel('RMS error, averaged over states')
    plt.xlim(0, 100)
    plt.ylim(0, 0.25)
    plt.legend()

    plt.savefig('../images/figure_6_2.png')
    plt.close()

if __name__ == '__main__':
    figure()