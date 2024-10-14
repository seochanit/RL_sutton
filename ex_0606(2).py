import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

class CliffWalking():
    WORLD_WIDTH = 12
    WORLD_HEIGHT = 4
    START = [3,0]
    GOAL = [3,11]
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTIONS = [UP, DOWN, LEFT, RIGHT]
    CLIFF = [[3, y] for y in range(1,11)]

class Config():
    ALPHA = 0.4
    EPSILON = 0.1
    GAMMA = 1

def step(state, action, env=CliffWalking()):
    x, y = state
    if action == env.UP:
        next_state = [max(0, x-1), y]
    elif action == env.DOWN:
        next_state = [min(env.WORLD_HEIGHT-1, x+1), y]
    elif action == env.LEFT:
        next_state = [x, max(0,y-1)]
    elif action == env.RIGHT:
        next_state = [x, min(env.WORLD_WIDTH-1, y+1)]
    
    reward = -1
    if next_state in env.CLIFF:
        next_state = env.START
        reward = -10
    return next_state, reward

def select_action(state, q_value, env=CliffWalking(), conf=Config()):
    if np.random.uniform() < conf.EPSILON:
        return np.random.choice(env.ACTIONS)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([env.ACTIONS[action] for action, value in enumerate(values) if value == np.max(values)])

def sarsa(q_value, env=CliffWalking, conf=Config()):
    state = env.START
    action = select_action(state, q_value)
    rewards = 0
    path = [state]
    while True:
        next_state, reward = step(state, action)
        rewards += reward
        next_action = select_action(next_state, q_value)
        sarsa_tgt = reward + conf.GAMMA*(q_value[next_state[0], next_state[1], next_action])
        q_value[state[0], state[1], action] += conf.ALPHA * (sarsa_tgt - q_value[state[0], state[1], action])
        path.append(next_state)
        if next_state == env.GOAL:
            break
        state,action = next_state, next_action
    return rewards, path

def qlearning(q_value, env=CliffWalking, conf=Config()):
    state = env.START
    rewards = 0
    path=[state]
    while True:
        action = select_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        q_tgt = reward + conf.GAMMA*np.max(q_value[next_state[0], next_state[1],:])
        q_value[state[0], state[1], action] += conf.ALPHA*(q_tgt - q_value[state[0], state[1], action])
        path.append(next_state)
        if next_state == env.GOAL:
            break
        state = next_state
    return rewards, path

def visualize_paths(safer_path, optimal_path, env=CliffWalking()):
    fig, ax = plt.subplots()

    # 그리드 그리기 (0,0 부터 시작하는 방식, 네모 칸이 좌표가 되도록)
    for x in range(env.WORLD_WIDTH + 1):
        ax.plot([x, x], [0, env.WORLD_HEIGHT], color="black")
    for y in range(env.WORLD_HEIGHT + 1):
        ax.plot([0, env.WORLD_WIDTH], [y, y], color="black")

    # 절벽 그리기 (절벽이 0번째 줄에 표시)
    for cliff in env.CLIFF:
        ax.fill_between([cliff[1], cliff[1]+1], cliff[0], cliff[0]+1, color='gray')

    # 경로 그리기 (경로 좌표를 그대로 사용)
    safer_path_x, safer_path_y = zip(*[(s[1]+0.5, s[0]+0.5) for s in safer_path])
    optimal_path_x, optimal_path_y = zip(*[(s[1]+0.5, s[0]+0.5) for s in optimal_path])
    

    # 경로 표시
    ax.plot(safer_path_x, safer_path_y, color='blue', label="Safer path", marker='o')
    ax.plot(optimal_path_x, optimal_path_y, color='red', label="Optimal path", marker='o')

    # 시작과 목표 표시 (각 칸이 좌표로 사용됨)
    ax.text(env.START[1]+0.5, env.START[0]+0.5, 'S', fontsize=12, ha='center', va='center', color='white', bbox=dict(facecolor='black', edgecolor='none'))
    ax.text(env.GOAL[1]+0.5, env.GOAL[0]+0.5, 'G', fontsize=12, ha='center', va='center', color='white', bbox=dict(facecolor='black', edgecolor='none'))

    # 축 설정 (좌표가 각 네모 칸을 대표)
    ax.set_xticks(np.arange(env.WORLD_WIDTH + 1))
    ax.set_yticks(np.arange(env.WORLD_HEIGHT + 1))
    ax.set_xticklabels(np.arange(env.WORLD_WIDTH + 1))
    ax.set_yticklabels(np.arange(env.WORLD_HEIGHT + 1))
    
    plt.xlim([0, env.WORLD_WIDTH])
    plt.ylim([0, env.WORLD_HEIGHT])
    plt.gca().invert_yaxis()  # 좌표가 아래에서 위로 증가하도록 반전
    plt.legend()
    plt.grid(True)  # 각 칸이 좌표로 구분되도록 그리드 표시
    plt.show()


def experiment(env=CliffWalking()):
    num_epi = 500
    num_run = 50

    rewards_sarsa = np.zeros(num_epi)
    rewards_q = np.zeros(num_epi)

    for _ in tqdm(range(num_run)):
        q_sarsa = np.zeros((env.WORLD_HEIGHT, env.WORLD_WIDTH, 4))
        q_q = deepcopy(q_sarsa)
        for i in range(num_epi):
            reward_sarsa, safer_path = sarsa(q_sarsa)
            reward_q, optimal_path = qlearning(q_q)
            rewards_sarsa[i] += reward_sarsa
            rewards_q[i] += reward_q
    rewards_sarsa /= num_run
    rewards_q /= num_run
    print(safer_path)

    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()

    visualize_paths(safer_path, optimal_path)

if __name__ == "__main__":
    experiment()