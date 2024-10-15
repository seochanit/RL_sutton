import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Envirenment():
    def __init__(self, num_state=19):
        self.num_state = num_state
        self.terminal_left = 0
        self.terminal_right = num_state + 1
        self.reset()

    def reset(self):
        self.state = self.num_state // 2
        return self.state

    def step(self):
        if np.random.uniform(0, 1) < 0.5:
            next_state = self.state - 1
        else:
            next_state = self.state + 1
        reward = self.get_reward(next_state)
        done = True if next_state == self.terminal_left or next_state == self.terminal_right else False
        self.state = next_state
        return next_state, reward, done

    def get_reward(self, state):
        if state == self.terminal_right:
            return 1
        elif state == self.terminal_left:
            return -1
        else:
            return 0

def n_step_TD(env, n, alpha, gamma=1.0, num_episodes=10):
    values = np.zeros(env.num_state + 2)  # 추가된 두 터미널 상태를 포함한 초기 상태 가치
    values[env.terminal_left] = 0.0
    values[env.terminal_right] = 0.0
    true_values = np.arange(-20, 22, 2) / 20.0  # Bellman 방정식을 통한 참값
    true_values[0] = true_values[-1] = 0
    rms_errors = []

    for episode in range(num_episodes):
        # 초기 상태 설정
        state = env.reset()
        states = [state]
        rewards = [0]

        T = float('inf')
        tau = 0
        time = 0

        while tau != T - 1:
            if time < T:
                # 다음 상태로 이동
                next_state, reward, done = env.step()
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = time + 1

            # 업데이트할 상태 결정
            tau = time - n + 1
            if tau >= 0:
                G = 0.0
                # G 계산: n 스텝 내의 보상 합산 및 상태 가치 반영
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma ** n) * values[states[tau + n]]
                state_to_update = states[tau]
                if state_to_update not in [env.terminal_left, env.terminal_right]:
                    
                    values[state_to_update] += alpha * (G - values[state_to_update])

            time += 1

        # 매 에피소드마다 RMS error 계산
        rms_error = compute_rmse(values, true_values)
        rms_errors.append(rms_error)

    return values, np.mean(rms_errors)

def compute_rmse(values, true_values):
    return np.sqrt(np.mean((values - true_values) ** 2))

def run_experiment(env, n, alpha, num_runs=100, num_episodes=10):
    rms_errors = []
    for _ in range(num_runs):
        _, rms_error = n_step_TD(env, n, alpha, num_episodes=num_episodes)
        rms_errors.append(rms_error)
    return np.mean(rms_errors)

# 실험 설정
env = Envirenment()
n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
alpha_values = np.linspace(0, 1, 21)
results = {}

for n in tqdm(n_values):
    rms_errors = []
    for alpha in alpha_values:
        rms_error = run_experiment(env, n, alpha)
        rms_errors.append(rms_error)
    results[n] = rms_errors

# 결과 시각화
plt.figure(figsize=(10, 6))
for n, rms_errors in results.items():
    plt.plot(alpha_values, rms_errors, label=f'n={n}')
plt.xlabel('alpha')
plt.ylabel('Average RMS error')
plt.ylim([0.25, 0.55])
plt.title('Average RMS error over 19 states and first 10 episodes')
plt.legend()
plt.show()
