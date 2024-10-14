import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Environment:
    def __init__(self) -> None:
        ''' 0 : terminal, 1 : B, 2 : A, 3 : terminal'''
        self.state: int = 2

    def step(self, action: int) -> tuple[int, float, bool]:
        if self.state == 2:  # A
            if action == 0:  # right
                return 3, 0, True 
            elif action == 1:  
                self.state = 1
                return self.state, 0, False

        elif self.state == 1:  # B
            reward: float = np.random.normal(-0.1, 1)
            return 0, reward, True

    def reset(self) -> int:
        self.state = 2
        return self.state

class Q:
    def __init__(self, alpha: float = 0.1, gamma: int = 1, epsilon: float = 0.5) -> None:
        '''4 states (Left Terminal: 0, B: 1, A: 2, Right Terminal: 3) and 2 actions for A, 10 actions for B'''
        self.q_s_a = [np.zeros(2), np.zeros(10), np.zeros(2), np.zeros(1)] 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, s_now: int) -> int:
        if s_now == 2:
            if random.uniform(0, 1) < self.epsilon:
                return random.choice([0, 1])
            else:
                values = self.q_s_a[s_now]
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])

        elif s_now == 1: 
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(range(10))
            else:
                values = self.q_s_a[s_now]
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])

    def update_table(self, transition: tuple[int, int, float, int]) -> None:
        s_now, a, r, s_next = transition
        self.q_s_a[s_now][a] += self.alpha * (r + self.gamma * np.max(self.q_s_a[s_next]) - self.q_s_a[s_now][a])


class dbl_Q:
    def __init__(self, alpha: float = 0.1, gamma: int = 1, epsilon: float = 0.5) -> None:
        self.q_s_a_1 = [np.zeros(2), np.zeros(10), np.zeros(2), np.zeros(1)] 
        self.q_s_a_2 = [np.zeros(2), np.zeros(10), np.zeros(2), np.zeros(1)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, s_now: int) -> int:
        if s_now == 2: 
            if random.uniform(0, 1) < self.epsilon:
                return random.choice([0, 1])
            else:
                q_value_sum = self.q_s_a_1[s_now] + self.q_s_a_2[s_now] 
                return random.choice([action for action, value in enumerate(q_value_sum) if value == np.max(q_value_sum)])

        elif s_now == 1:  
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(range(10))  
            else:
                q_value_sum = self.q_s_a_1[s_now] + self.q_s_a_2[s_now]
                return random.choice([action for action, value in enumerate(q_value_sum) if value == np.max(q_value_sum)])

    def update_table(self, transition: tuple[int, int, float, int]) -> None:
        s_now, a, r, s_next = transition

        if random.uniform(0, 1) < 0.5:
            self.q_s_a_1[s_now][a] += self.alpha * (r + self.gamma * self.q_s_a_2[s_next][np.argmax(self.q_s_a_1[s_next])] - self.q_s_a_1[s_now][a])
        else:
            self.q_s_a_2[s_now][a] += self.alpha * (r + self.gamma * self.q_s_a_1[s_next][np.argmax(self.q_s_a_2[s_next])] - self.q_s_a_2[s_now][a])



num_epi: int = 300
num_run: int = 10000


q_left_cnt: np.ndarray = np.zeros((num_run, num_epi))
dbl_q_left_cnt: np.ndarray = np.zeros((num_run, num_epi))


for run_i in range(num_run):
    env = Environment()
    q_agent = Q() 
    dbl_q_agent = dbl_Q()
    
    for epi_i in range(num_epi):
        '''Q-learning'''
        s_now = env.reset()
        done_q: bool = False
        q_cnt: int = 0
        
        while not done_q:
            a = q_agent.select_action(s_now)
            if s_now == 2 and a == 1:
                q_cnt += 1
            s_next, r, done_q = env.step(a)
            q_agent.update_table((s_now, a, r, s_next))
            s_now = s_next 
            
        q_left_cnt[run_i, epi_i] = q_cnt 
        
        '''Double Q Learning'''
        s_now = env.reset()
        done_dbl_q: bool = False
        dbl_q_cnt: int = 0
        
        while not done_dbl_q:
            a = dbl_q_agent.select_action(s_now)
            if s_now == 2 and a == 1: 
                dbl_q_cnt += 1
            s_next, r, done_dbl_q = env.step(a)
            dbl_q_agent.update_table((s_now, a, r, s_next))
            s_now = s_next
        dbl_q_left_cnt[run_i, epi_i] = dbl_q_cnt  



q_left_percent = (np.sum(q_left_cnt, axis=0) / num_run) * 100
dbl_q_left_percent = (np.sum(dbl_q_left_cnt, axis=0) / num_run) * 100


plt.figure(figsize=(16, 9))
plt.plot(q_left_percent, label='Q Learning', color='red')
plt.plot(dbl_q_left_percent, label='Double Q Learning', color='green')
plt.xlabel('Episodes')
plt.ylabel('Percentage of left actions at A')
plt.ylim([0, 100])  
plt.xlim([1, 300])
plt.legend()
plt.grid(True)
plt.show()
