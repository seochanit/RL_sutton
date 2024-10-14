import numpy as np
import random
import matplotlib.pyplot as plt


class CliffWalking:
    def __init__(self, width: int = 12, height: int = 4) -> None:
        self.width = width
        self.height = height
        self.start: tuple[int, int] = (self.height - 1, 0)
        self.goal: tuple[int, int] = (self.height - 1, self.width - 1)
        self.cliff: list[tuple[int, int]] = [(self.height - 1, i) for i in range(1, self.width - 1)]
        self.x: int
        self.y: int
        self.reset()

    def step(self, action: int) -> tuple[tuple[int, int], int, bool]:
        if action == 0:  # Up
            self.x = max(self.x - 1, 0)
        elif action == 1:  # Down
            self.x = min(self.x + 1, self.height - 1)
        elif action == 2:  # Right
            self.y = min(self.y + 1, self.width - 1)
        elif action == 3:  # Left
            self.y = max(self.y - 1, 0)

        s_next: tuple[int, int] = (self.x, self.y)
        reward: int = -1

        if s_next in self.cliff:
            reward = -100
            self.x, self.y = self.start  # cliff로 가면 다시 원점

        done: bool = self.is_done()

        return (self.x, self.y), reward, done

    def is_done(self) -> bool:
        return (self.x, self.y) == self.goal

    def reset(self) -> tuple[int, int]:
        self.x, self.y = self.start
        return (self.x, self.y)


class Agent:
    def __init__(self, width: int = 12, height: int = 4, action: int = 4) -> None:
        self.q_s_a: np.ndarray = np.zeros((height, width, action))
        self.eps: float = 0.1

    def select_action(self, s: tuple[int, int]) -> int:
        x, y = s
        if random.uniform(0, 1) < self.eps:
            action = random.randint(0, 3)
        else:
            action = int(np.argmax(self.q_s_a[x, y, :]))
        return action

    def update_table(self, transition: tuple[tuple[int, int], int, int, tuple[int, int]]) -> None:
        pass


class Sarsa(Agent):
    def update_table(self, transition: tuple[tuple[int, int], int, int, tuple[int, int]]) -> None:
        s, a, r, s_next = transition
        x, y = s
        next_x, next_y = s_next

        a_next = self.select_action(s_next)

        self.q_s_a[x, y, a] += 0.1 * (
            r + self.q_s_a[next_x, next_y, a_next] - self.q_s_a[x, y, a]
        )


class Qlearning(Agent):
    def update_table(self, transition: tuple[tuple[int, int], int, int, tuple[int, int]]) -> None:
        s, a, r, s_next = transition
        x, y = s
        next_x, next_y = s_next

        self.q_s_a[x, y, a] += 0.1 * (
            r + np.max(self.q_s_a[next_x, next_y, :]) - self.q_s_a[x, y, a]
        )


num_epi: int = 500
num_run: int = 100
returns_sarsa: np.ndarray = np.zeros((num_run, num_epi))
returns_q: np.ndarray = np.zeros((num_run, num_epi))

for run_i in range(num_run):
    env = CliffWalking()
    agent_sarsa = Sarsa()
    agent_q = Qlearning()
    for epi_i in range(num_epi):
        """sarsa"""
        done_sarsa: bool = False
        s = env.reset()
        total_reward_sarsa: int = 0

        while not done_sarsa:
            a = agent_sarsa.select_action(s)
            s_next, r, done_sarsa = env.step(a)
            agent_sarsa.update_table((s, a, r, s_next))
            s = s_next
            total_reward_sarsa += r
        returns_sarsa[run_i, epi_i] += total_reward_sarsa

        """Q_leaning"""
        done_q: bool = False
        s = env.reset()
        total_reward_q: int = 0

        while not done_q:
            a = agent_q.select_action(s)
            s_next, r, done_q = env.step(a)
            agent_q.update_table((s, a, r, s_next))
            s = s_next
            total_reward_q += r
        returns_q[run_i, epi_i] += total_reward_q
        
mean_returns_sarsa = np.mean(returns_sarsa, axis=0)
mean_returns_q = np.mean(returns_q, axis=0)

plt.figure(figsize=(16, 9))
plt.plot(mean_returns_sarsa, label="Sarsa")
plt.plot(mean_returns_q, label="Q-learning")
plt.xlabel("Episodes")
plt.ylabel("Sum of rewards during episode")
plt.ylim([-300, 0])
plt.legend()
plt.grid(True)
plt.show()
