import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
colors = ["b", "g", "r"]


true_values = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0]
V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]
state = ["L", "A", "B", "C", "D", "E", "R"]

ALPHA = 0.01
GAMMA = 1
num_epi = 100

for epi in range(num_epi):
    s_now = 3  # 'C'

    while s_now not in [0, 6]:
        if np.random.random() < 0.5:
            s_next = s_now - 1  # left
        else:

            s_next = s_now + 1  # right
        reward = 1 if s_next == 6 else 0
        TD_tgt = reward + GAMMA * V[s_next]
        V[s_now] = V[s_now] + ALPHA * (TD_tgt - V[s_now])

        s_now = s_next

    if (epi + 1) in [1, 10, 100]:
        color_index = [1, 10, 100].index(epi + 1)
        plt.plot(
            state[1:6],
            V[1:6],
            marker="o",
            linestyle="-",
            color=colors[color_index],
            label=f"Estimated Value {epi+1}",
        )


plt.plot(
    state[1:6],
    true_values[1:6],
    marker="x",
    linestyle="--",
    color="black",
    label="True Value",
)
plt.xlabel("States")
plt.ylabel("Value")
plt.title(f"Value Estimates After {num_epi} Episode")
plt.legend()
plt.grid(True)


plt.show()
