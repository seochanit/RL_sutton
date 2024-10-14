import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


def draw_card():
    card = random.randint(1, 13)
    return min(card, 10)


def draw_hand():
    return [draw_card(), draw_card()]


def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def hand_value(hand):
    value = sum(hand)
    if usable_ace(hand):
        value += 10
    return value


def is_bust(hand):
    return hand_value(hand) > 21


def policy(player_sum):
    return "hit" if player_sum < 20 else "stick"


def step(state, action):
    player_sum, dealer_card, usable = state

    if action == "hit":
        new_card = draw_card()
        player_sum += new_card
        if is_bust([player_sum]):
            return (player_sum, dealer_card, usable), -1, True
        return (player_sum, dealer_card, usable), 0, False

    dealer_hand = [dealer_card, draw_card()]
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(draw_card())
    dealer_sum = hand_value(dealer_hand)
    if is_bust(dealer_hand):
        return (player_sum, dealer_card, usable), 1, True
    return (player_sum, dealer_card, usable), compare(player_sum, dealer_sum), True


def compare(player_sum, dealer_sum):
    if player_sum > dealer_sum:
        return 1
    elif player_sum < dealer_sum:
        return -1
    return 0


def monte_carlo(num_episodes):
    V = {}
    Returns = {}

    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            for ace in [True, False]:
                V[(player_sum, dealer_card, ace)] = 0
                Returns[(player_sum, dealer_card, ace)] = []

    for _ in range(num_episodes):
        episode = []
        player_hand = draw_hand()
        dealer_card = draw_card()
        usable = usable_ace(player_hand)
        player_sum = hand_value(player_hand)

        if player_sum < 12:
            continue

        state = (player_sum, dealer_card, usable)
        done = False
        while not done:
            action = policy(player_sum)
            next_state, reward, done = step(state, action)
            episode.append((state, reward))
            state = next_state
            player_sum = state[0]

        G = 0
        visited_states = set()
        for state, reward in reversed(episode):
            G = reward + G
            if state not in visited_states:
                visited_states.add(state)
                Returns[state].append(G)
                V[state] = np.mean(Returns[state])

    return V


def plot_heatmap(V, title):
    x_range = np.arange(12, 22)
    y_range = np.arange(1, 11)

    # 히트맵을 위한 두 개의 데이터 배열
    values_usable = np.zeros((len(x_range), len(y_range)))
    values_no_usable = np.zeros((len(x_range), len(y_range)))

    for i, player_sum in enumerate(x_range):
        for j, dealer_card in enumerate(y_range):
            values_usable[i, j] = V[(player_sum, dealer_card, True)]
            values_no_usable[i, j] = V[(player_sum, dealer_card, False)]

    values_usable = np.flipud(values_usable)
    values_no_usable = np.flipud(values_no_usable)

    # 서브플롯으로 히트맵 시각화
    _, axs = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(
        values_usable,
        annot=True,
        fmt=".2f",
        xticklabels=y_range,
        yticklabels=x_range[::-1],
        cmap="coolwarm",
        ax=axs[0],
    )
    axs[0].set_title(f"{title} (Usable Ace)")
    axs[0].set_xlabel("Dealer Showing")
    axs[0].set_ylabel("Player Sum")

    sns.heatmap(
        values_no_usable,
        annot=True,
        fmt=".2f",
        xticklabels=y_range,
        yticklabels=x_range[::-1],
        cmap="coolwarm",
        ax=axs[1],
    )
    axs[1].set_title(f"{title} (No Usable Ace)")
    axs[1].set_xlabel("Dealer Showing")
    axs[1].set_ylabel("Player Sum")

    plt.show()


num_episodes = 10000
V = monte_carlo(num_episodes)

# 히트맵 플롯
plot_heatmap(V, title="State-Value Function")
