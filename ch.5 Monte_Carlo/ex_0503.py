import numpy as np
import random
import matplotlib.pyplot as plt


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


def monte_carlo_ES(num_epi):
    q_s_a = {}
    Returns = {}
    policy = {}

    """초기 배열 생성"""
    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            for ace in [True, False]:
                for action in ["hit", "stick"]:
                    q_s_a[((player_sum, dealer_card, ace), action)] = 0
                    Returns[((player_sum, dealer_card, ace), action)] = []
                policy[(player_sum, dealer_card, ace)] = (
                    "hit" if player_sum < 20 else "stick"
                )

    for _ in range(num_epi):
        episode = []
        player_hand = draw_hand()
        dealer_card = draw_card()
        usable = usable_ace(player_hand)
        player_sum = hand_value(player_hand)

        if player_sum < 12:
            continue

        action = random.choice(
            ["hit", "stick"]
        )  # 처음엔 무작위 행동/ 이후에 policy에 따라서
        state = (player_sum, dealer_card, usable)
        done = False
        while not done:
            next_state, reward, done = step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if not done:
                action = policy[state]

        G = 0
        visited_states_action = set()
        for state, action, reward in reversed(episode):
            G = reward + G
            if (state, action) not in visited_states_action:
                visited_states_action.add((state, action))
                Returns[(state, action)].append(G)
                q_s_a[(state, action)] = np.mean(Returns[(state, action)])
                policy[state] = max(["hit", "stick"], key=lambda a: q_s_a[state, a])

    return q_s_a, policy


num_episodes = 500000
q_s_a, policy = monte_carlo_ES(num_episodes)


def plot_policy_lines(policy):
    x = np.arange(1, 11)  # Dealer's showing card
    y_usable_ace = []
    y_no_usable_ace = []

    for dealer_card in x:
        for player_sum in range(12, 22):
            if policy[(player_sum, dealer_card, True)] == "stick":
                y_usable_ace.append(player_sum)
                break
        else:
            y_usable_ace.append(21)

        for player_sum in range(12, 22):
            if policy[(player_sum, dealer_card, False)] == "stick":
                y_no_usable_ace.append(player_sum)
                break
        else:
            y_no_usable_ace.append(21)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, y_usable_ace, drawstyle="steps-post")
    plt.title("Usable Ace")
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")
    plt.xticks(np.arange(1, 11), ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.yticks(np.arange(12, 22))

    plt.subplot(1, 2, 2)
    plt.plot(x, y_no_usable_ace, drawstyle="steps-post")
    plt.title("No Usable Ace")
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")
    plt.xticks(np.arange(1, 11), ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.yticks(np.arange(12, 22))

    plt.show()


plot_policy_lines(policy)
