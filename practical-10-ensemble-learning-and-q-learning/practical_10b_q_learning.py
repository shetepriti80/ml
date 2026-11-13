# NOTE: This file was ADDED because the original notebook contains the
# question text for Question 10 Option B (Q-Learning on a 4x4 Grid World)
# but no code for it. The other 10 programs in this repository are
# unmodified copies of the notebook code.

import numpy as np

# ---------------- Environment: 4 x 4 Grid World ----------------
N = 4                      # grid size
START, GOAL = (0, 0), (3, 3)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]       # up, down, left, right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
STEP_REWARD, GOAL_REWARD = -1, 10                  # rewards

def step(state, action):
    """Apply an action; stay in place if it would leave the grid."""
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < N and 0 <= nc < N):
        nr, nc = r, c
    next_state = (nr, nc)
    if next_state == GOAL:
        return next_state, GOAL_REWARD, True      # terminal state
    return next_state, STEP_REWARD, False

# ---------------- Q-Learning ----------------
alpha, gamma = 0.1, 0.9                  # learning rate, discount factor
epsilon, eps_min, eps_decay = 1.0, 0.05, 0.995
episodes, max_steps = 1000, 100
rng = np.random.default_rng(42)
Q = np.zeros((N, N, len(ACTIONS)))

successes = 0
for ep in range(episodes):
    state = START
    for _ in range(max_steps):
        # epsilon-greedy: explore with prob. epsilon, otherwise exploit
        if rng.random() < epsilon:
            action = int(rng.integers(len(ACTIONS)))
        else:
            action = int(np.argmax(Q[state]))
        next_state, reward, done = step(state, action)
        # Q-learning update (Bellman equation)
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state = next_state
        if done:
            successes += 1
            break
    epsilon = max(eps_min, epsilon * eps_decay)

print(f"Training success rate: {successes / episodes:.2%}")

# ---------------- Learned policy ----------------
print("\nLearned policy (greedy):")
arrows = ["^", "v", "<", ">"]
for r in range(N):
    row = []
    for c in range(N):
        row.append("G" if (r, c) == GOAL else arrows[int(np.argmax(Q[r, c]))])
    print(" ".join(row))

# ---------------- Evaluate greedy policy ----------------
state, path, total = START, [START], 0
for _ in range(max_steps):
    action = int(np.argmax(Q[state]))
    state, reward, done = step(state, action)
    path.append(state)
    total += reward
    if done:
        break
print("\nGreedy path:", path)
print("Reached goal:", state == GOAL, "| Steps:", len(path) - 1,
      "| Total reward:", total)
