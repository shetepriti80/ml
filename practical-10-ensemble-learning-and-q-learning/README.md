# Practical 10 — Option A: Ensemble Learning · Option B: Q-Learning

| Option | File | Origin |
|---|---|---|
| **A** – Decision Tree vs Random Forest vs AdaBoost | [`practical_10a_ensemble_learning.py`](practical_10a_ensemble_learning.py) | from your notebook (unchanged) |
| **B** – Q-Learning on a 4×4 Grid World | [`practical_10b_q_learning.py`](practical_10b_q_learning.py) | **added** — the notebook has the question but no code for it |

---

# Option A — Ensemble Learning

**Dataset:** Breast Cancer Wisconsin

## 1. Aim
Compare a single Decision Tree with a bagging ensemble (Random Forest) and a boosting ensemble (AdaBoost).

## 2. Theory

### Decision tree
Recursively splits the data on the feature/threshold that gives the best impurity reduction (**Gini** = 1 − Σpₖ² or **entropy**). An unpruned tree fits training data perfectly → **high variance / overfitting**.

### Ensemble learning
Combine many weak/diverse models so the group outperforms any single one.

| | Bagging (Random Forest) | Boosting (AdaBoost) |
|---|---|---|
| Training | trees trained **in parallel** on bootstrap samples | learners trained **sequentially** |
| Focus | reduces **variance** | reduces **bias** |
| Idea | each tree sees a random sample of rows (and a random subset of features at each split); predictions are **averaged / majority-voted** | each new weak learner (a shallow tree "stump") gives **more weight to samples previously misclassified**; learners are combined with weights αₜ |
| Overfitting | resistant | can overfit noisy data |

AdaBoost weight update (intuition): αₜ = ½ ln((1 − εₜ)/εₜ); misclassified samples' weights are multiplied by e^{αₜ}.

## 3. Code walkthrough
| Step | Explanation |
|---|---|
| Split | 80/20 stratified, `random_state=42`. |
| Models | `DecisionTreeClassifier` (baseline), `RandomForestClassifier(n_estimators=150)`, `AdaBoostClassifier(n_estimators=100)`. |
| Loop | Fit each, predict, record accuracy / precision / recall / F1 → DataFrame. |

### Full code
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append([name, accuracy_score(y_test, pred),
                 precision_score(y_test, pred),
                 recall_score(y_test, pred),
                 f1_score(y_test, pred)])

print(pd.DataFrame(rows, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-score"]))
```

### Output
```text
                      Model  Accuracy  Precision    Recall  F1-score
0  Decision Tree (baseline)  0.912281   0.955882  0.902778  0.928571
1             Random Forest  0.956140   0.958904  0.972222  0.965517
2                  AdaBoost  0.956140   0.946667  0.986111  0.965986
```

## 4. Interpretation
* Baseline tree: 91.2 % accuracy. Both ensembles reach **95.6 %** → ensembles clearly beat the single tree (+4.4 points).
* Random Forest and AdaBoost tie on accuracy; AdaBoost has the highest recall (0.986), Random Forest the higher precision (0.959).

## 5. Viva questions
* **Bagging vs boosting?** Parallel + variance reduction vs sequential + bias reduction.
* **Why is Random Forest better than one tree?** Averaging many decorrelated trees lowers variance.
* **What is a bootstrap sample?** Random sample of the training set drawn with replacement.

## 6. Notes from debugging
Runs without errors. The notebook shows a `FutureWarning` about `SAMME.R` (older scikit-learn 1.4/1.5). Newer scikit-learn (≥ 1.6) removed `SAMME.R`, so the warning no longer appears and the results are the same. No change needed.

---

# Option B — Q-Learning (4 × 4 Grid World)

> ⚠️ **Added file.** Your notebook lists this experiment in Question 10 but contains no code for it, so `practical_10b_q_learning.py` was written to complete the set. Your original programs are untouched.

## 1. Aim
Teach an agent to reach the goal of a 4×4 grid using Q-learning, with **epsilon-greedy** exploration, then evaluate the learned policy.

## 2. Theory

### Reinforcement learning (RL)
An **agent** interacts with an **environment**: in state *s* it takes action *a*, receives reward *r* and moves to state *s′*. The goal is a **policy π(s)** that maximizes the expected **discounted return** Σ γᵗ rₜ.

### Q-learning (model-free, off-policy, tabular)
Learns **Q(s, a)** = expected return of taking *a* in *s* and acting optimally afterwards, using the Bellman update:

**Q(s,a) ← Q(s,a) + α [ r + γ · maxₐ′ Q(s′,a′) − Q(s,a) ]**

* **α (learning rate)** – how much new information overrides old.
* **γ (discount factor)** – importance of future vs immediate reward.
* The bracket is the **TD (temporal-difference) error**.

### Exploration vs exploitation — ε-greedy
* With probability **ε** → *explore* (random action).
* With probability **1 − ε** → *exploit* (action with highest Q).
* ε starts at 1.0 and **decays** (×0.995 per episode, floor 0.05) so the agent explores early and exploits later.

## 3. Environment and code walkthrough
| Element | Value |
|---|---|
| Grid | 4×4, states (row, col) |
| Start / Goal | (0,0) / (3,3) — goal is terminal |
| Actions | Up, Down, Left, Right (moving into a wall = stay) |
| Rewards | −1 per step, +10 on reaching the goal |
| Hyper-parameters | α = 0.1, γ = 0.9, 1000 episodes, ≤100 steps each |
| Q-table | array of shape (4, 4, 4) initialized to 0 |

Flow: for each episode start at (0,0) → choose action (ε-greedy) → `step()` → Bellman update → stop at goal → decay ε. After training, the **greedy policy** is printed as arrows and executed once to measure the path and reward.

### Full code
```python
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
```

### Output
```text
Training success rate: 99.80%

Learned policy (greedy):
v v v v
> v v v
> v v v
> > > G

Greedy path: [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)]
Reached goal: True | Steps: 6 | Total reward: 5
```

## 4. Interpretation
* **Success rate ≈ 99.8 %** of training episodes reached the goal (early random episodes occasionally time out).
* The policy grid shows arrows leading from every cell toward `G`.
* The greedy path has **6 steps = the shortest possible** (Manhattan distance 3 + 3), with total reward 5 (five −1 steps + the +10 final step).

## 5. Viva questions
* **Why "off-policy"?** The update uses max over next actions (the greedy policy) regardless of the action actually taken.
* **What if γ = 0?** Agent is myopic — only immediate reward matters.
* **Why decay ε?** To shift from exploration to exploitation as knowledge improves.
