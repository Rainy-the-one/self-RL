# 🎯 Week 3: Direct Decision Making (Policy-Based & Hybrid Methods)

Value-based methods (like DQN) struggle in high-dimensional or continuous action spaces because finding the max Q-value across infinite or massive action choices becomes computationally prohibitive. This week shifts toward Policy-Based methods that directly parameterize and optimize the policy, culminating in advanced hybrid architectures like Actor-Critic and PPO.

## 🧠 Core Concepts Covered
* **Policy Gradient Theorem:** Directly adjusting policy parameters via gradient ascent using the log-likelihood trick to maximize expected cumulative rewards (`REINFORCE`).
* **Actor-Critic Architecture:** Combining the best of both worlds—an **Actor** that selects actions based on a policy, and a **Critic** that evaluates actions using a value function to drastically reduce variance.
* **The Stability Problem of Basic Actor-Critic:** Why online 1-step updates suffer from high variance and policy collapse, requiring careful gradient clipping and loss balancing.
* **Proximal Policy Optimization (PPO):** Utilizing batching, multiple training epochs, and a **clipped surrogate objective** ($\pm 20\%$) to guarantee stable, monotonic policy improvements without catastrophic drops.

## 📚 Summary Documentation
* **`week3_summary.pdf`**: A comprehensive LaTeX-generated summary document detailing policy gradients, REINFORCE derivation, Actor-Critic architectures, and the mathematics behind PPO clipping.

## 📂 Files & Environments
* **`day11_reinforce.py`**: Implementation of the vanilla `REINFORCE` algorithm, evaluating cumulative returns ($\mathcal{G}_t$) at the end of complete episodes.
* **`day12_actor_critic.py`**: Implementation of a 1-step `Actor-Critic` network enhanced with loss weighting, entropy bonuses, and gradient clipping.
* **`test_actor_critic.py`**: Script to load saved weights (`actor_critic_best.pth`) and perform graphical rendering of the trained Actor-Critic agent[cite: 21].
* **`day13_ppo.py`**: Implementation of standard `PPO` with experience collection, advantage normalization, and the clipping mechanism for robust convergence.
* **`test_ppo.py`**: Script to load `ppo_best.pth` and watch the optimal PPO agent effortlessly balance CartPole.

## 🚀 How to Run
Navigate to this folder and run any script directly using Python like these examples:
```bash
python day13_ppo.py
```
or
```bash
python week3_policy_gradients/day13_ppo.py
```

(Note: After training Actor-Critic or PPO, you can run their respective test scripts to observe the trained agent performing visually via graphical rendering, for example: ```python test_actor_critic.py``` or ```python test_ppo.py```).