# ⚡ Week 2: The Deep RL Era (Value-Based & Neural Approximators)

When environments have massive or continuous state spaces, tabular methods (Q-Tables) completely collapse due to memory and computation limits. This week introduces Deep Reinforcement Learning (DRL) by combining traditional RL with Deep Neural Networks.

## 🧠 Core Concepts Covered
* **Function Approximation:** Using Deep Neural Networks (PyTorch) to estimate Q-values instead of looking them up in a giant table.
* **The Stability Triad Challenge:** Why naive combinations of Neural Networks and Reinforcement Learning lead to divergence due to correlated data and non-stationary targets.
* **Experience Replay (Replay Buffer):** Storing past transitions $(s, a, r, s', done)$ and sampling random mini-batches to break correlations between consecutive experiences.
* **Target Network:** Using a secondary, slowly-updated network to calculate stable Temporal Difference (TD) targets, preventing the moving-target problem during gradient descent.

## 📚 Summary Documentation
* **`week2_summary.pdf`**: A comprehensive LaTeX-generated summary document detailing the transition from tabular methods to Deep Q-Networks, loss functions, and core stability mechanisms.

## 📂 Files & Environments
* **`day8_qnetwork.py`**: Introduction to PyTorch tensors, autograd, and building a basic `Q-Network` architecture to evaluate states in the `CartPole-v1` environment.
* **`day10_dqn.py`**: Full implementation of **Deep Q-Networks (DQN)** featuring a custom `ReplayBuffer`, `main_net`, `target_net`, and Epsilon-Greedy decay to solve CartPole.

## 🚀 How to Run
Navigate to this folder and run any script directly using Python like these examples:
```bash
python day10_dqn.py
``` 
or
```bash
python week2_deep_q_learning/day10_dqn.py
```