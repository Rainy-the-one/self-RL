# 🧊 Week 1: Core Foundations (Tabular Methods)

Welcome to Week 1 of the Reinforcement Learning journey! This week focuses on the foundational mathematics of RL, dealing with discrete states and actions where an agent learns through trial-and-error exploration. 

## 🧠 Core Concepts Covered
* **Markov Decision Processes (MDPs):** The formal mathematical framework for sequential decision-making, defined by states, actions, transition dynamics, reward functions, and discount factor.
* **Bellman Equations:** Expressing the value of a state or state-action pair recursively in terms of its potential successors.
* **Exploration vs. Exploitation:** Balancing the need to discover unknown strategies versus maximizing immediate rewards using the Epsilon-Greedy strategy.
* **Off-Policy vs. On-Policy:** Understanding how algorithms learn (e.g., Q-learning learns the optimal policy independently of the agent's behavior, while SARSA updates estimates based on the actions actually taken).

## 📚 Summary Documentation
* **`week1_summary.pdf`**: A comprehensive LaTeX-generated summary document detailing all theoretical foundations, mathematical formulations, and tabular algorithms covered in Week 1.

## 📂 Files & Environments
* **`day3_frozenlake_intro.py`**: Introduction to the `gymnasium` library and random interaction with the `FrozenLake-v1` environment[cite: 19].
* **`day5_qlearning.py`**: Implementation of Q-Learning (Off-policy TD control) to solve FrozenLake using a lookup table (Q-Table).
* **`day6_sarsa.py`**: Implementation of SARSA (On-policy TD control) to compare its behavior with Q-Learning on slippery ice.
* **`day7_taxi.py`**: Applying Q-Learning and Epsilon-Greedy strategy to a more complex discrete environment (`Taxi-v4`).

## 🚀 How to Run
Navigate to this folder and run any script directly using Python like these examples:
```bash
python day5_qlearning.py
``` 
or
```bash
python week1_tabular_rl/day5_qlearning.py
```