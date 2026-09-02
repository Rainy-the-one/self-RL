# 🚀 Journey to Master Reinforcement Learning: From Zero to PPO

**Acknowledgments:** This repository and its structured curriculum were created as a personal learning project in collaboration with AI. The theoretical foundations, algorithmic definitions, and core concepts presented throughout this repository heavily reference the comprehensive paper *"Introduction to Reinforcement Learning"* by Majid Ghasemi and Dariush Ebrahimi.

> **⚠️ Disclaimer:** This repository is intended solely for reference and personal learning support. It does not replace formal academic courses, official university subjects, or comprehensive certified curriculums. The content and structure have been curated to fit my personal learning path, but you are more than welcome to customize, fork, and adapt it to suit your own learning goals and projects!

## 📌 Overview

Reinforcement Learning (RL) is a subfield of Artificial Intelligence that focuses on training agents to make decisions by interacting with their environment to maximize cumulative rewards. This 13-day practical roadmap takes you from the absolute basics of Tabular RL (grid worlds) to implementing industry-standard Deep RL algorithms.

## 📦 Installation & Setup

To run the code in this repository, you need to install the core libraries. It is recommended to use a virtual environment.

```bash
git clone <your_repo_url>
cd <your_repo_name>
pip install -r requirements.txt
```

## 💡 Friendly Reminder: 
If you plan to make modifications, experiment with hyper-parameters, or add your own code, it is strongly recommended to create a new branch rather than pushing directly to the `main` branch to keep the `main` branch clean and stable:
```bash
git checkout -b feature/my-experiments # Feel free to name your branch anything you prefer
```


## 🗺️ Curriculum Roadmap

### Week 1: Core Foundations (Tabular RL)
This phase focuses on the mathematical foundations of RL, dealing with discrete states and actions where an agent learns through trial-and-error exploration. We solve simple grid environments using lookup tables (Q-Tables).  
* **Day 1 & 2:** Theory of Markov Decision Processes (MDP) and Bellman Equations.
* **Day 3:** Introduction to the `gymnasium` library and environment interaction.
* **Day 5 (Q-Learning):** Implementation of Q-learning, an off-policy algorithm that learns the value of the optimal policy independently of the agent's behavior.  
* **Day 6 (SARSA):** Implementation of SARSA, an on-policy algorithm that updates its value estimates based on the actions actually taken by the policy.  
* **Day 7:** Applying Q-Learning with an Epsilon-Greedy strategy to the Taxi-v4 environment.

### Week 2: The Deep RL Era (Value-Based)
When environments become too large or continuous, tabular methods collapse. This week introduces Deep Reinforcement Learning (DRL) by integrating neural networks with traditional RL algorithms.  
* **Day 8:** Introduction to PyTorch and building a basic Q-Network.
* **Day 9:** Theory on Catastrophic Forgetting and the need for stability in neural networks.
* **Day 10 (DQN):** Full implementation of Deep Q-Networks (DQN) using an experience replay mechanism to stabilize learning by breaking correlations between consecutive experiences.  

### Week 3: Direct Decision Making (Policy-Based & Hybrid)
Value-based methods can struggle in high-dimensional action spaces; therefore, this week shifts to methods that directly parameterize and optimize the policy.  
* **Day 11 (REINFORCE):** Implementation of a core policy gradient method that uses a log-likelihood gradient estimator to update policy parameters.  
* **Day 12 (Actor-Critic):** A hybrid architecture containing two components: the Actor, which selects actions based on a policy, and the Critic, which evaluates the actions based on a value function.
* **Day 13 (PPO):** Implementation of Proximal Policy Optimization (PPO). This algorithm achieves reliable performance and sample efficiency by using a surrogate objective with a clipped probability ratio to prevent destructively large policy updates.