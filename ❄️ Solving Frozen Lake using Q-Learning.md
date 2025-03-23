---
title: ❄️ Solving Frozen Lake using Q-Learning

---

# ❄️ Solving Frozen Lake using Q-Learning

This project demonstrates the implementation of **Varoius Algorithms** to solve the **Frozen Lake** environment in OpenAI Gymnasium. I implemented various Algorithms and compared the results with one another.


## 1) Q - Learning 
The agent learns an optimal policy to navigate through the frozen lake and reach the goal while avoiding holes.

The governing Equation is:

$$
Q(s, a) = Q(s, a) + \alpha \left[ r + \gamma \, \text{max} \, Q(s', a') - Q(s, a) \right]
$$


where

Alpha (α): Learning rate 
Gamma (γ): Discount factor 

And in Exploration I had implemented the greedy algorithm with exponential decay.

### Results:

