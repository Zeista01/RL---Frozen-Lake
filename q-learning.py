import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def run_q_learning(episodes=15000, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    
    # Initialize Q-table
    q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Learning parameters
    alpha = 0.8       # Learning rate
    gamma = 0.95      # Discount factor
    epsilon = 1.0     # Exploration rate
    epsilon_decay = 0.0001
    min_epsilon = 0.01
    
    rewards_per_episode = np.zeros(episodes)
    
    print("Training using Q-Learning...")
    start_time = time.time()
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Choose action using epsilon-greedy policy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Update Q-table using Q-learning update rule
            q[state, action] = q[state, action] + alpha * (
                reward + gamma * np.max(q[next_state]) - q[state, action]
            )
            
            # Update state
            state = next_state
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        
        # Gradually decrease learning rate
        if epsilon <= min_epsilon:
            alpha = max(0.01, alpha * 0.999)
        
        if reward == 1:
            rewards_per_episode[i] = 1
        
        # Print progress
        if (i+1) % 1000 == 0:
            success_rate = np.mean(rewards_per_episode[i-999:i+1]) * 100
            print(f"Episode {i+1}/{episodes} - Success rate: {success_rate:.2f}% - Epsilon: {epsilon:.4f} - Alpha: {alpha:.4f}")
    
    end_time = time.time()
    print(f"Q-Learning training completed in {end_time - start_time:.2f} seconds")
    
    # Save Q-table
    with open("frozen_lake_q_learning.pkl", "wb") as f:
        pickle.dump(q, f)
    
    # Test the policy
    test_episodes = 1000
    test_rewards = np.zeros(test_episodes)
    
    print("Testing policy...")
    for i in range(test_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action = np.argmax(q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            
            if reward == 1:
                test_rewards[i] = 1
    
    env.close()
    
    print(f"Final success rate: {np.mean(test_rewards) * 100:.2f}%")
    
    # Plot the results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.figure(figsize=(10, 5))
    plt.plot(sum_rewards)
    plt.title('Q-Learning Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Success (last 100 episodes)')
    plt.savefig('frozen_lake_q_learning.png')
    plt.close()
    
    return q

if __name__ == '__main__':
    q = run_q_learning(render=True)
    
    # Extract policy from Q-table
    policy = np.argmax(q, axis=1)
    print("\nLearned Policy (0=left, 1=down, 2=right, 3=up):")
    print(policy.reshape(8, 8))
