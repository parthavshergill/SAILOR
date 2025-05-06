import numpy as np
import time
import matplotlib.pyplot as plt
from src.environment.env_wrapper import Game2048Env, test_random_agent

def main():
    """
    Main function to test the 2048 environment with a random agent.
    """
    print("Initializing 2048 environment...")
    env = Game2048Env(render_mode="human")
    
    # Verify compatibility with Stable-Baselines3
    env.verify_sb3_compatibility()
    
    # Test with random agent
    print("\nTesting environment with random agent...")
    episode_rewards = test_random_agent(env, num_episodes=5)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o')
    plt.title('Random Agent Performance in 2048')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('random_agent_performance.png')
    plt.show()
    
    env.close()
    print("Environment test completed!")

if __name__ == "__main__":
    main()