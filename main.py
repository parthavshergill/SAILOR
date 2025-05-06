"""
SAILOR Project - Main script for environment testing
"""
import numpy as np
from src.environment.env_wrapper import Game2048Env, test_random_agent

def main():
    """Main function to test the 2048 environment."""
    print("SAILOR Project - 2048 Environment Test")
    print("======================================")
    
    # Create and test the environment
    env = Game2048Env(render_mode="human")
    
    # Verify compatibility with Stable-Baselines3
    env.verify_sb3_compatibility()
    
    # Run random agent test
    print("\nRunning random agent test...")
    rewards = test_random_agent(env, num_episodes=3)
    
    print(f"\nAverage reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()