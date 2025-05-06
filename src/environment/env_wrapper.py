import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

class Game2048Env:
    """
    Wrapper for the 2048 Gymnasium environment.
    Ensures compatibility with Stable-Baselines3 and adds additional functionality.
    """
    def __init__(self, render_mode=None):
        """
        Initialize the 2048 environment.
        
        Args:
            render_mode (str, optional): Rendering mode ('human', 'rgb_array', etc.)
        """
        try:
            # Try to load the gymnasium-2048 environment
            # Note: The actual environment ID may vary depending on the package
            self.env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", render_mode=render_mode)
        except gym.error.NamespaceNotFound:
            try:
                # Alternative environment ID
                self.env = gym.make('2048-v0', render_mode=render_mode)
            except gym.error.NamespaceNotFound:
                raise ImportError(
                    "Could not find the 2048 environment. "
                    "Please install a compatible gymnasium-2048 package."
                )
        
        # Store environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Print environment info
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        return self.env.reset(seed=seed)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0-3 for up, right, down, left)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        return self.env.step(action)
    
    def render(self):
        """Render the current state of the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def verify_sb3_compatibility(self):
        """Verify that the environment is compatible with Stable-Baselines3."""
        try:
            check_env(self.env)
            print("Environment is compatible with Stable-Baselines3!")
            return True
        except Exception as e:
            print(f"Environment is NOT compatible with Stable-Baselines3: {e}")
            return False

def test_random_agent(env, num_episodes=5, max_steps=1000):
    """
    Test the environment with a random agent.
    
    Args:
        env: The environment to test
        num_episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
        
    Returns:
        list: List of episode rewards
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while step_count < max_steps:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Optional: Render the environment
            # env.render()
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {step_count}")
        episode_rewards.append(episode_reward)
    
    return episode_rewards