import unittest
import numpy as np
import gymnasium as gym
from src.environment.env_wrapper import Game2048Env

class TestGame2048Environment(unittest.TestCase):
    """Test cases for the 2048 environment wrapper."""
    
    def setUp(self):
        """Set up the environment before each test."""
        self.env = Game2048Env()
    
    def tearDown(self):
        """Clean up after each test."""
        self.env.close()
    
    def test_environment_creation(self):
        """Test that the environment can be created."""
        self.assertIsNotNone(self.env)
        self.assertIsNotNone(self.env.observation_space)
        self.assertIsNotNone(self.env.action_space)
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertTrue(isinstance(info, dict))
    
    def test_step(self):
        """Test taking a step in the environment."""
        self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsNotNone(obs)
        self.assertTrue(isinstance(reward, (int, float)))
        self.assertTrue(isinstance(terminated, bool))
        self.assertTrue(isinstance(truncated, bool))
        self.assertTrue(isinstance(info, dict))
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        for _ in range(3):
            obs, info = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < 100:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                steps += 1
            
            self.assertTrue(steps > 0)

if __name__ == "__main__":
    unittest.main()