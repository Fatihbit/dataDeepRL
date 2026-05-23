"""
Vectorized Trading Environment
==============================

Runs multiple CryptoTradingEnv instances synchronously but batches
observations for efficient GPU inference. Auto-resets environments
when episodes end.

Usage:
    vec_env = VectorizedTradingEnv(num_envs=8, env_kwargs={...})
    obs_list = vec_env.reset()
    for step in range(n_steps):
        # obs_list is a list of N observation dicts
        actions = agent.select_actions_batch(obs_list)
        obs_list, rewards, dones, infos = vec_env.step(actions)
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from src.envs.trading_env import CryptoTradingEnv


class VectorizedTradingEnv:
    """
    Synchronous vectorized environment that runs N environment instances.
    
    Each step advances all environments simultaneously, auto-resetting
    any that terminate. This produces N transitions per step, giving
    the GPU larger batches for inference.
    
    Supports both CryptoTradingEnv (dict obs) and FlatCryptoTradingEnv (flat obs).
    """
    
    def __init__(self, num_envs: int, env_kwargs: dict, env_class=None):
        """
        Args:
            num_envs: Number of parallel environments.
            env_kwargs: Keyword arguments passed to env_class().
            env_class: Environment class to instantiate (default: CryptoTradingEnv).
        """
        if env_class is None:
            env_class = CryptoTradingEnv
        self.num_envs = num_envs
        self.envs = [env_class(**env_kwargs) for _ in range(num_envs)]
        
        # Proxy attributes from first env
        self.prices = self.envs[0].prices
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
    
    def reset(self) -> List[dict]:
        """Reset all environments and return list of observations."""
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return obs_list
    
    def step(self, actions: List[int]) -> Tuple[List[dict], np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments with given actions.
        
        Auto-resets terminated environments and returns the new obs
        from the reset (standard vectorized env behavior).
        
        Args:
            actions: List of N actions, one per environment.
            
        Returns:
            obs_list: List of N observation dicts (post-reset for done envs)
            rewards: Array of N rewards (from the step that ended)
            dones: Array of N booleans
            infos: List of N info dicts (includes terminal info for done envs)
        """
        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards[i] = reward
            dones[i] = done
            
            if done:
                # Store terminal info before reset
                info['terminal_observation'] = obs
                info['terminal_info'] = info.copy()
                # Auto-reset
                obs, _ = env.reset()
            
            obs_list.append(obs)
            infos.append(info)
        
        return obs_list, rewards, dones, infos
