"""
Thompson Sampling Fallback — cold-start safety net.

Architecture reference:
  - Maintains Beta(α, β) distribution per arm
  - Used when arm has fewer than `confidence_threshold` observations
  - Sampling from Beta distribution is O(1) — nanoseconds
  - As arm accumulates observations, automatically transitions to LinUCB
  
  Memory: 256 arms × 2 floats × 4 bytes = 2KB
"""

import random
from typing import Tuple


class ThompsonSampler:
    """
    Thompson Sampling with Beta priors for binary rewards.
    
    How it works (plain English):
    - Each arm has a Beta(α, β) distribution
    - α counts "successes" (trace was useful), β counts "failures" (was not)
    - To decide: sample p ~ Beta(α, β), compare to threshold
    - If p > threshold → SAMPLE
    - Arms with more successes get higher α → higher samples → more likely to be sampled
    - Arms with no data have α=β=1 (uniform distribution) → 50/50 chance → explores freely
    
    This is the "cold start safety net" — when LinUCB doesn't have enough
    data to make a good decision, Thompson provides a principled way to
    explore while still making decisions.
    """
    
    def __init__(
        self,
        n_arms: int = 256,
        threshold: float = 0.5,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        """
        Args:
            n_arms: Number of arms (256)
            threshold: Comparison threshold for sampling decision
            prior_alpha: Initial α (prior successes). 1.0 = uniform prior
            prior_beta: Initial β (prior failures). 1.0 = uniform prior
        """
        self.n_arms = n_arms
        self.threshold = threshold
        
        # Pre-allocate: [alpha, beta] for each arm
        self.alphas = [prior_alpha] * n_arms
        self.betas = [prior_beta] * n_arms
    
    def should_sample(self, arm_index: int) -> Tuple[bool, float]:
        """
        Make a sampling decision using Thompson sampling.
        
        Samples p ~ Beta(α_k, β_k) and compares to threshold.
        
        Args:
            arm_index: Which arm (0–255)
        
        Returns:
            (decision, sampled_p): decision is True=SAMPLE, False=DROP
                                   sampled_p is the value drawn from Beta
        """
        arm_index = arm_index % self.n_arms
        
        # Sample from Beta distribution
        # Python's random.betavariate is implemented in C — very fast
        p = random.betavariate(
            self.alphas[arm_index],
            self.betas[arm_index]
        )
        
        return p >= self.threshold, p
    
    def update(self, arm_index: int, reward: float) -> None:
        """
        Update the Beta prior with an observation.
        
        For binary rewards:
          reward = 1.0 → α += 1 (success)
          reward = 0.0 → β += 1 (failure)
        
        For continuous rewards in [0, 1]:
          α += reward
          β += (1 - reward)
        
        Args:
            arm_index: Which arm
            reward: Observed reward value [0, 1]
        """
        arm_index = arm_index % self.n_arms
        self.alphas[arm_index] += reward
        self.betas[arm_index] += (1.0 - reward)
    
    def get_stats(self, arm_index: int) -> dict:
        """Get diagnostic info for an arm."""
        arm_index = arm_index % self.n_arms
        alpha = self.alphas[arm_index]
        beta = self.betas[arm_index]
        return {
            'arm_index': arm_index,
            'alpha': alpha,
            'beta': beta,
            'mean': alpha / (alpha + beta),  # expected value of Beta
            'observations': int(alpha + beta - 2),  # subtract priors
        }
    
    @property
    def memory_bytes(self) -> int:
        """Memory footprint: 256 arms × 2 floats × 8 bytes = 4KB"""
        return self.n_arms * 2 * 8