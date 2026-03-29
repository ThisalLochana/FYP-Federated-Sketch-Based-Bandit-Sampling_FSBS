"""
LinUCB Bandit — contextual bandit for intelligent sampling decisions.

Architecture reference:
  - 256 arms, each with:
    - A matrix: 4×4 float32 (64 bytes) — accumulated context outer products
    - b vector: 4×1 float32 (16 bytes) — accumulated context-reward products
    - Total per arm: 80 bytes
    - Total all arms: 256 × 80 = 20KB
  
  - UCB formula:
    θ* = A^{-1} · b          (expected reward estimate)
    UCB = x^T·θ* + α·√(x^T·A^{-1}·x)   (exploitation + exploration)
  
  - Decision: if UCB ≥ τ_sample → SAMPLE, else → DROP
  
  - Hot-path cost: 16 multiply-adds + 1 sqrt ≈ 2 microseconds
"""

import numpy as np
from typing import Optional, Tuple


class LinUCBArm:
    """
    Single arm of the LinUCB bandit.
    
    Each arm corresponds to one combination of (svc_cluster, topo_prefix_top4).
    It learns: "for spans with this character, how valuable is it to sample them?"
    
    The arm maintains:
      A: 4×4 matrix — starts as identity (I), accumulates x·x^T on each update
      b: 4×1 vector — starts as zeros, accumulates x·reward on each update
      n: count of times this arm has been pulled (for confidence check)
    """
    
    __slots__ = ['A', 'b', 'A_inv', 'n']
    
    def __init__(self, d: int = 4):
        """
        Args:
            d: Dimension of the context vector (4 for FSBS)
        """
        # A starts as identity matrix — this is the prior:
        # "we have no data, assume each feature is independent with unit variance"
        self.A = np.eye(d, dtype=np.float32)
        
        # b starts as zeros — "no reward signal yet"
        self.b = np.zeros(d, dtype=np.float32)
        
        # Pre-compute A_inv for fast decisions
        # We maintain this incrementally rather than recomputing from scratch
        self.A_inv = np.eye(d, dtype=np.float32)
        
        # Count of observations
        self.n = 0
    
    def ucb_score(self, x: np.ndarray, alpha: float) -> float:
        """
        Compute the Upper Confidence Bound score.
        
        UCB = x^T · θ* + α · √(x^T · A^{-1} · x)
        
        where θ* = A^{-1} · b
        
        First term (x^T·θ*): exploitation — predicted reward
        Second term (α·√...): exploration — uncertainty bonus
        
        High UCB = either we predict high reward, or we're uncertain
                   → should SAMPLE to learn more
        
        Args:
            x: 4-element context vector (from FeatureVector.to_bandit_context())
            alpha: Exploration parameter (higher = more exploration)
                   Typical values: 0.5–2.0
        
        Returns:
            UCB score (float, can be any positive value)
        """
        # θ* = A^{-1} · b  — the estimated reward weights
        theta = self.A_inv @ self.b    # 4×4 @ 4×1 = 4×1
        
        # Exploitation: predicted reward for this context
        exploitation = float(x @ theta)    # 1×4 @ 4×1 = scalar
        
        # Exploration bonus: uncertainty for this context
        # x^T · A^{-1} · x measures how "unexplored" this context is
        exploration_var = float(x @ self.A_inv @ x)  # scalar
        exploration = alpha * np.sqrt(max(exploration_var, 0.0))
        
        return exploitation + exploration
    
    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Update the arm with an observed (context, reward) pair.
        Called ASYNCHRONOUSLY by the reward plane (not on hot path).
        
        Updates:
          A ← A + x · x^T   (rank-1 update)
          b ← b + reward · x
          A_inv ← Sherman-Morrison update of A^{-1}
        
        Sherman-Morrison formula for rank-1 update:
          (A + x·x^T)^{-1} = A^{-1} - (A^{-1}·x·x^T·A^{-1}) / (1 + x^T·A^{-1}·x)
        
        This avoids O(n^3) matrix inversion — just O(n^2) for n=4.
        
        Args:
            x: 4-element context vector
            reward: Observed reward (e.g., 1.0 if this trace was useful,
                    0.0 if not, or a continuous value)
        """
        x = x.astype(np.float32)
        
        # Rank-1 update of A
        self.A += np.outer(x, x)
        
        # Update b
        self.b += reward * x
        
        # Sherman-Morrison update of A_inv
        # Much faster than np.linalg.inv(self.A)
        Ainv_x = self.A_inv @ x                           # 4×1
        denominator = 1.0 + float(x @ Ainv_x)              # scalar
        self.A_inv -= np.outer(Ainv_x, Ainv_x) / denominator  # 4×4
        
        self.n += 1


class LinUCBBandit:
    """
    Full LinUCB bandit with 256 arms.
    
    Total memory: 256 arms × 80 bytes = 20KB
    → fits entirely in L2 cache
    
    Usage on hot path:
        arm_idx = feature_vector.arm_index
        x = feature_vector.to_bandit_context()
        score = bandit.score(arm_idx, x)
        should_sample = score >= threshold
    """
    
    def __init__(
        self,
        n_arms: int = 256,
        d: int = 4,
        alpha: float = 1.0,
        threshold: float = 0.5,
        confidence_threshold: int = 10,
    ):
        """
        Args:
            n_arms: Number of arms (256 = 8-bit address space)
            d: Dimension of context vector (4 features)
            alpha: Exploration parameter for UCB
                   Higher → more exploration (sample more novel traces)
                   Lower → more exploitation (sample what we know is valuable)
            threshold: τ_sample — UCB score threshold for SAMPLE decision
                       Lower → sample more traces
                       Higher → sample fewer traces
            confidence_threshold: Minimum observations before trusting LinUCB
                                  Below this, defer to Thompson sampling
        """
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        
        # Pre-allocate all arms
        self.arms = [LinUCBArm(d) for _ in range(n_arms)]
    
    def score(self, arm_index: int, x: np.ndarray) -> float:
        """
        Compute UCB score for the given arm and context.
        Called on HOT PATH — must be fast.
        
        Args:
            arm_index: Which arm (0–255), from FeatureVector.arm_index
            x: Context vector from FeatureVector.to_bandit_context()
        
        Returns:
            UCB score (float)
        """
        arm_index = arm_index % self.n_arms  # safety clamp
        return self.arms[arm_index].ucb_score(x, self.alpha)
    
    def should_sample(self, arm_index: int, x: np.ndarray) -> Tuple[bool, float]:
        """
        Make the SAMPLE/DROP decision.
        
        Args:
            arm_index: Which arm
            x: Context vector
        
        Returns:
            (decision, score): decision is True=SAMPLE, False=DROP
                               score is the UCB value
        """
        score = self.score(arm_index, x)
        return bool(score >= self.threshold), score
    
    def is_confident(self, arm_index: int) -> bool:
        """
        Check if we have enough data for this arm to trust LinUCB.
        If not, the sampler should fall back to Thompson sampling.
        
        Args:
            arm_index: Which arm
        
        Returns:
            True if arm has been observed enough times
        """
        arm_index = arm_index % self.n_arms
        return self.arms[arm_index].n >= self.confidence_threshold
    
    def update(self, arm_index: int, x: np.ndarray, reward: float) -> None:
        """
        Update arm with reward signal.
        Called ASYNCHRONOUSLY (not on hot path).
        
        Args:
            arm_index: Which arm was pulled
            x: The context vector that was used
            reward: The observed reward
        """
        arm_index = arm_index % self.n_arms
        self.arms[arm_index].update(x, reward)
    
    def get_arm_stats(self, arm_index: int) -> dict:
        """Get diagnostic info for an arm."""
        arm_index = arm_index % self.n_arms
        arm = self.arms[arm_index]
        return {
            'arm_index': arm_index,
            'n_observations': arm.n,
            'confident': arm.n >= self.confidence_threshold,
            'b_norm': float(np.linalg.norm(arm.b)),
        }
    
    @property
    def memory_bytes(self) -> int:
        """Total memory footprint."""
        # Per arm: A (4×4×4) + b (4×4) + A_inv (4×4×4) + n (8) = 144 bytes
        # But architecture spec says 80 bytes (A + b only, A_inv is derived)
        # We store A_inv for speed, so actual is ~144 bytes × 256 = ~36KB
        return self.n_arms * (4 * self.d * self.d * 2 + self.d * 4 + 8)