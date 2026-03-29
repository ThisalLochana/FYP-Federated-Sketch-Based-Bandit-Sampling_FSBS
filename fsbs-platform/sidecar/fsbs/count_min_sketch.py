"""
Count-Min Sketch for trace pattern frequency estimation.

Architecture reference:
  - 4 rows × 512 columns of 4-byte counters = 8KB total
  - Query gives estimated frequency of a feature combination
  - High frequency → routine (low novelty)
  - Low frequency → rare (high novelty → should sample)

The sketch is READ on the hot path, WRITTEN by the background thread.
"""

import struct
import math
from typing import List


class CountMinSketch:
    """
    Count-Min Sketch data structure.
    
    How it works (plain English):
    - Imagine 4 separate arrays, each with 512 slots
    - To ADD an item: hash it 4 different ways, get 4 slot indices,
      increment the counter in each slot
    - To QUERY an item: hash it the same 4 ways, look up the 4 counters,
      return the MINIMUM (hence "Count-Min")
    - The minimum is always >= the true count (never under-counts)
    - Collisions cause over-counting, but the minimum across 4 rows
      reduces this error dramatically
    
    Memory: 4 rows × 512 cols × 4 bytes = 8,192 bytes = 8KB
    """
    
    # Fixed dimensions matching architecture spec
    NUM_ROWS = 4
    NUM_COLS = 512
    
    def __init__(self):
        # Pre-allocate all counters as a flat list (no heap allocation later)
        # Using a list of lists for clarity; in production this would be
        # a flat array or numpy array
        self._counters: List[List[int]] = [
            [0] * self.NUM_COLS for _ in range(self.NUM_ROWS)
        ]
        
        # Pre-computed hash seeds (one per row)
        # These are arbitrary primes — they just need to be different
        self._seeds = [0x9E3779B1, 0x517CC1B7, 0x6C62272E, 0x2E1B2138]
    
    def _hash(self, key: int, row: int) -> int:
        """
        Hash function for row `row`.
        Uses a simple multiply-shift hash with the row's seed.
        
        Args:
            key: The feature vector packed as a 32-bit integer
            row: Which row (0-3) — determines which hash function
        
        Returns:
            Column index (0 to NUM_COLS-1)
        """
        # Multiply-shift hash: fast and good distribution
        # h = ((key * seed) >> 16) % NUM_COLS
        h = (key * self._seeds[row]) & 0xFFFFFFFF  # keep 32-bit
        h = (h >> 16) ^ (h & 0xFFFF)               # mix bits
        return h % self.NUM_COLS
    
    def update(self, key: int, count: int = 1) -> None:
        """
        Add `count` occurrences of `key` to the sketch.
        Called by BACKGROUND thread (not on hot path).
        
        Args:
            key: The feature vector as a packed 32-bit integer
            count: How many times to add (usually 1)
        """
        for row in range(self.NUM_ROWS):
            col = self._hash(key, row)
            self._counters[row][col] += count
    
    def estimate(self, key: int) -> int:
        """
        Estimate the frequency of `key`.
        Called on HOT PATH (read-only, no mutation).
        
        Returns the minimum counter value across all rows.
        This is an upper bound on the true frequency, but
        the minimum makes it tight.
        
        Args:
            key: The feature vector as a packed 32-bit integer
        
        Returns:
            Estimated count (always >= true count)
        """
        min_count = float('inf')
        for row in range(self.NUM_ROWS):
            col = self._hash(key, row)
            count = self._counters[row][col]
            if count < min_count:
                min_count = count
        return int(min_count)
    
    def novelty_score(self, key: int) -> float:
        """
        Convert frequency estimate to a novelty score in [0, 1].
        
        Formula: novelty = 1.0 / (1 + log(estimate + 1))
        
        - estimate = 0 → novelty = 1.0 (never seen before, very novel!)
        - estimate = 1 → novelty ≈ 0.59
        - estimate = 10 → novelty ≈ 0.29
        - estimate = 100 → novelty ≈ 0.18
        - estimate = 1000 → novelty ≈ 0.13
        
        High novelty → FSBS should prefer to SAMPLE this trace
        """
        est = self.estimate(key)
        return 1.0 / (1.0 + math.log(est + 1))
    
    def merge(self, other: 'CountMinSketch') -> None:
        """
        Merge another sketch into this one using element-wise maximum.
        Used by the aggregator when receiving gossip from peers.
        
        Why maximum (not addition)?
        - Each node counts independently
        - Taking max avoids double-counting the same events
        - This is the standard CMS merge for distributed systems
        """
        for row in range(self.NUM_ROWS):
            for col in range(self.NUM_COLS):
                if other._counters[row][col] > self._counters[row][col]:
                    self._counters[row][col] = other._counters[row][col]
    
    def serialize(self) -> bytes:
        """
        Serialize sketch to bytes for gossip transmission.
        4 rows × 512 cols × 4 bytes = 8,192 bytes
        
        In the gossip protocol, we'd send deltas (changes since last send)
        rather than the full sketch, to stay under the 256-byte gossip budget.
        This full serialization is for initial sync / anti-entropy.
        """
        data = bytearray()
        for row in range(self.NUM_ROWS):
            for col in range(self.NUM_COLS):
                data.extend(struct.pack('<I', self._counters[row][col] & 0xFFFFFFFF))
        return bytes(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'CountMinSketch':
        """Deserialize sketch from bytes."""
        sketch = cls()
        offset = 0
        for row in range(cls.NUM_ROWS):
            for col in range(cls.NUM_COLS):
                sketch._counters[row][col] = struct.unpack('<I', data[offset:offset+4])[0]
                offset += 4
        return sketch
    
    def reset(self) -> None:
        """Clear all counters. Used for testing or periodic decay."""
        for row in range(self.NUM_ROWS):
            for col in range(self.NUM_COLS):
                self._counters[row][col] = 0
    
    @property
    def memory_bytes(self) -> int:
        """Return the memory footprint in bytes."""
        return self.NUM_ROWS * self.NUM_COLS * 4  # 8,192 bytes