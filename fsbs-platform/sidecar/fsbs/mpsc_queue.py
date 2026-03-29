"""
MPSC (Multi-Producer Single-Consumer) Queue — lock-free ring buffer.

Architecture reference:
  - After the sampling decision, a 12-byte record is pushed onto this queue
  - Record: (trace_id_hash, arm_index, decision)
  - Push is ~10 nanoseconds (single CAS instruction in C; in Python we
    use threading.Lock for correctness — still microseconds)
  - The background sketch updater thread drains this queue

In production (Rust/C++), this would be a true lock-free ring buffer.
In Python, we use collections.deque (which is thread-safe for append/popleft)
with an explicit capacity limit.
"""

import collections
import threading
from typing import NamedTuple, Optional, List


class SamplingRecord(NamedTuple):
    """
    12-byte record pushed after every sampling decision.
    
    Fields:
        trace_id_hash: 32-bit hash of the trace ID (for sketch update)
        arm_index: Which bandit arm was used (0–255)
        decision: 1 = SAMPLE, 0 = DROP
        feature_key: Packed feature vector key (for sketch update)
    """
    trace_id_hash: int
    arm_index: int
    decision: int
    feature_key: int


class MPSCQueue:
    """
    Thread-safe bounded queue for passing records from hot path
    to background workers.
    
    In Python, collections.deque with maxlen is thread-safe for
    single-append and single-popleft operations (CPython GIL).
    We add a threading.Event so the consumer can wait efficiently.
    """
    
    def __init__(self, capacity: int = 4096):
        """
        Args:
            capacity: Maximum items in the queue.
                      If full, new items are silently dropped (acceptable
                      since dropping a tracking record just means the sketch
                      update is slightly delayed — no correctness impact).
        """
        self.capacity = capacity
        self._queue = collections.deque(maxlen=capacity)
        self._event = threading.Event()  # signals consumer that items are available
        self._dropped = 0  # counter for monitoring
    
    def push(self, record: SamplingRecord) -> bool:
        """
        Push a record onto the queue. Called from HOT PATH.
        
        This must be as fast as possible. In Python, deque.append()
        with maxlen automatically drops the oldest item if full.
        We detect this and count drops.
        
        Args:
            record: The SamplingRecord to enqueue
        
        Returns:
            True if enqueued, False if queue was full (oldest was evicted)
        """
        was_full = len(self._queue) >= self.capacity
        self._queue.append(record)
        if was_full:
            self._dropped += 1
        self._event.set()  # signal consumer
        return not was_full
    
    def drain(self, max_items: int = 256) -> List[SamplingRecord]:
        """
        Drain up to max_items from the queue. Called by BACKGROUND thread.
        
        Args:
            max_items: Maximum number of items to drain in one batch
        
        Returns:
            List of SamplingRecord items
        """
        items = []
        for _ in range(max_items):
            try:
                items.append(self._queue.popleft())
            except IndexError:
                break
        
        if not self._queue:
            self._event.clear()
        
        return items
    
    def wait(self, timeout: float = 1.0) -> bool:
        """
        Block until items are available or timeout.
        Used by the background consumer thread.
        
        Args:
            timeout: Maximum seconds to wait
        
        Returns:
            True if items are available, False if timed out
        """
        return self._event.wait(timeout=timeout)
    
    @property
    def size(self) -> int:
        return len(self._queue)
    
    @property
    def dropped_count(self) -> int:
        return self._dropped