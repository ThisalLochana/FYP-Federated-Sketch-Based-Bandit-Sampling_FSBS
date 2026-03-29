"""
FSBS Sampler — the decision orchestrator.

This is the central component that ties together:
  - Feature Extractor (span → 4-byte vector)
  - Count-Min Sketch (vector → novelty score)
  - LinUCB Bandit (context + arm → UCB score) [warm path]
  - Thompson Sampling (arm → Beta sample) [cold path]
  - MPSC Queue (record decision for background processing)

Hot-path flow:
  1. Receive span
  2. Extract features (~50ns)
  3. Query sketch for novelty (~100ns)
  4. Check confidence → LinUCB or Thompson
  5. Compare score to threshold → SAMPLE or DROP
  6. Push record to MPSC queue (~10ns)
  Total: <5µs
"""

import time
import threading
import logging
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .feature_extractor import FeatureExtractor, FeatureVector
from .count_min_sketch import CountMinSketch
from .linucb import LinUCBBandit
from .thompson import ThompsonSampler
from .mpsc_queue import MPSCQueue, SamplingRecord

logger = logging.getLogger(__name__)


class SamplingDecision:
    """Result of a sampling decision."""
    __slots__ = ['should_sample', 'score', 'method', 'arm_index', 'feature_vector']
    
    def __init__(
        self,
        should_sample: bool,
        score: float,
        method: str,  # "linucb", "thompson", or "force_sample"
        arm_index: int,
        feature_vector: FeatureVector,
    ):
        self.should_sample = should_sample
        self.score = score
        self.method = method
        self.arm_index = arm_index
        self.feature_vector = feature_vector
    
    def __repr__(self):
        action = "SAMPLE" if self.should_sample else "DROP"
        return f"SamplingDecision({action}, score={self.score:.3f}, method={self.method}, arm={self.arm_index})"


class FSBSSampler:
    """
    Main sampler class — the heart of the FSBS sidecar.
    
    Usage:
        sampler = FSBSSampler(service_name="frontend")
        
        # On every incoming span (hot path):
        decision = sampler.decide(span_data)
        if decision.should_sample:
            forward_to_jaeger(span)
        
        # Background thread (started automatically):
        # - Drains MPSC queue
        # - Updates sketch counters
        # - (Future: gossip, reward processing)
    """
    
    def __init__(
        self,
        service_name: str,
        alpha: float = 1.0,
        threshold: float = 0.5,
        confidence_threshold: int = 10,
        force_sample_errors: bool = True,
    ):
        """
        Args:
            service_name: Name of the service this sidecar is attached to
            alpha: LinUCB exploration parameter
            threshold: UCB/Thompson score threshold for SAMPLE decision
            confidence_threshold: Min observations before trusting LinUCB
            force_sample_errors: Always sample spans with errors (regardless of bandit)
        """
        self.service_name = service_name
        self.force_sample_errors = force_sample_errors
        
        # Core components
        self.feature_extractor = FeatureExtractor(service_name)
        self.sketch = CountMinSketch()
        self.bandit = LinUCBBandit(
            alpha=alpha,
            threshold=threshold,
            confidence_threshold=confidence_threshold,
        )
        self.thompson = ThompsonSampler(threshold=threshold)
        self.queue = MPSCQueue(capacity=4096)
        
        # Metrics
        self._total_spans = 0
        self._sampled_spans = 0
        self._dropped_spans = 0
        self._thompson_decisions = 0
        self._linucb_decisions = 0
        self._forced_samples = 0
        self._lock = threading.Lock()
        
        # Start background worker
        self._running = True
        self._bg_thread = threading.Thread(
            target=self._background_worker,
            name="fsbs-sketch-updater",
            daemon=True,
        )
        self._bg_thread.start()
        
        logger.info(
            f"FSBS Sampler initialized for service '{service_name}' "
            f"(alpha={alpha}, threshold={threshold}, "
            f"confidence_threshold={confidence_threshold})"
        )
    
    def decide(self, span_data: Dict[str, Any]) -> SamplingDecision:
        """
        Make a SAMPLE/DROP decision for an incoming span.
        THIS IS THE HOT PATH — must be fast.
        
        Args:
            span_data: Dict with span fields (see FeatureExtractor.extract)
        
        Returns:
            SamplingDecision with the verdict
        """
        self._total_spans += 1
        
        # Step 1: Extract features (~50ns target)
        fv = self.feature_extractor.extract(span_data)
        
        # Step 2: Query sketch for novelty (~100ns target, read-only)
        novelty = self.sketch.novelty_score(fv.packed_key)
        fv.novelty_score = novelty
        
        # Step 3: Force-sample errors (bypass bandit)
        if self.force_sample_errors and fv.has_error:
            self._forced_samples += 1
            self._sampled_spans += 1
            decision = SamplingDecision(
                should_sample=True,
                score=1.0,
                method="force_sample",
                arm_index=fv.arm_index,
                feature_vector=fv,
            )
            self._enqueue(span_data, fv, decision)
            return decision
        
        # Step 4: Context vector for bandit
        x = np.array(fv.to_bandit_context(), dtype=np.float32)
        
        # Step 5: Confidence check → LinUCB or Thompson
        if self.bandit.is_confident(fv.arm_index):
            # Warm path: LinUCB decision (~2µs)
            should_sample, score = self.bandit.should_sample(fv.arm_index, x)
            method = "linucb"
            self._linucb_decisions += 1
        else:
            # Cold path: Thompson fallback (~10ns)
            should_sample, score = self.thompson.should_sample(fv.arm_index)
            method = "thompson"
            self._thompson_decisions += 1
        
        if should_sample:
            self._sampled_spans += 1
        else:
            self._dropped_spans += 1
        
        decision = SamplingDecision(
            should_sample=should_sample,
            score=score,
            method=method,
            arm_index=fv.arm_index,
            feature_vector=fv,
        )
        
        # Step 6: Enqueue record for background processing (~10ns)
        self._enqueue(span_data, fv, decision)
        
        return decision
    
    def _enqueue(
        self,
        span_data: Dict[str, Any],
        fv: FeatureVector,
        decision: SamplingDecision,
    ) -> None:
        """Push a record onto the MPSC queue (hot path, must be fast)."""
        # Hash the trace ID for compact storage
        trace_id = span_data.get('trace_id', '')
        if isinstance(trace_id, str):
            trace_id_hash = hash(trace_id) & 0xFFFFFFFF
        else:
            trace_id_hash = int(trace_id) & 0xFFFFFFFF
        
        record = SamplingRecord(
            trace_id_hash=trace_id_hash,
            arm_index=fv.arm_index,
            decision=1 if decision.should_sample else 0,
            feature_key=fv.packed_key,
        )
        self.queue.push(record)
    
    def _background_worker(self) -> None:
        """
        Background thread that drains the MPSC queue and updates the sketch.
        Runs continuously until shutdown.
        
        This is the ONLY writer to the sketch — no locking needed for
        sketch reads on the hot path (reads see slightly stale data, which
        is perfectly acceptable for frequency estimation).
        """
        logger.info("FSBS background worker started")
        while self._running:
            # Wait for items (blocks up to 1 second)
            if not self.queue.wait(timeout=1.0):
                continue
            
            # Drain batch
            records = self.queue.drain(max_items=256)
            
            # Update sketch with each record
            for record in records:
                self.sketch.update(record.feature_key)
        
        logger.info("FSBS background worker stopped")
    
    def process_reward(self, arm_index: int, context: list, reward: float) -> None:
        """
        Process a reward signal from the reward plane.
        Updates both LinUCB and Thompson.
        Called ASYNCHRONOUSLY (not on hot path).
        
        Args:
            arm_index: Which arm received the reward
            context: The 4-element context vector
            reward: Reward value (0.0 to 1.0)
        """
        x = np.array(context, dtype=np.float32)
        self.bandit.update(arm_index, x, reward)
        self.thompson.update(arm_index, reward)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current sampling metrics for monitoring.
        """
        sample_rate = (
            self._sampled_spans / self._total_spans
            if self._total_spans > 0
            else 0.0
        )
        return {
            'service': self.service_name,
            'total_spans': self._total_spans,
            'sampled_spans': self._sampled_spans,
            'dropped_spans': self._dropped_spans,
            'sample_rate': round(sample_rate, 4),
            'thompson_decisions': self._thompson_decisions,
            'linucb_decisions': self._linucb_decisions,
            'forced_samples': self._forced_samples,
            'queue_size': self.queue.size,
            'queue_dropped': self.queue.dropped_count,
            'sketch_memory_bytes': self.sketch.memory_bytes,
            'bandit_memory_bytes': self.bandit.memory_bytes,
        }
    
    def shutdown(self) -> None:
        """Stop the background worker thread."""
        self._running = False
        self._bg_thread.join(timeout=5.0)
        logger.info("FSBS Sampler shut down")