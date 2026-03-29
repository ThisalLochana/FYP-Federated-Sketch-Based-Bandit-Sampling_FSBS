"""
FSBS Sidecar — Phase 3: Integrated with Docker pipeline.

Changes from Phase 2:
  - Trace-level decision cache (all spans in a trace get same decision)
  - Removed unused imports that could cause ImportError
  - Robust protobuf field access (no HasField needed)
  - Added span/trace counters to metrics logging
  - Error handling for gRPC forwarding
"""

import os
import sys
import time
import signal
import logging
import json
import threading
from concurrent import futures
from collections import OrderedDict

import grpc

# --- Verify proto imports early ---
try:
    from opentelemetry.proto.collector.trace.v1 import (
        trace_service_pb2,
        trace_service_pb2_grpc,
    )
    from opentelemetry.proto.trace.v1 import trace_pb2
except ImportError as e:
    print(f"FATAL: Cannot import OTLP proto definitions: {e}")
    print("Install with: pip install opentelemetry-proto grpcio")
    sys.exit(1)

from fsbs.sampler import FSBSSampler

# Configure logging
logging.basicConfig(
    level=os.environ.get('FSBS_LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger('fsbs-sidecar')


class TraceDecisionCache:
    """
    LRU cache mapping trace_id → sampling decision.

    Why we need this:
      Without caching, each span is decided independently.
      This would cause partial traces in Jaeger (some spans sampled,
      others dropped from the same trace).

    With caching:
      The FIRST span of a trace triggers the FSBS decision.
      All subsequent spans of that trace inherit the same decision.
      Result: complete traces in Jaeger — either fully sampled or fully dropped.
    """

    def __init__(self, max_size: int = 50000):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, trace_id: str):
        """Look up cached decision. Returns bool or None if not cached."""
        with self._lock:
            if trace_id in self._cache:
                self._cache.move_to_end(trace_id)
                self.hits += 1
                return self._cache[trace_id]
            self.misses += 1
            return None

    def put(self, trace_id: str, should_sample: bool):
        """Cache a decision for a trace_id."""
        with self._lock:
            self._cache[trace_id] = should_sample
            self._cache.move_to_end(trace_id)
            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._cache)


class FSBSTraceService(trace_service_pb2_grpc.TraceServiceServicer):
    """
    gRPC service implementing OTLP TraceService/Export.

    Microservices send their spans here (thinking it's a normal collector).
    We run each span through the FSBS decision pipeline, then forward
    only the sampled spans to the real OTel Collector.
    """

    def __init__(self, sampler: FSBSSampler, forward_endpoint: str):
        self.sampler = sampler
        self.trace_cache = TraceDecisionCache()

        # gRPC channel to the real OTel Collector
        self._forward_channel = grpc.insecure_channel(
            forward_endpoint,
            options=[
                ('grpc.max_send_message_length', 16 * 1024 * 1024),
                ('grpc.max_receive_message_length', 16 * 1024 * 1024),
            ],
        )
        self._forward_stub = trace_service_pb2_grpc.TraceServiceStub(
            self._forward_channel
        )

        # Counters
        self.total_spans_in = 0
        self.total_spans_out = 0
        self.total_export_calls = 0
        self.forward_errors = 0

        logger.info(f"Forwarding sampled spans to: {forward_endpoint}")

    def Export(self, request, context):
        """
        Handle ExportTraceServiceRequest from a microservice.

        For each span:
          1. Check trace-level cache
          2. If cache miss → run FSBS decision, cache result
          3. If cache hit → reuse cached decision
          4. Error spans always force SAMPLE (override cache)
          5. Collect sampled spans, forward to real collector
        """
        self.total_export_calls += 1
        sampled_resource_spans = []

        for resource_spans in request.resource_spans:
            # Extract service name from resource attributes
            svc_name = self._get_service_name(resource_spans)

            sampled_scope_spans = []

            for scope_spans in resource_spans.scope_spans:
                sampled_spans = []

                for span in scope_spans.spans:
                    self.total_spans_in += 1
                    should_sample = self._decide_span(span, svc_name)

                    if should_sample:
                        sampled_spans.append(span)
                        self.total_spans_out += 1

                if sampled_spans:
                    new_scope = trace_pb2.ScopeSpans()
                    # Copy the instrumentation scope (library name/version)
                    try:
                        new_scope.scope.CopyFrom(scope_spans.scope)
                    except Exception:
                        pass  # scope not set, that's OK
                    new_scope.spans.extend(sampled_spans)
                    sampled_scope_spans.append(new_scope)

            if sampled_scope_spans:
                new_rs = trace_pb2.ResourceSpans()
                try:
                    new_rs.resource.CopyFrom(resource_spans.resource)
                except Exception:
                    pass  # resource not set
                new_rs.scope_spans.extend(sampled_scope_spans)
                sampled_resource_spans.append(new_rs)

        # Forward sampled spans to real collector
        if sampled_resource_spans:
            forward_request = trace_service_pb2.ExportTraceServiceRequest()
            forward_request.resource_spans.extend(sampled_resource_spans)
            try:
                self._forward_stub.Export(forward_request)
            except grpc.RpcError as e:
                self.forward_errors += 1
                logger.error(f"Forward to collector failed: {e.code()}: {e.details()}")

        return trace_service_pb2.ExportTraceServiceResponse()

    def _decide_span(self, span, svc_name: str) -> bool:
        """
        Make SAMPLE/DROP decision for a single span.
        Uses trace-level caching for consistency.
        """
        trace_id = span.trace_id.hex() if span.trace_id else ''

        # Always sample error spans (architecture: force_sample_errors)
        status_code = span.status.code  # 0=UNSET, 1=OK, 2=ERROR
        is_error = (status_code == 2)

        if is_error:
            # Force sample AND update cache so rest of trace is also sampled
            self.trace_cache.put(trace_id, True)
            return True

        # Check trace-level cache
        cached = self.trace_cache.get(trace_id)
        if cached is not None:
            return cached

        # Cache miss — run FSBS decision on this span
        span_data = self._span_to_dict(span, svc_name, trace_id)
        decision = self.sampler.decide(span_data)

        # Cache the decision for all future spans of this trace
        self.trace_cache.put(trace_id, decision.should_sample)

        if decision.should_sample:
            logger.debug(
                f"SAMPLE: {svc_name}/{span.name} "
                f"[{decision.method}] score={decision.score:.3f} "
                f"arm={decision.arm_index}"
            )
        else:
            logger.debug(
                f"DROP:   {svc_name}/{span.name} "
                f"[{decision.method}] score={decision.score:.3f} "
                f"arm={decision.arm_index}"
            )

        return decision.should_sample

    def _get_service_name(self, resource_spans) -> str:
        """Extract service.name from resource attributes."""
        try:
            for kv in resource_spans.resource.attributes:
                if kv.key == 'service.name':
                    return kv.value.string_value
        except Exception:
            pass
        return 'unknown'

    def _span_to_dict(self, span, service_name: str, trace_id: str) -> dict:
        """Convert protobuf Span to dict for FeatureExtractor."""
        # Duration in microseconds
        duration_ns = span.end_time_unix_nano - span.start_time_unix_nano
        duration_us = max(duration_ns // 1000, 0)

        # Status code: 0=UNSET, 1=OK, 2=ERROR
        status_code = span.status.code

        # Extract span attributes into a plain dict
        attributes = {}
        try:
            for kv in span.attributes:
                key = kv.key
                val = kv.value
                # Check each possible value type
                if val.string_value:
                    attributes[key] = val.string_value
                elif val.int_value:
                    attributes[key] = val.int_value
                elif val.bool_value:
                    attributes[key] = val.bool_value
                elif val.double_value:
                    attributes[key] = val.double_value
        except Exception:
            pass

        return {
            'trace_id': trace_id,
            'span_id': span.span_id.hex() if span.span_id else '',
            'service_name': service_name,
            'operation_name': span.name,
            'duration_us': duration_us,
            'status_code': status_code,
            'attributes': attributes,
            'parent_services': [],  # populated in Phase 4 via trace context
        }

    def get_stats(self) -> dict:
        """Return service-level statistics."""
        return {
            'total_export_calls': self.total_export_calls,
            'total_spans_in': self.total_spans_in,
            'total_spans_out': self.total_spans_out,
            'spans_dropped': self.total_spans_in - self.total_spans_out,
            'forward_errors': self.forward_errors,
            'trace_cache_size': self.trace_cache.size,
            'trace_cache_hits': self.trace_cache.hits,
            'trace_cache_misses': self.trace_cache.misses,
        }


class MetricsReporter:
    """Periodically logs FSBS metrics to stdout."""

    def __init__(
        self,
        sampler: FSBSSampler,
        trace_service: FSBSTraceService,
        interval: float = 10.0,
    ):
        self.sampler = sampler
        self.trace_service = trace_service
        self.interval = interval
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="fsbs-metrics", daemon=True
        )
        self._thread.start()

    def _loop(self):
        while self._running:
            time.sleep(self.interval)
            sm = self.sampler.get_metrics()
            ts = self.trace_service.get_stats()

            span_rate = (
                ts['total_spans_out'] / ts['total_spans_in']
                if ts['total_spans_in'] > 0
                else 0.0
            )

            logger.info(
                f"FSBS METRICS | "
                f"spans_in={ts['total_spans_in']} "
                f"spans_out={ts['total_spans_out']} "
                f"span_pass_rate={span_rate:.1%} | "
                f"sampler: total={sm['total_spans']} "
                f"thompson={sm['thompson_decisions']} "
                f"linucb={sm['linucb_decisions']} "
                f"forced={sm['forced_samples']} | "
                f"cache: size={ts['trace_cache_size']} "
                f"hits={ts['trace_cache_hits']} "
                f"misses={ts['trace_cache_misses']} | "
                f"fwd_errors={ts['forward_errors']}"
            )

    def stop(self):
        self._running = False


def serve():
    """Start the FSBS sidecar gRPC server."""

    # Configuration
    service_name = os.environ.get('FSBS_SERVICE_NAME', 'fsbs-sidecar')
    listen_port = os.environ.get('FSBS_LISTEN_PORT', '4317')
    forward_endpoint = os.environ.get('FSBS_FORWARD_ENDPOINT', 'otel-collector:4317')
    alpha = float(os.environ.get('FSBS_ALPHA', '1.0'))
    threshold = float(os.environ.get('FSBS_THRESHOLD', '0.5'))
    confidence = int(os.environ.get('FSBS_CONFIDENCE_THRESHOLD', '10'))
    metrics_interval = float(os.environ.get('FSBS_METRICS_INTERVAL', '10'))

    logger.info("=" * 60)
    logger.info("FSBS Sidecar starting")
    logger.info(f"  Listen:      0.0.0.0:{listen_port}")
    logger.info(f"  Forward to:  {forward_endpoint}")
    logger.info(f"  Alpha:       {alpha}")
    logger.info(f"  Threshold:   {threshold}")
    logger.info(f"  Confidence:  {confidence}")
    logger.info(f"  Log level:   {logging.getLevelName(logger.getEffectiveLevel())}")
    logger.info("=" * 60)

    # Create sampler
    sampler = FSBSSampler(
        service_name=service_name,
        alpha=alpha,
        threshold=threshold,
        confidence_threshold=confidence,
        force_sample_errors=True,
    )

    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_receive_message_length', 16 * 1024 * 1024),
            ('grpc.max_send_message_length', 16 * 1024 * 1024),
        ],
    )

    # Register trace service
    trace_service = FSBSTraceService(
        sampler=sampler,
        forward_endpoint=forward_endpoint,
    )
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(
        trace_service, server
    )

    # Start metrics reporter
    reporter = MetricsReporter(sampler, trace_service, interval=metrics_interval)

    # Start server
    server.add_insecure_port(f'0.0.0.0:{listen_port}')
    server.start()
    logger.info(f"FSBS Sidecar listening on 0.0.0.0:{listen_port}")
    logger.info("Waiting for spans from microservices...")

    # Shutdown handling
    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        logger.info(f"Signal {signum} received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    # Cleanup
    reporter.stop()
    sampler.shutdown()
    server.stop(grace=5)

    # Final report
    final_sampler = sampler.get_metrics()
    final_service = trace_service.get_stats()
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info(f"  Sampler:  {json.dumps(final_sampler, indent=2)}")
    logger.info(f"  Service:  {json.dumps(final_service, indent=2)}")
    logger.info("=" * 60)


if __name__ == '__main__':
    serve()