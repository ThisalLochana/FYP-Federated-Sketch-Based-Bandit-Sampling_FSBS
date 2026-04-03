r"""
FSBS Sidecar — with local checkpoint for crash recovery.
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

try:
    from opentelemetry.proto.collector.trace.v1 import (
        trace_service_pb2,
        trace_service_pb2_grpc,
    )
    from opentelemetry.proto.trace.v1 import trace_pb2
except ImportError as e:
    print(f"FATAL: Cannot import OTLP proto definitions: {e}")
    sys.exit(1)

from fsbs.sampler import FSBSSampler
from fsbs.http_api import start_http_server

logging.basicConfig(
    level=os.environ.get('FSBS_LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger('fsbs-sidecar')


class TraceDecisionCache:
    def __init__(self, max_size: int = 50000):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, trace_id: str):
        with self._lock:
            if trace_id in self._cache:
                self._cache.move_to_end(trace_id)
                self.hits += 1
                return self._cache[trace_id]
            self.misses += 1
            return None

    def put(self, trace_id: str, should_sample: bool):
        with self._lock:
            self._cache[trace_id] = should_sample
            self._cache.move_to_end(trace_id)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._cache)


class FSBSTraceService(trace_service_pb2_grpc.TraceServiceServicer):
    def __init__(self, sampler: FSBSSampler, forward_endpoint: str):
        self.sampler = sampler
        self.trace_cache = TraceDecisionCache()
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
        self.total_spans_in = 0
        self.total_spans_out = 0
        self.total_export_calls = 0
        self.forward_errors = 0
        logger.info(f"Forwarding sampled spans to: {forward_endpoint}")

    def Export(self, request, context):
        self.total_export_calls += 1
        sampled_resource_spans = []

        for resource_spans in request.resource_spans:
            svc_name = self._get_service_name(resource_spans)
            sampled_scope_spans = []

            for scope_spans in resource_spans.scope_spans:
                sampled_spans = []
                for span in scope_spans.spans:
                    self.total_spans_in += 1
                    if self._decide_span(span, svc_name):
                        sampled_spans.append(span)
                        self.total_spans_out += 1

                if sampled_spans:
                    new_scope = trace_pb2.ScopeSpans()
                    try:
                        new_scope.scope.CopyFrom(scope_spans.scope)
                    except Exception:
                        pass
                    new_scope.spans.extend(sampled_spans)
                    sampled_scope_spans.append(new_scope)

            if sampled_scope_spans:
                new_rs = trace_pb2.ResourceSpans()
                try:
                    new_rs.resource.CopyFrom(resource_spans.resource)
                except Exception:
                    pass
                new_rs.scope_spans.extend(sampled_scope_spans)
                sampled_resource_spans.append(new_rs)

        if sampled_resource_spans:
            fwd = trace_service_pb2.ExportTraceServiceRequest()
            fwd.resource_spans.extend(sampled_resource_spans)
            try:
                self._forward_stub.Export(fwd)
            except grpc.RpcError as e:
                self.forward_errors += 1
                logger.error(f"Forward failed: {e.code()}: {e.details()}")

        return trace_service_pb2.ExportTraceServiceResponse()

    def _decide_span(self, span, svc_name: str) -> bool:
        trace_id = span.trace_id.hex() if span.trace_id else ''

        # ── Error detection (check MULTIPLE indicators) ──
        is_error = False

        # Check 1: OTLP status code
        if span.status.code == 2:
            is_error = True

        # Check 2: Span attributes / tags
        if not is_error:
            try:
                for kv in span.attributes:
                    key = kv.key
                    val = kv.value
                    # error=true tag
                    if key == 'error' and val.bool_value is True:
                        is_error = True
                        break
                    # otel.status_code=ERROR tag
                    if key == 'otel.status_code' and val.string_value == 'ERROR':
                        is_error = True
                        break
                    # HTTP 5xx status
                    if key == 'http.status_code' and val.int_value >= 500:
                        is_error = True
                        break
                    # gRPC error codes (codes 1-16 are errors)
                    if key == 'rpc.grpc.status_code' and val.int_value > 0:
                        is_error = True
                        break
            except Exception:
                pass
        
        # Force-sample any error span
        if is_error:
            self.trace_cache.put(trace_id, True)
            logger.debug(f"Force-sampling error span: trace={trace_id[:16]} svc={svc_name}")
            return True

        # Check trace cache for existing decision
        cached = self.trace_cache.get(trace_id)
        if cached is not None:
            return cached

        # Normal bandit decision
        span_data = self._span_to_dict(span, svc_name, trace_id)
        decision = self.sampler.decide(span_data)
        self.trace_cache.put(trace_id, decision.should_sample)
        return decision.should_sample

    def _get_service_name(self, resource_spans) -> str:
        try:
            for kv in resource_spans.resource.attributes:
                if kv.key == 'service.name':
                    return kv.value.string_value
        except Exception:
            pass
        return 'unknown'

    def _span_to_dict(self, span, service_name, trace_id):
        duration_ns = span.end_time_unix_nano - span.start_time_unix_nano
        duration_us = max(duration_ns // 1000, 0)
        status_code = span.status.code
        attributes = {}
        try:
            for kv in span.attributes:
                key = kv.key
                val = kv.value
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
            'parent_services': [],
        }

    def get_stats(self) -> dict:
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
    def __init__(self, sampler, trace_service, interval=10.0):
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
                if ts['total_spans_in'] > 0 else 0.0
            )

            ckpt = sm.get('checkpoint', {})                    # ← CHECKPOINT
            ckpt_info = (
                f"ckpt_saves={ckpt.get('saves_completed', 0)}"
                if ckpt else "ckpt=disabled"
            )

            logger.info(
                f"FSBS METRICS | "
                f"spans_in={ts['total_spans_in']} "
                f"spans_out={ts['total_spans_out']} "
                f"rate={span_rate:.1%} | "
                f"thompson={sm['thompson_decisions']} "
                f"linucb={sm['linucb_decisions']} "
                f"forced={sm['forced_samples']} | "
                f"rewards={sm['rewards_received']} "
                f"avg_rwd={sm['avg_reward']:.3f} | "
                f"arms: active={sm['active_arms']} "
                f"confident={sm['confident_arms']} | "
                f"{ckpt_info} | "                              # ← CHECKPOINT
                f"fwd_err={ts['forward_errors']}"
            )

    def stop(self):
        self._running = False


def serve():
    service_name = os.environ.get('FSBS_SERVICE_NAME', 'fsbs-sidecar')
    listen_port = os.environ.get('FSBS_LISTEN_PORT', '4317')
    forward_endpoint = os.environ.get('FSBS_FORWARD_ENDPOINT', 'otel-collector:4317')
    http_port = int(os.environ.get('FSBS_HTTP_PORT', '8081'))
    alpha = float(os.environ.get('FSBS_ALPHA', '1.0'))
    threshold = float(os.environ.get('FSBS_THRESHOLD', '0.5'))
    confidence = int(os.environ.get('FSBS_CONFIDENCE_THRESHOLD', '10'))
    metrics_interval = float(os.environ.get('FSBS_METRICS_INTERVAL', '10'))
    checkpoint_dir = os.environ.get('FSBS_CHECKPOINT_DIR', '')         # ← CHECKPOINT
    checkpoint_interval = float(os.environ.get('FSBS_CHECKPOINT_INTERVAL', '60'))  # ← CHECKPOINT

    logger.info("=" * 60)
    logger.info("FSBS Sidecar starting")
    logger.info(f"  gRPC:        0.0.0.0:{listen_port}")
    logger.info(f"  HTTP API:    0.0.0.0:{http_port}")
    logger.info(f"  Forward to:  {forward_endpoint}")
    logger.info(f"  Alpha={alpha}  Threshold={threshold}  Confidence={confidence}")
    logger.info(f"  Checkpoint:  {'enabled → ' + checkpoint_dir if checkpoint_dir else 'disabled'}")  # ← CHECKPOINT
    logger.info("=" * 60)

    sampler = FSBSSampler(
        service_name=service_name,
        alpha=alpha,
        threshold=threshold,
        confidence_threshold=confidence,
        force_sample_errors=True,
        checkpoint_dir=checkpoint_dir,                         # ← CHECKPOINT
        checkpoint_interval=checkpoint_interval,               # ← CHECKPOINT
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_receive_message_length', 16 * 1024 * 1024),
            ('grpc.max_send_message_length', 16 * 1024 * 1024),
        ],
    )

    trace_service = FSBSTraceService(sampler, forward_endpoint)
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(trace_service, server)

    start_http_server(sampler, trace_service, port=http_port)

    reporter = MetricsReporter(sampler, trace_service, interval=metrics_interval)

    server.add_insecure_port(f'0.0.0.0:{listen_port}')
    server.start()
    logger.info(f"gRPC listening on 0.0.0.0:{listen_port}")

    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        logger.info(f"Signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    reporter.stop()
    sampler.shutdown()
    server.stop(grace=5)

    final = sampler.get_metrics()
    logger.info(f"FINAL: {json.dumps(final, indent=2)}")


if __name__ == '__main__':
    serve()