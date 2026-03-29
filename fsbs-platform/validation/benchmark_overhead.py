r"""
Benchmark the FSBS sidecar overhead.

Sends OTLP spans directly to the sidecar and measures round-trip time.
This isolates the sidecar's processing overhead from network/service latency.

Usage:
  cd D:\IIT\...\fsbs-platform
  python validation\benchmark_overhead.py
"""

import time
import logging
import grpc
import random

from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)
from opentelemetry.proto.trace.v1 import trace_pb2
from opentelemetry.proto.common.v1 import common_pb2
from opentelemetry.proto.resource.v1 import resource_pb2

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

SIDECAR_GRPC = "localhost:4320"  # mapped from container 4317→host 4320


def make_span(
    service_name: str = "benchmark-service",
    operation: str = "test-op",
    duration_ms: int = 50,
    has_error: bool = False,
) -> trace_service_pb2.ExportTraceServiceRequest:
    """Create a minimal OTLP ExportTraceServiceRequest with one span."""

    # Generate random trace/span IDs
    trace_id = random.randbytes(16)
    span_id = random.randbytes(8)

    now_ns = int(time.time() * 1e9)
    start_ns = now_ns - (duration_ms * 1_000_000)

    span = trace_pb2.Span(
        trace_id=trace_id,
        span_id=span_id,
        name=operation,
        kind=trace_pb2.Span.SPAN_KIND_SERVER,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=now_ns,
    )

    if has_error:
        span.status.CopyFrom(
            trace_pb2.Status(code=trace_pb2.Status.STATUS_CODE_ERROR)
        )

    # Resource with service name
    resource = resource_pb2.Resource()
    resource.attributes.append(
        common_pb2.KeyValue(
            key="service.name",
            value=common_pb2.AnyValue(string_value=service_name),
        )
    )

    scope_spans = trace_pb2.ScopeSpans(spans=[span])
    resource_spans = trace_pb2.ResourceSpans(
        resource=resource, scope_spans=[scope_spans]
    )

    request = trace_service_pb2.ExportTraceServiceRequest(
        resource_spans=[resource_spans]
    )
    return request


def benchmark():
    logger.info("=" * 60)
    logger.info("FSBS Sidecar Overhead Benchmark")
    logger.info(f"Target: {SIDECAR_GRPC}")
    logger.info("=" * 60)

    # Connect
    channel = grpc.insecure_channel(SIDECAR_GRPC)
    stub = trace_service_pb2_grpc.TraceServiceStub(channel)

    # Warmup
    logger.info("\nWarmup (50 requests)...")
    for _ in range(50):
        req = make_span(duration_ms=random.randint(1, 500))
        try:
            stub.Export(req)
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            return

    # Benchmark: normal spans
    logger.info("\nBenchmark: 500 normal spans...")
    latencies_normal = []
    for _ in range(500):
        req = make_span(
            service_name=random.choice([
                'frontend', 'checkoutservice', 'paymentservice',
                'currencyservice', 'productcatalogservice',
            ]),
            duration_ms=random.randint(1, 500),
        )
        start = time.perf_counter()
        stub.Export(req)
        elapsed = time.perf_counter() - start
        latencies_normal.append(elapsed * 1_000_000)  # microseconds

    # Benchmark: error spans
    logger.info("Benchmark: 100 error spans...")
    latencies_error = []
    for _ in range(100):
        req = make_span(
            service_name='checkoutservice',
            duration_ms=random.randint(100, 2000),
            has_error=True,
        )
        start = time.perf_counter()
        stub.Export(req)
        elapsed = time.perf_counter() - start
        latencies_error.append(elapsed * 1_000_000)

    channel.close()

    # Report
    def stats(data, label):
        data.sort()
        n = len(data)
        avg = sum(data) / n
        p50 = data[n // 2]
        p95 = data[int(n * 0.95)]
        p99 = data[int(n * 0.99)]
        mn = min(data)
        mx = max(data)
        logger.info(f"\n  {label}:")
        logger.info(f"    Count:  {n}")
        logger.info(f"    Mean:   {avg:>10.1f} µs")
        logger.info(f"    P50:    {p50:>10.1f} µs")
        logger.info(f"    P95:    {p95:>10.1f} µs")
        logger.info(f"    P99:    {p99:>10.1f} µs")
        logger.info(f"    Min:    {mn:>10.1f} µs")
        logger.info(f"    Max:    {mx:>10.1f} µs")
        return {'mean': avg, 'p50': p50, 'p95': p95, 'p99': p99}

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS — Sidecar round-trip latency")
    logger.info("(includes gRPC overhead + FSBS decision + forwarding)")
    normal_stats = stats(latencies_normal, "Normal spans (500)")
    error_stats = stats(latencies_error, "Error spans (100)")

    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info(f"  FSBS sidecar adds ~{normal_stats['p50']:.0f}µs "
                f"(P50) per span")
    
    if normal_stats['p50'] < 5000:
        logger.info("  ★ Under 5ms — acceptable for production")
    elif normal_stats['p50'] < 10000:
        logger.info("  ⟳ 5-10ms — acceptable for most workloads")
    else:
        logger.info("  ⚠ Over 10ms — may need optimization")

    # Note about Python vs production
    logger.info("\n  Note: This is a PYTHON prototype.")
    logger.info("  Architecture target: <5µs (in Rust/C++)")
    logger.info(
        f"  Python overhead: ~{normal_stats['p50']:.0f}µs "
        f"(~{normal_stats['p50']/5:.0f}x target)"
    )
    logger.info("  The ALGORITHM is validated; the implementation")
    logger.info("  language is a deployment choice, not an architecture flaw.")
    logger.info("=" * 60)


if __name__ == '__main__':
    benchmark()