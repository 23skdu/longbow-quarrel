#!/usr/bin/env python3
"""
Load test script for Longbow-Quarrel WebUI API
Tests concurrent connections (100+)
"""

import asyncio
import aiohttp
import time
import statistics
import argparse
import json
from typing import List, Dict, Tuple

# Configuration
DEFAULT_PROMPT = "Why is the sky blue?"
DEFAULT_MODEL = "mistral:latest"
DEFAULT_CONCURRENT = 100
DEFAULT_REQUESTS = 1000
DEFAULT_TIMEOUT = 60
DEFAULT_URL = "http://localhost:8080"


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    model: str,
    api_key: str = None,
) -> Tuple[float, int, str]:
    """Make a single inference request and return timing/results"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    payload = {"prompt": prompt, "model": model, "temperature": 0.7, "max_tokens": 100}

    start_time = time.time()
    try:
        async with session.post(
            f"{url}/api/generate",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT),
        ) as response:
            if response.status == 200:
                data = await response.json()
                elapsed = time.time() - start_time
                return elapsed, len(data.get("text", "")), data.get("tokens_per_sec", 0)
            else:
                error_text = await response.text()
                return -1, 0, 0
    except Exception as e:
        return -1, 0, 0


async def run_load_test(
    url: str,
    prompt: str,
    model: str,
    concurrent: int,
    total_requests: int,
    api_key: str = None,
) -> Dict:
    """Run concurrent load test"""
    print(f"Starting load test:")
    print(f"  URL: {url}")
    print(f"  Concurrent connections: {concurrent}")
    print(f"  Total requests: {total_requests}")
    print(f"  Model: {model}")
    print()

    connector = aiohttp.TCPConnector(limit=concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.time()

        # Create tasks
        tasks = []
        for i in range(total_requests):
            task = make_request(session, url, prompt, model, api_key)
            tasks.append(task)

        # Run with semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrent)

        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro

        # Execute all tasks
        results = await asyncio.gather(*[run_with_semaphore(t) for t in tasks])

        total_time = time.time() - start_time

    # Process results
    successful = [r for r in results if r[0] > 0]
    failed = [r for r in results if r[0] < 0]

    if not successful:
        print("No successful requests!")
        return {}

    latencies = [r[0] for r in successful]
    tokens_per_sec = [r[2] for r in successful if r[2] > 0]

    stats = {
        "total_time": total_time,
        "total_requests": total_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "success_rate": len(successful) / total_requests * 100,
        "requests_per_second": total_requests / total_time if total_time > 0 else 0,
        "avg_latency": statistics.mean(latencies),
        "p50_latency": statistics.quantiles(latencies, n=2)[0]
        if len(latencies) > 1
        else latencies[0],
        "p95_latency": statistics.quantiles(latencies, n=20)[18]
        if len(latencies) > 19
        else statistics.quantiles(latencies, n=len(latencies))[-1],
        "p99_latency": statistics.quantiles(latencies, n=100)[98]
        if len(latencies) > 99
        else statistics.quantiles(latencies, n=len(latencies))[-1],
        "min_latency": min(latencies),
        "max_latency": max(latencies),
    }

    if tokens_per_sec:
        stats["avg_tokens_per_sec"] = statistics.mean(tokens_per_sec)

    return stats


def print_results(stats: Dict):
    """Print test results in a formatted way"""
    if not stats:
        return

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total time:           {stats['total_time']:.2f} seconds")
    print(f"Total requests:       {stats['total_requests']}")
    print(f"Successful requests:  {stats['successful_requests']}")
    print(f"Failed requests:      {stats['failed_requests']}")
    print(f"Success rate:         {stats['success_rate']:.2f}%")
    print(f"Requests/second:      {stats['requests_per_second']:.2f}")
    print()
    print("Latency Statistics:")
    print(f"  Average:            {stats['avg_latency']:.3f}s")
    print(f"  P50 (Median):       {stats['p50_latency']:.3f}s")
    print(f"  P95:                {stats['p95_latency']:.3f}s")
    print(f"  P99:                {stats['p99_latency']:.3f}s")
    print(f"  Min:                {stats['min_latency']:.3f}s")
    print(f"  Max:                {stats['max_latency']:.3f}s")

    if "avg_tokens_per_sec" in stats:
        print(f"\nThroughput:")
        print(f"  Avg tokens/sec:     {stats['avg_tokens_per_sec']:.2f}")

    print("=" * 60)


async def test_streaming(url: str, prompt: str, model: str, api_key: str = None):
    """Test streaming endpoint with a single request"""
    print("\nTesting streaming endpoint...")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    payload = {"prompt": prompt, "model": model, "temperature": 0.7, "max_tokens": 50}

    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/api/stream",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT),
            ) as response:
                if response.status == 200:
                    tokens_received = 0
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            tokens_received += 1
                    elapsed = time.time() - start_time
                    print(f"  Streaming test completed in {elapsed:.2f}s")
                    print(f"  Tokens received: {tokens_received}")
                    print(f"  Throughput: {tokens_received / elapsed:.2f} tokens/sec")
                else:
                    print(f"  Streaming test failed: {response.status}")
    except Exception as e:
        print(f"  Streaming test error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Load test Longbow-Quarrel WebUI API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the API")
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Prompt to use for testing"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model to use for testing"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=DEFAULT_CONCURRENT,
        help="Number of concurrent connections",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help="Total number of requests",
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--test-streaming", action="store_true", help="Also test streaming endpoint"
    )

    args = parser.parse_args()

    # Run the load test
    stats = asyncio.run(
        run_load_test(
            url=args.url,
            prompt=args.prompt,
            model=args.model,
            concurrent=args.concurrent,
            total_requests=args.requests,
            api_key=args.api_key,
        )
    )

    print_results(stats)

    # Test streaming if requested
    if args.test_streaming:
        asyncio.run(test_streaming(args.url, args.prompt, args.model, args.api_key))


if __name__ == "__main__":
    main()
