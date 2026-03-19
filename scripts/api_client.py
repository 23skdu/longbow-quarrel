#!/usr/bin/env python3
"""
Longbow-Quarrel API Client

A comprehensive Python client for interacting with the Longbow-Quarrel WebUI API.
Provides CRUD operations, API validation, and administrative functionality.

Usage:
    python api_client.py [command] [options]

Commands:
    health          - Check API health status
    models          - List available models
    generate        - Generate text (synchronous)
    stream          - Stream text generation (SSE)
    validate        - Run API validation suite
    admin           - Run administrative checks
    benchmark       - Run performance benchmarks
    websocket       - Test WebSocket connection
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Dict, Any, List
import os


class LongbowClient:
    """Client for Longbow-Quarrel WebUI API"""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"

        req_data = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)

        try:
            response = urllib.request.urlopen(req, timeout=self.timeout)
            content = response.read().decode("utf-8")

            if stream:
                return {"status": response.status, "stream": True}

            if content:
                return {"status": response.status, "data": json.loads(content)}
            return {"status": response.status, "data": {}}

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else "{}"
            try:
                error_json = json.loads(error_body)
            except:
                error_json = {"error": error_body}

            return {
                "status": e.code,
                "error": error_json,
                "success": False,
            }
        except urllib.error.URLError as e:
            return {"status": 0, "error": str(e), "success": False}

    def health(self) -> Dict[str, Any]:
        """GET /health - Full health check with status and checks"""
        result = self._make_request("GET", "/health")
        result["endpoint"] = "/health"
        result["description"] = "Full health check with system status"
        return result

    def healthz(self) -> Dict[str, Any]:
        """GET /healthz - Simple liveness probe"""
        result = self._make_request("GET", "/healthz")
        result["endpoint"] = "/healthz"
        result["description"] = "Simple liveness probe"
        return result

    def readyz(self) -> Dict[str, Any]:
        """GET /readyz - Readiness probe with system checks"""
        result = self._make_request("GET", "/readyz")
        result["endpoint"] = "/readyz"
        result["description"] = "Readiness probe with memory/goroutine checks"
        return result

    def version(self) -> Dict[str, Any]:
        """GET /version - Get API version info"""
        result = self._make_request("GET", "/version")
        result["endpoint"] = "/version"
        result["description"] = "API version information"
        return result

    def metrics(self) -> Dict[str, Any]:
        """GET /metrics - Get Prometheus metrics"""
        result = self._make_request("GET", "/metrics")
        result["endpoint"] = "/metrics"
        result["description"] = "Prometheus metrics endpoint"
        return result

    def list_models(self) -> Dict[str, Any]:
        """GET /api/models - List available models"""
        result = self._make_request("GET", "/api/models")
        result["endpoint"] = "/api/models"
        result["description"] = "List available models"
        return result

    def generate(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """POST /api/generate - Generate text synchronously"""
        data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "topk": top_k,
            "topp": top_p,
        }
        result = self._make_request("POST", "/api/generate", data)
        result["endpoint"] = "/api/generate"
        result["description"] = "Synchronous text generation"
        return result

    def stream_generate(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """POST /api/generate - Stream text generation via SSE"""
        url = f"{self.base_url}/api/stream"
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"

        data = json.dumps(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "topk": top_k,
                "topp": top_p,
            }
        ).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        tokens = []

        try:
            response = urllib.request.urlopen(req, timeout=self.timeout)
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        token_data = json.loads(line[6:])
                        tokens.append(token_data)
                    except json.JSONDecodeError:
                        pass
        except urllib.error.HTTPError as e:
            return [{"error": e.reason, "status": e.code}]
        except Exception as e:
            return [{"error": str(e)}]

        return tokens


def print_result(result: Dict[str, Any], verbose: bool = False) -> bool:
    """Print API result in formatted way"""
    if "success" in result and not result["success"]:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return False

    status = result.get("status", 0)
    if status >= 200 and status < 300:
        print(f"✅ {result.get('endpoint')}: {result.get('description')}")
        if verbose and "data" in result:
            print(f"   Data: {json.dumps(result['data'], indent=2)}")
        return True
    else:
        print(f"❌ {result.get('endpoint')}: HTTP {status}")
        if "error" in result:
            print(f"   Error: {result['error']}")
        return False


def validate_api(client: LongbowClient, verbose: bool = False) -> Dict[str, Any]:
    """Run full API validation suite"""
    print("\n" + "=" * 60)
    print("API VALIDATION SUITE")
    print("=" * 60 + "\n")

    tests = [
        ("Health Check (/health)", lambda: client.health()),
        ("Liveness (/healthz)", lambda: client.healthz()),
        ("Readiness (/readyz)", lambda: client.readyz()),
        ("Version (/version)", lambda: client.version()),
        ("Metrics (/metrics)", lambda: client.metrics()),
        ("List Models (/api/models)", lambda: client.list_models()),
    ]

    results = []
    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            success = print_result(result, verbose)
            results.append({"test": name, "passed": success, "result": result})
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {name}: Exception - {e}")
            results.append({"test": name, "passed": False, "error": str(e)})
            failed += 1

    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("-" * 60)

    return {
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "results": results,
    }


def run_admin_checks(client: LongbowClient, verbose: bool = False) -> Dict[str, Any]:
    """Run administrative checks"""
    print("\n" + "=" * 60)
    print("ADMINISTRATIVE CHECKS")
    print("=" * 60 + "\n")

    print("🔍 System Information:")
    version = client.version()
    if "data" in version:
        data = version["data"]
        print(f"   Version: {data.get('version', 'unknown')}")
        print(f"   Go Version: {data.get('go_version', 'unknown')}")

    print("\n🔍 Health Status:")
    health = client.health()
    if "data" in health:
        data = health["data"]
        print(f"   Status: {data.get('status', 'unknown')}")
        print(f"   Uptime: {data.get('uptime', 'unknown')}")
        if "checks" in data:
            print("   Checks:")
            for check_name, check_data in data["checks"].items():
                status = check_data.get("status", "unknown")
                print(f"     - {check_name}: {status}")

    print("\n🔍 Model Availability:")
    models = client.list_models()
    if "data" in models:
        model_list = models["data"]
        if isinstance(model_list, list):
            if model_list:
                for model in model_list:
                    if isinstance(model, dict):
                        print(
                            f"   - {model.get('name', 'unknown')}: {model.get('parameters', 'unknown')}"
                        )
                    else:
                        print(f"   - {model}")
            else:
                print("   No models loaded")
        else:
            print(f"   Models: {model_list}")
    else:
        print("   Could not retrieve models")

    return {
        "version": version.get("data"),
        "health": health.get("data"),
        "models": models.get("data"),
    }


def run_benchmark(
    client: LongbowClient,
    prompt: str = "Hello, how are you?",
    num_requests: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run simple benchmark"""
    print("\n" + "=" * 60)
    print(f"BENCHMARK ({num_requests} requests)")
    print("=" * 60 + "\n")

    latencies = []
    tokens_generated = []
    tokens_per_sec = []

    for i in range(num_requests):
        start = time.time()
        result = client.generate(prompt, max_tokens=50)
        elapsed = time.time() - start

        if "data" in result:
            latencies.append(elapsed)
            tokens_generated.append(result["data"].get("tokens_generated", 0))
            tokens_per_sec.append(result["data"].get("tokens_per_sec", 0))

        if verbose:
            print(f"Request {i + 1}/{num_requests}: {elapsed:.3f}s")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        avg_tps = sum(tokens_per_sec) / len(tokens_per_sec)

        print(f"\nResults:")
        print(f"   Avg Latency: {avg_latency:.3f}s")
        print(f"   Avg Tokens: {avg_tokens:.1f}")
        print(f"   Avg Throughput: {avg_tps:.2f} tokens/sec")

        return {
            "requests": num_requests,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
            "avg_tokens_per_sec": avg_tps,
        }

    return {"error": "No successful requests"}


def test_websocket(client: LongbowClient, verbose: bool = False) -> Dict[str, Any]:
    """Test WebSocket endpoint (basic connectivity check)"""
    print("\n" + "=" * 60)
    print("WEBSOCKET TEST")
    print("=" * 60 + "\n")

    print(
        "Note: Full WebSocket testing requires websocket client library (e.g., websockets)"
    )
    print(f"WebSocket endpoint: {client.base_url}/ws")
    print("\nTo test manually:")
    print("  1. Connect to ws://localhost:8080/ws")
    print('  2. Send: {"type": "status"}')
    print('  3. Expect: {"type": "status", "payload": {...}}')

    return {
        "endpoint": f"{client.base_url}/ws",
        "tested": False,
        "note": "Manual test required",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Longbow-Quarrel API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_client.py health
  python api_client.py validate --verbose
  python api_client.py generate -p "Hello world"
  python api_client.py admin
  python api_client.py benchmark --requests 20
        """,
    )

    parser.add_argument(
        "--url",
        default=os.environ.get("LONGBOW_URL", "http://localhost:8080"),
        help="Base URL of the API (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LONGBOW_API_KEY"),
        help="API key for authentication",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    subparsers.add_parser("health", help="Check API health status")
    subparsers.add_parser("healthz", help="Simple liveness probe")
    subparsers.add_parser("readyz", help="Readiness probe")
    subparsers.add_parser("version", help="Get API version")
    subparsers.add_parser("metrics", help="Get Prometheus metrics")
    subparsers.add_parser("models", help="List available models")
    subparsers.add_parser("validate", help="Run full API validation suite")
    subparsers.add_parser("admin", help="Run administrative checks")
    subparsers.add_parser("websocket", help="Test WebSocket connection")

    gen_parser = subparsers.add_parser("generate", help="Generate text (synchronous)")
    gen_parser.add_argument("-p", "--prompt", required=True, help="Prompt text")
    gen_parser.add_argument("-m", "--model", default="default", help="Model name")
    gen_parser.add_argument(
        "-t", "--temperature", type=float, default=0.7, help="Temperature"
    )
    gen_parser.add_argument(
        "-n", "--max-tokens", type=int, default=256, help="Max tokens"
    )
    gen_parser.add_argument("--top-k", type=int, default=40, help="Top-K")
    gen_parser.add_argument("--top-p", type=float, default=0.95, help="Top-P")

    stream_parser = subparsers.add_parser("stream", help="Stream text generation")
    stream_parser.add_argument("-p", "--prompt", required=True, help="Prompt text")
    stream_parser.add_argument("-m", "--model", default="default", help="Model name")
    stream_parser.add_argument(
        "-t", "--temperature", type=float, default=0.7, help="Temperature"
    )
    stream_parser.add_argument(
        "-n", "--max-tokens", type=int, default=256, help="Max tokens"
    )

    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument(
        "-p", "--prompt", default="Hello, how are you?", help="Prompt text"
    )
    bench_parser.add_argument(
        "-r", "--requests", type=int, default=10, help="Number of requests"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    client = LongbowClient(
        base_url=args.url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    if args.command == "health":
        result = client.health()
        print_result(result, args.verbose)
    elif args.command == "healthz":
        result = client.healthz()
        print(result.get("data", {}).get("status", "OK"))
    elif args.command == "readyz":
        result = client.readyz()
        print(result.get("data", {}).get("status", "NOT READY"))
    elif args.command == "version":
        result = client.version()
        if "data" in result:
            print(json.dumps(result["data"], indent=2))
    elif args.command == "metrics":
        result = client.metrics()
        if "data" in result:
            print(result["data"])
    elif args.command == "models":
        result = client.list_models()
        if "data" in result:
            print(json.dumps(result["data"], indent=2))
    elif args.command == "generate":
        result = client.generate(
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        if "data" in result:
            print(json.dumps(result["data"], indent=2))
        else:
            print_result(result)
    elif args.command == "stream":
        tokens = client.stream_generate(
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(f"Received {len(tokens)} tokens:")
        for i, token in enumerate(tokens):
            print(f"  {i + 1}: {token}")
    elif args.command == "validate":
        validate_api(client, args.verbose)
    elif args.command == "admin":
        run_admin_checks(client, args.verbose)
    elif args.command == "benchmark":
        run_benchmark(client, args.prompt, args.requests, args.verbose)
    elif args.command == "websocket":
        test_websocket(client, args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
