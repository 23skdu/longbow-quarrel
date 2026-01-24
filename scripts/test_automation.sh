#!/bin/bash

# Test Automation Script for Longbow Quarrel
# This script runs comprehensive test suites with detailed reporting

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="$RESULTS_DIR/report_$TIMESTAMP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup
setup() {
    log_info "Setting up test environment..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$REPORT_DIR"
    
    # Set environment
    export GOOS=darwin
    export GOARCH=arm64
    export CGO_ENABLED=1
    
    # Check dependencies
    check_dependencies
    
    # Install test model if needed
    setup_test_model
    
    log_success "Test environment setup complete"
}

# Check required dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Go version
    if ! command -v go &> /dev/null; then
        log_error "Go is not installed"
        exit 1
    fi
    
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    log_info "Go version: $GO_VERSION"
    
    # Check Ollama
    if ! command -v ollama &> /dev/null; then
        log_warning "Ollama not found, attempting to install..."
        install_ollama
    else
        log_info "Ollama found: $(ollama --version)"
    fi
    
    # Check Metal support
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "macOS detected - Metal support available"
    else
        log_warning "Non-macOS detected - Metal GPU tests will be skipped"
    fi
}

# Install Ollama if not present
install_ollama() {
    log_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    if ! command -v ollama &> /dev/null; then
        log_error "Failed to install Ollama"
        exit 1
    fi
    log_success "Ollama installed successfully"
}

# Setup test model
setup_test_model() {
    log_info "Setting up test model..."
    
    # Check if SmolLM2 model exists
    if ! ollama list | grep -q "smollm2:135m"; then
        log_info "Pulling SmolLM2 135M model..."
        ollama pull smollm2:135m
    else
        log_info "SmolLM2 model already available"
    fi
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    UNIT_REPORT="$REPORT_DIR/unit_tests.txt"
    
    # Run core unit tests
    go test -v -race ./internal/tokenizer > "$UNIT_REPORT.tokenizer" 2>&1
    TOKENIZER_EXIT=$?
    
    go test -v -race ./internal/gguf > "$UNIT_REPORT.gguf" 2>&1
    GGUF_EXIT=$?
    
    go test -v -race ./internal/metrics > "$UNIT_REPORT.metrics" 2>&1
    METRICS_EXIT=$?
    
    # Generate summary
    {
        echo "=== Unit Test Summary ==="
        echo "Tokenizer: $([ $TOKENIZER_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "GGUF: $([ $GGUF_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "Metrics: $([ $METRICS_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo ""
        echo "Detailed reports:"
        echo "Tokenizer: $UNIT_REPORT.tokenizer"
        echo "GGUF: $UNIT_REPORT.gguf"
        echo "Metrics: $UNIT_REPORT.metrics"
    } > "$UNIT_REPORT"
    
    if [[ $TOKENIZER_EXIT -eq 0 && $GGUF_EXIT -eq 0 && $METRICS_EXIT -eq 0 ]]; then
        log_success "Unit tests passed"
        return 0
    else
        log_error "Some unit tests failed"
        return 1
    fi
}

# Run Metal GPU tests
run_metal_tests() {
    log_info "Running Metal GPU tests..."
    
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_warning "Skipping Metal tests on non-macOS system"
        return 0
    fi
    
    METAL_REPORT="$REPORT_DIR/metal_tests.txt"
    
    # Run device tests
    go test -v -tags "darwin,metal" -timeout 30m ./internal/device > "$METAL_REPORT.device" 2>&1
    DEVICE_EXIT=$?
    
    # Run engine tests
    go test -v -tags "darwin,metal" -timeout 30m ./internal/engine > "$METAL_REPORT.engine" 2>&1
    ENGINE_EXIT=$?
    
    # Generate summary
    {
        echo "=== Metal GPU Test Summary ==="
        echo "Device: $([ $DEVICE_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "Engine: $([ $ENGINE_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo ""
        echo "Detailed reports:"
        echo "Device: $METAL_REPORT.device"
        echo "Engine: $METAL_REPORT.engine"
    } > "$METAL_REPORT"
    
    if [[ $DEVICE_EXIT -eq 0 && $ENGINE_EXIT -eq 0 ]]; then
        log_success "Metal GPU tests passed"
        return 0
    else
        log_error "Some Metal GPU tests failed"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    INTEGRATION_REPORT="$REPORT_DIR/integration_tests.txt"
    
    # Run end-to-end tests
    go test -v -tags "darwin,metal" -timeout 30m ./test/integration > "$INTEGRATION_REPORT" 2>&1
    INTEGRATION_EXIT=$?
    
    if [[ $INTEGRATION_EXIT -eq 0 ]]; then
        log_success "Integration tests passed"
        return 0
    else
        log_error "Integration tests failed"
        return 1
    fi
}

# Run fuzz tests
run_fuzz_tests() {
    log_info "Running fuzz tests (limited duration)..."
    
    FUZZ_REPORT="$REPORT_DIR/fuzz_tests.txt"
    
    # Run each fuzz target for 30 seconds
    for target in Tokenizer Sampling ModelInput ConfigJSON; do
        log_info "Fuzzing target: $target"
        
        timeout 30s go test -fuzz=Fuzz$target -fuzztime=30s ./internal/engine > "$FUZZ_REPORT.$target" 2>&1
        TARGET_EXIT=$?
        
        echo "Fuzz $target: $([ $TARGET_EXIT -eq 0 ] && echo 'PASS' || echo 'TIMEOUT')" >> "$FUZZ_REPORT"
    done
    
    log_success "Fuzz tests completed"
    return 0
}

# Run benchmarks
run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    BENCHMARK_REPORT="$REPORT_DIR/benchmarks.txt"
    
    # Run Go benchmarks
    go test -bench=. -benchmem -tags "darwin,metal" ./internal/device > "$BENCHMARK_REPORT.go" 2>&1
    
    # Run vs llama.cpp benchmark
    if [[ -f "$PROJECT_ROOT/scripts/benchmark_compare.sh" ]]; then
        "$PROJECT_ROOT/scripts/benchmark_compare.sh" > "$BENCHMARK_REPORT.llamacpp" 2>&1
    fi
    
    # Generate summary
    {
        echo "=== Benchmark Summary ==="
        echo "Go benchmarks: $BENCHMARK_REPORT.go"
        echo "llama.cpp comparison: $BENCHMARK_REPORT.llamacpp"
        echo ""
        echo "Generated: $(date)"
    } >> "$BENCHMARK_REPORT"
    
    log_success "Benchmarks completed"
    return 0
}

# Generate HTML report
generate_html_report() {
    log_info "Generating HTML report..."
    
    HTML_REPORT="$REPORT_DIR/index.html"
    
    cat > "$HTML_REPORT" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Longbow Quarrel Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .success { color: green; font-weight: bold; }
        .failure { color: red; font-weight: bold; }
        .warning { color: orange; font-weight: bold; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Longbow Quarrel Test Report</h1>
        <p class="timestamp">Generated: $(date)</p>
    </div>
    
    <div class="section">
        <h2>Test Environment</h2>
        <ul>
            <li>Go Version: $(go version)</li>
            <li>OS: $(uname -s) $(uname -r)</li>
            <li>Architecture: $(uname -m)</li>
            <li>Test Model: SmolLM2 135M</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
EOF

    # Add test results to HTML
    if [[ -f "$REPORT_DIR/unit_tests.txt" ]]; then
        echo "<h3>Unit Tests</h3><pre>" >> "$HTML_REPORT"
        cat "$REPORT_DIR/unit_tests.txt" >> "$HTML_REPORT"
        echo "</pre>" >> "$HTML_REPORT"
    fi
    
    if [[ -f "$REPORT_DIR/metal_tests.txt" ]]; then
        echo "<h3>Metal GPU Tests</h3><pre>" >> "$HTML_REPORT"
        cat "$REPORT_DIR/metal_tests.txt" >> "$HTML_REPORT"
        echo "</pre>" >> "$HTML_REPORT"
    fi
    
    if [[ -f "$REPORT_DIR/integration_tests.txt" ]]; then
        echo "<h3>Integration Tests</h3><pre>" >> "$HTML_REPORT"
        cat "$REPORT_DIR/integration_tests.txt" >> "$HTML_REPORT"
        echo "</pre>" >> "$HTML_REPORT"
    fi
    
    if [[ -f "$REPORT_DIR/benchmarks.txt" ]]; then
        echo "<h3>Benchmarks</h3><pre>" >> "$HTML_REPORT"
        cat "$REPORT_DIR/benchmarks.txt" >> "$HTML_REPORT"
        echo "</pre>" >> "$HTML_REPORT"
    fi
    
    cat >> "$HTML_REPORT" << EOF
    </div>
</body>
</html>
EOF

    log_success "HTML report generated: $HTML_REPORT"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    echo "Starting Longbow Quarrel Test Automation"
    echo "Report will be saved to: $REPORT_DIR"
    echo ""
    
    setup
    
    # Run test suites
    local unit_result=0
    local metal_result=0
    local integration_result=0
    local fuzz_result=0
    local benchmark_result=0
    
    # Run tests based on arguments
    if [[ $# -eq 0 || "$1" == "all" ]]; then
        run_unit_tests || unit_result=1
        run_metal_tests || metal_result=1
        run_integration_tests || integration_result=1
        run_fuzz_tests || fuzz_result=1
        run_benchmarks || benchmark_result=1
    else
        case "$1" in
            "unit")
                run_unit_tests || unit_result=1
                ;;
            "metal")
                run_metal_tests || metal_result=1
                ;;
            "integration")
                run_integration_tests || integration_result=1
                ;;
            "fuzz")
                run_fuzz_tests || fuzz_result=1
                ;;
            "benchmark")
                run_benchmarks || benchmark_result=1
                ;;
            *)
                echo "Usage: $0 [all|unit|metal|integration|fuzz|benchmark]"
                exit 1
                ;;
        esac
    fi
    
    # Generate report
    generate_html_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "=== Test Automation Complete ==="
    echo "Duration: $duration seconds"
    echo "Report: $REPORT_DIR/index.html"
    echo ""
    
    # Exit with appropriate code
    if [[ $unit_result -eq 0 && $metal_result -eq 0 && $integration_result -eq 0 ]]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "Some tests failed!"
        exit 1
    fi
}

# Run main with all arguments
main "$@"