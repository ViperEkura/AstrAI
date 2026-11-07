#!/bin/bash

# AstrAI Pre-commit Check Script
# Runs code format check and tests before committing

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if in project root directory
check_project_root() {
    if [ ! -f "pyproject.toml" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
}

# Check if Python is installed
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed"
        exit 1
    fi
    print_info "Python version: $(python --version)"
}

# Install development dependencies
install_dependencies() {
    print_info "Installing development dependencies..."
    pip install --upgrade pip
    pip install .[dev]
    print_success "Dependencies installed"
}

# Run code format check
run_lint() {
    print_info "Running code format check (ruff format)..."
    if ruff format --check .; then
        print_success "Code format check passed"
    else
        print_error "Code format check failed. Please run 'ruff format .' to fix formatting issues"
        exit 1
    fi
}

# Run code style check (linter - import sorting)
run_ruff_lint_import() {
    print_info "Running import sorting check (ruff check --select I)..."
    if ruff check . --select I; then
        print_success "Import sorting check passed"
    else
        print_error "Import sorting check failed. Please run 'ruff check --select I --fix .' to fix import issues"
        exit 1
    fi
}

# Run tests
run_tests() {
    print_info "Running tests..."
    if python -m pytest tests -q; then
        print_success "All tests passed"
    else
        print_error "Tests failed"
        exit 1
    fi
}

# Main function
main() {
    echo "========================================"
    echo "    AstrAI Pre-commit Check Script"
    echo "========================================"
    echo ""

    check_project_root
    check_python

    # Parse arguments
    SKIP_DEPS=false
    SKIP_LINT=false
    SKIP_TESTS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-lint)
                SKIP_LINT=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-deps    Skip dependency installation"
                echo "  --skip-lint    Skip code checks"
                echo "  --skip-tests   Skip tests"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Install dependencies
    if [ "$SKIP_DEPS" = false ]; then
        install_dependencies
    else
        print_info "Skipping dependency installation"
    fi

    # Run code checks
    if [ "$SKIP_LINT" = false ]; then
        run_lint
        run_ruff_lint_import
    else
        print_info "Skipping code checks"
    fi

    # Run tests
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    else
        print_info "Skipping tests"
    fi

    echo ""
    echo "========================================"
    print_success "All checks passed! Ready to commit."
    echo "========================================"
}

main "$@"
