#!/bin/bash

# RSTorch Test Runner Script
# Runs all tests with organized output and timing

set -e

echo "🦀 RSTorch Test Suite Runner"
echo "============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run tests with timing
run_test_category() {
    local category=$1
    local pattern=$2
    local description=$3
    
    echo -e "${BLUE}📋 Running $category Tests${NC}"
    echo -e "   $description"
    echo "   Pattern: $pattern"
    echo ""
    
    start_time=$(date +%s)
    
    if cargo test $pattern -- --nocapture; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}✅ $category tests passed${NC} (${duration}s)"
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${RED}❌ $category tests failed${NC} (${duration}s)"
        return 1
    fi
    
    echo ""
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Please run this script from the rstorch root directory${NC}"
    exit 1
fi

# Print some system info
echo "🔧 Environment Info:"
echo "   Rust version: $(rustc --version)"
echo "   Cargo version: $(cargo --version)"
echo "   Working directory: $(pwd)"
echo ""

# Build first to catch compilation errors
echo -e "${BLUE}🔨 Building project...${NC}"
if cargo build; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi
echo ""

total_start=$(date +%s)

# Parse command line arguments
case "${1:-all}" in
    "unit")
        echo -e "${YELLOW}Running Unit Tests Only${NC}"
        echo ""
        run_test_category "Unit" "--lib" "Tests individual components in isolation"
        ;;
    
    "integration") 
        echo -e "${YELLOW}Running Integration Tests Only${NC}"
        echo ""
        run_test_category "Integration" "--test integration" "Tests components working together"
        ;;
    
    "autograd")
        echo -e "${YELLOW}Running Autograd Tests Only${NC}"
        echo ""
        run_test_category "Autograd" "autograd" "Tests automatic differentiation"
        ;;
    
    "tensor")
        echo -e "${YELLOW}Running Tensor Tests Only${NC}"
        echo ""
        run_test_category "Tensor" "tensor" "Tests tensor operations"
        ;;
    
    "nn")
        echo -e "${YELLOW}Running Neural Network Tests Only${NC}"
        echo ""
        run_test_category "Neural Network" "nn" "Tests neural network components"
        ;;
    
    "quick")
        echo -e "${YELLOW}Running Quick Test Suite (Essential Tests Only)${NC}"
        echo ""
        run_test_category "Tensor Unit" "test_tensor_unit" "Core tensor functionality"
        run_test_category "NN Unit" "test_nn_unit" "Core neural network functionality"
        run_test_category "Training Integration" "test_full_training" "End-to-end training"
        ;;
    
    "all"|*)
        echo -e "${YELLOW}Running Complete Test Suite${NC}"
        echo ""
        
        # Run unit tests
        echo -e "${BLUE}🧪 UNIT TESTS${NC}"
        echo "=============="
        run_test_category "Tensor Unit" "test_tensor_unit" "Individual tensor operations"
        run_test_category "NN Unit" "test_nn_unit" "Individual neural network components"  
        run_test_category "Optimizer Unit" "test_optim_unit" "Individual optimizer components"
        run_test_category "Autograd Unit" "autograd_test" "Individual autograd functions"
        run_test_category "Operations Unit" "test_operations" "Basic tensor operations"
        run_test_category "Autograd Extended" "test_autograd" "Extended autograd verification"
        
        echo -e "${BLUE}🚀 INTEGRATION TESTS${NC}"
        echo "===================="
        run_test_category "Full Training" "test_full_training" "Complete training workflows"
        run_test_category "NN Components" "test_nn_components" "Neural network integration"
        run_test_category "Dataset" "test_dataset" "Data loading and processing"
        
        # Skip distributed tests if MPI not available
        if command -v mpirun &> /dev/null; then
            run_test_category "Distributed" "distributed" "Distributed training features"
        else
            echo -e "${YELLOW}⚠️  Skipping distributed tests (MPI not available)${NC}"
            echo ""
        fi
        ;;
esac

total_end=$(date +%s)
total_duration=$((total_end - total_start))

echo "================================"
echo -e "${GREEN}🎉 All requested tests completed!${NC}"
echo -e "   Total time: ${total_duration}s"
echo ""

# Run a quick smoke test to verify basic functionality
echo -e "${BLUE}🔍 Quick Smoke Test${NC}"
echo "=================="
if cargo test test_tensor_creation --lib -- --exact --nocapture; then
    echo -e "${GREEN}✅ Smoke test passed - basic functionality working${NC}"
else
    echo -e "${RED}❌ Smoke test failed - basic functionality broken${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🚀 RSTorch is ready to use!${NC}"
echo ""
echo "Next steps:"
echo "  • Run 'cargo test' to run all tests"
echo "  • Run './run_tests.sh quick' for faster testing"
echo "  • Run './run_tests.sh unit' for unit tests only"
echo "  • Run './run_tests.sh integration' for integration tests only"
echo "  • Check tests/README.md for detailed testing documentation"