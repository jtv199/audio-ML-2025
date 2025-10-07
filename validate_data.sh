#!/bin/bash

# Freesound Audio Tagging 2019 - Data Validation Script
# Checks that all required files are in the correct locations

set -e

echo "=== Freesound Audio Tagging 2019 - Data Validation ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        ((ERRORS++))
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found directory: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing directory: $1"
        ((ERRORS++))
        return 1
    fi
}

# Function to count files and validate
check_file_count() {
    local dir=$1
    local expected=$2
    local description=$3

    if [ ! -d "$dir" ]; then
        echo -e "${RED}✗${NC} Directory not found: $dir"
        ((ERRORS++))
        return 1
    fi

    local actual=$(ls "$dir"/*.wav 2>/dev/null | wc -l)

    if [ "$actual" -eq "$expected" ]; then
        echo -e "${GREEN}✓${NC} $description: $actual files (expected: $expected)"
        return 0
    else
        echo -e "${RED}✗${NC} $description: $actual files (expected: $expected)"
        ((ERRORS++))
        return 1
    fi
}

# Function to validate CSV row count matches file count
validate_csv_files() {
    local csv=$1
    local dir=$2
    local description=$3

    if [ ! -f "$csv" ]; then
        echo -e "${RED}✗${NC} CSV not found: $csv"
        ((ERRORS++))
        return 1
    fi

    # Count rows in CSV (excluding header)
    local csv_count=$(($(wc -l < "$csv") - 1))

    # Count actual wav files
    local file_count=$(ls "$dir"/*.wav 2>/dev/null | wc -l)

    if [ "$csv_count" -eq "$file_count" ]; then
        echo -e "${GREEN}✓${NC} $description CSV matches files: $csv_count entries"
        return 0
    else
        echo -e "${RED}✗${NC} $description: CSV has $csv_count entries but found $file_count files"
        ((ERRORS++))
        return 1
    fi
}

echo "Checking required CSV files..."
check_file "input/train_curated.csv"
check_file "input/sample_submission.csv"

echo ""
echo "Checking data directories..."
check_dir "input/trn_curated"
check_dir "input/test"
check_dir "work"

echo ""
echo "Validating file counts..."

# Get expected counts from CSVs
if [ -f "input/train_curated.csv" ]; then
    TRAIN_EXPECTED=$(($(wc -l < "input/train_curated.csv") - 1))
    check_file_count "input/trn_curated" "$TRAIN_EXPECTED" "Training files"
fi

if [ -f "input/sample_submission.csv" ]; then
    TEST_EXPECTED=$(($(wc -l < "input/sample_submission.csv") - 1))
    check_file_count "input/test" "$TEST_EXPECTED" "Test files"
fi

echo ""
echo "Cross-validating CSV entries with actual files..."

validate_csv_files "input/train_curated.csv" "input/trn_curated" "Training data"
validate_csv_files "input/sample_submission.csv" "input/test" "Test data"

echo ""
echo "Checking for specific files mentioned in CSVs..."

# Check first file from train_curated.csv exists
if [ -f "input/train_curated.csv" ]; then
    FIRST_TRAIN=$(tail -n +2 "input/train_curated.csv" | head -n 1 | cut -d',' -f1)
    if [ -n "$FIRST_TRAIN" ]; then
        check_file "input/trn_curated/$FIRST_TRAIN"
    fi
fi

# Check first file from sample_submission.csv exists
if [ -f "input/sample_submission.csv" ]; then
    FIRST_TEST=$(tail -n +2 "input/sample_submission.csv" | head -n 1 | cut -d',' -f1)
    if [ -n "$FIRST_TEST" ]; then
        check_file "input/test/$FIRST_TEST"
    fi
fi

# Check the previously problematic file
if [ -f "input/sample_submission.csv" ]; then
    if grep -q "426eb1e0.wav" "input/sample_submission.csv"; then
        check_file "input/test/426eb1e0.wav"
    fi
fi

echo ""
echo "=== Validation Summary ==="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "Your data is properly configured and ready to use."
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    echo ""
    echo "Please fix the errors above before running the notebook."
    echo "You may need to re-extract the data files."
    exit 1
fi
