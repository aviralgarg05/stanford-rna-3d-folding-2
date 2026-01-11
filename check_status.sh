#!/bin/bash

# Comprehensive status check for Kaggle training

echo "=================================="
echo "Kaggle Training Status Check"
echo "Time: $(date)"
echo "=================================="

# Check kernel status
echo -e "\n1. Kernel Status:"
kaggle kernels status aviralgarg05/stanford-rna-3d-folding-starter

# Try to get logs
echo -e "\n2. Attempting to fetch logs:"
rm -rf /tmp/kaggle_check
kaggle kernels output aviralgarg05/stanford-rna-3d-folding-starter --path /tmp/kaggle_check 2>&1

# List what we got
echo -e "\n3. Files downloaded:"
ls -lah /tmp/kaggle_check/ 2>/dev/null || echo "No files available yet"

# Show log content if available
if [ -d "/tmp/kaggle_check" ]; then
    echo -e "\n4. Log content:"
    find /tmp/kaggle_check -name "*.log" -exec echo "File: {}" \; -exec head -50 {} \; 2>/dev/null
fi

echo -e "\n=================================="
echo "Check complete"
echo "=================================="
