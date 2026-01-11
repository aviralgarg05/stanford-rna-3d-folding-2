#!/bin/bash

# Monitor Kaggle kernel training progress

KERNEL_ID="aviralgarg05/stanford-rna-3d-folding-starter"

echo "Monitoring kernel: $KERNEL_ID"
echo "=================================="

while true; do
    clear
    echo "Time: $(date)"
    echo "=================================="
    
    # Get status
    STATUS=$(kaggle kernels status $KERNEL_ID 2>&1)
    echo "$STATUS"
    echo ""
    
    # Check if complete or error
    if echo "$STATUS" | grep -q "complete"; then
        echo "✓ Kernel execution complete!"
        echo ""
        echo "Downloading output..."
        kaggle kernels output $KERNEL_ID -p ./output
        echo ""
        echo "Files downloaded to ./output/"
        ls -lh ./output/
        break
    elif echo "$STATUS" | grep -q "error"; then
        echo "✗ Kernel execution failed!"
        echo ""
        echo "Downloading logs..."
        kaggle kernels output $KERNEL_ID -p ./output
        break
    fi
    
    # Wait before next check
    echo "Checking again in 60 seconds..."
    sleep 60
done
