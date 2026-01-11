#!/bin/bash
echo "Pushing kernel to Kaggle..."
kaggle kernels push -p .
echo "Check status checks: kaggle kernels status aviralgarg05/stanford-rna-3d-folding-starter"
