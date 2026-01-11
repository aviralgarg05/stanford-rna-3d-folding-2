# Stanford RNA 3D Folding Competition - Part 2

Deep learning solution for predicting 3D RNA structures.

## Overview

- **Goal**: Predict 5 diverse 3D structures per RNA sequence
- **Metric**: TM-score (best of 5 predictions)
- **Prize**: $75,000
- **Deadline**: March 25, 2026

## Files

- `train_remote.ipynb` - Training notebook (runs on Kaggle GPU)
- `inference_submission.ipynb` - Generate submission.csv
- `explore_data.ipynb` - Data analysis (optional)

## Quick Start

1. Push training notebook to Kaggle:
```bash
./setup_and_push.sh
```

2. Monitor training:
```bash
kaggle kernels status aviralgarg05/stanford-rna-3d-folding-starter
```

3. Download trained model:
```bash
kaggle kernels output aviralgarg05/stanford-rna-3d-folding-starter -p ./
```

## Model Architecture

- Transformer encoder (6 layers, 8 heads, 256-dim)
- 5 prediction heads for diverse structures
- ~10M parameters

## Expected Performance

- Baseline TM-score: 0.3-0.5
- Training time: 2-4 hours on Kaggle GPU
