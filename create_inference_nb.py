#!/usr/bin/env python3
"""Create inference notebook for submission."""
import json

notebook = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": []
}

def add_md(content):
    notebook["cells"].append({
        "cell_type": "markdown", 
        "metadata": {}, 
        "source": [line + "\n" for line in content.split('\n')]
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code", 
        "metadata": {}, 
        "source": [line + "\n" for line in content.split('\n')],
        "outputs": [], 
        "execution_count": None
    })

# Cell 1: Title
add_md("""# RNA 3D Structure Prediction - Inference

Generate submission for Stanford RNA 3D Folding competition.""")

# Cell 2: Imports
add_code("""import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

print(f"[{time.strftime('%H:%M:%S')}] Starting inference...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")""")

# Cell 3: Config
add_code("""CONFIG = {
    'data_dir': '../input/stanford-rna-3d-folding-2',
    'model_path': '../input/stanford-rna-3d-folding-starter/model.pth',
    'max_len': 384,
    'vocab_size': 5,
    'embed_dim': 256,
    'nhead': 8,
    'num_layers': 6,
    'num_predictions': 5,
    'dropout': 0.0,
}

# Training normalization stats from logs
TRAIN_MEAN = np.array([158.24141, 156.78705, 154.46898])
TRAIN_STD = np.array([131.6201, 132.98083, 127.77188])

print("Config loaded")
print(f"Normalization - Mean: {TRAIN_MEAN}, Std: {TRAIN_STD}")""")

# Cell 4: Model
add_code("""class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RNAStructurePredictor(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=256, nhead=8, num_layers=6,
                 num_predictions=5, dropout=0.0, max_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_predictions = num_predictions
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=4)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
                nn.Linear(embed_dim // 2, 3)
            ) for _ in range(num_predictions)
        ])
    
    def forward(self, x, mask=None):
        x_embed = self.embedding(x)
        x_embed = self.pos_encoder(x_embed)
        x_embed = self.embed_dropout(x_embed)
        padding_mask = ~mask if mask is not None else None
        encoded = self.transformer(x_embed, src_key_padding_mask=padding_mask)
        predictions = [head(encoded) for head in self.prediction_heads]
        return torch.stack(predictions, dim=3)

print("Model class defined")""")

# Cell 5: Load model
add_code("""print(f"[{time.strftime('%H:%M:%S')}] Loading model...")

model = RNAStructurePredictor(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    nhead=CONFIG['nhead'],
    num_layers=CONFIG['num_layers'],
    num_predictions=CONFIG['num_predictions'],
    dropout=CONFIG['dropout'],
    max_len=CONFIG['max_len']
).to(device)

# Load checkpoint - weights_only=False for PyTorch 2.6+
ckpt = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")
print(f"[{time.strftime('%H:%M:%S')}] Model loaded!")""")

# Cell 6: Load test data
add_code("""print(f"[{time.strftime('%H:%M:%S')}] Loading test data...")

test_seq = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test_sequences.csv'))
print(f"Test sequences: {len(test_seq)}")
print(f"Columns: {test_seq.columns.tolist()}")
print(test_seq.head())""")

# Cell 7: Prediction function
add_code("""def predict_structure(sequence, model, device, max_len=384):
    base2int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4, 'T': 3}
    
    seq_ids = [base2int.get(c.upper(), 4) for c in sequence]
    orig_len = len(seq_ids)
    
    if len(seq_ids) > max_len:
        seq_ids = seq_ids[:max_len]
        orig_len = max_len
    else:
        seq_ids = seq_ids + [4] * (max_len - len(seq_ids))
    
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[:orig_len] = True
    
    seq_tensor = torch.tensor([seq_ids], dtype=torch.long).to(device)
    mask_tensor = mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(seq_tensor, mask_tensor)
    
    preds = predictions[0, :orig_len].cpu().numpy()
    
    # Denormalize
    for i in range(5):
        preds[:, :, i] = preds[:, :, i] * TRAIN_STD + TRAIN_MEAN
    
    return preds, orig_len

print("Prediction function ready")""")

# Cell 8: Generate predictions
add_code("""print(f"[{time.strftime('%H:%M:%S')}] Generating predictions...")

results = []
total = len(test_seq)

for idx, row in test_seq.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    
    if idx % 100 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] Progress: {idx}/{total}")
    
    preds, seq_len = predict_structure(sequence, model, device, CONFIG['max_len'])
    
    for resid in range(seq_len):
        row_id = f"{target_id}_{resid+1}"
        result = {'ID': row_id}
        
        for pred_idx in range(5):
            result[f'x_{pred_idx+1}'] = float(preds[resid, 0, pred_idx])
            result[f'y_{pred_idx+1}'] = float(preds[resid, 1, pred_idx])
            result[f'z_{pred_idx+1}'] = float(preds[resid, 2, pred_idx])
        
        results.append(result)

print(f"[{time.strftime('%H:%M:%S')}] Generated {len(results)} rows")""")

# Cell 9: Create submission
add_code("""print(f"[{time.strftime('%H:%M:%S')}] Creating submission...")

submission = pd.DataFrame(results)
print(f"Shape: {submission.shape}")
print(f"Columns: {submission.columns.tolist()}")
print(submission.head())

# Check for issues
nan_count = submission.isna().sum().sum()
print(f"NaN values: {nan_count}")

submission.to_csv('submission.csv', index=False)
print(f"[{time.strftime('%H:%M:%S')}] Saved submission.csv")
print(f"File size: {os.path.getsize('submission.csv') / 1e6:.2f} MB")""")

# Cell 10: Validate
add_code("""print(f"[{time.strftime('%H:%M:%S')}] Validating...")

sample_sub = pd.read_csv(os.path.join(CONFIG['data_dir'], 'sample_submission.csv'))
print(f"Sample shape: {sample_sub.shape}")
print(f"Our shape: {submission.shape}")
print(f"Columns match: {list(submission.columns) == list(sample_sub.columns)}")

print(f"\\n[{time.strftime('%H:%M:%S')}] Done!")""")

# Save notebook
with open('inference_submission.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Created inference_submission.ipynb")
