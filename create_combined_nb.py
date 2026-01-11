#!/usr/bin/env python3
"""Create a combined training + inference notebook."""
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
        "source": [line + "\n" for line in content.strip().split('\n')]
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code", 
        "metadata": {}, 
        "source": [line + "\n" for line in content.strip().split('\n')],
        "outputs": [], 
        "execution_count": None
    })

# Cell 1: Title
add_md("""# Stanford RNA 3D Folding - Complete Pipeline

Train model and generate submission in one notebook.""")

# Cell 2: Setup
add_code("""import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

log("Starting...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# Seed
torch.manual_seed(42)
np.random.seed(42)""")

# Cell 3: Config
add_code("""CONFIG = {
    'data_dir': '../input/stanford-rna-3d-folding-2',
    'max_len': 384,
    'batch_size': 16,
    'epochs': 15,
    'lr': 1e-3,
    'min_lr': 1e-5,
    'weight_decay': 0.01,
    'warmup_epochs': 2,
    'embed_dim': 256,
    'nhead': 8,
    'num_layers': 6,
    'num_predictions': 5,
    'dropout': 0.1,
}
log("Config ready")""")

# Cell 4: Load data
add_code("""log("Loading data...")
train_seq = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train_sequences.csv'))
test_seq = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test_sequences.csv'))
train_labels = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train_labels.csv'))

log(f"Train: {len(train_seq)}, Test: {len(test_seq)}")
log(f"Labels: {len(train_labels)}")""")

# Cell 5: Preprocess
add_code("""log("Preprocessing...")

# Build coordinates dictionary
train_labels['target_id'] = train_labels['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
coords_dict = {}
coord_values = []

for target_id, group in train_labels.groupby('target_id'):
    group = group.sort_values('resid')
    x = group['x_1'].values
    y = group['y_1'].values  
    z = group['z_1'].values
    coords = np.stack([x, y, z], axis=1).astype(np.float32)
    # Replace NaN with 0
    coords = np.nan_to_num(coords, nan=0.0)
    coords_dict[target_id] = coords
    coord_values.append(coords)

# Global normalization stats
all_coords = np.concatenate(coord_values, axis=0)
COORD_MEAN = np.nanmean(all_coords, axis=0)
COORD_STD = np.nanstd(all_coords, axis=0) + 1e-6

log(f"Coords: {len(coords_dict)} targets")
log(f"Mean: {COORD_MEAN}, Std: {COORD_STD}")

# Normalize
for k in coords_dict:
    coords_dict[k] = (coords_dict[k] - COORD_MEAN) / COORD_STD

del train_labels, coord_values, all_coords
gc.collect()""")

# Cell 6: Dataset
add_code("""class RNADataset(Dataset):
    def __init__(self, seq_df, coords_dict=None, max_len=384, is_test=False):
        self.seq_df = seq_df.reset_index(drop=True)
        self.coords_dict = coords_dict
        self.max_len = max_len
        self.is_test = is_test
        self.base2int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4, 'T': 3}
        
        if coords_dict and not is_test:
            valid = set(coords_dict.keys())
            self.seq_df = self.seq_df[self.seq_df['target_id'].isin(valid)].reset_index(drop=True)
    
    def __len__(self):
        return len(self.seq_df)
    
    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        seq = row['sequence']
        
        seq_ids = [self.base2int.get(c.upper(), 4) for c in seq]
        orig_len = min(len(seq_ids), self.max_len)
        
        # Pad sequence
        if len(seq_ids) > self.max_len:
            seq_ids = seq_ids[:self.max_len]
        else:
            seq_ids = seq_ids + [4] * (self.max_len - len(seq_ids))
        
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:orig_len] = True
        
        if self.is_test:
            return torch.tensor(seq_ids, dtype=torch.long), mask, orig_len, row['target_id']
        
        # Training - get coords
        coords = self.coords_dict[row['target_id']][:self.max_len].copy()
        if len(coords) < self.max_len:
            coords = np.pad(coords, ((0, self.max_len - len(coords)), (0, 0)))
        
        return (torch.tensor(seq_ids, dtype=torch.long),
                torch.tensor(coords, dtype=torch.float32),
                mask, orig_len)

train_dataset = RNADataset(train_seq, coords_dict, CONFIG['max_len'])
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=True, num_workers=0, drop_last=True)

log(f"Train batches: {len(train_loader)}")""")

# Cell 7: Model
add_code("""class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RNAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['embed_dim']
        self.embedding = nn.Embedding(5, d, padding_idx=4)
        self.pos_enc = PositionalEncoding(d, cfg['max_len'])
        self.dropout = nn.Dropout(cfg['dropout'])
        
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg['nhead'],
            dim_feedforward=d*4, dropout=cfg['dropout'], batch_first=True,
            norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, cfg['num_layers'])
        
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg['dropout']),
                         nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 3))
            for _ in range(cfg['num_predictions'])
        ])
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        return torch.stack([h(x) for h in self.heads], dim=3)

model = RNAModel(CONFIG).to(device)
log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")""")

# Cell 8: Training
add_code("""def train_model(model, loader, epochs):
    criterion = nn.MSELoss(reduction='none')
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs-CONFIG['warmup_epochs'], eta_min=CONFIG['min_lr'])
    
    log("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Warmup
        if epoch < CONFIG['warmup_epochs']:
            lr = CONFIG['lr'] * (epoch + 1) / CONFIG['warmup_epochs']
            for g in optimizer.param_groups: g['lr'] = lr
        
        for i, batch in enumerate(loader):
            seq, coords, mask, _ = batch
            seq, coords, mask = seq.to(device), coords.to(device), mask.to(device)
            
            optimizer.zero_grad()
            preds = model(seq, mask)  # (B, L, 3, 5)
            
            # Expand targets for 5 predictions
            targets = coords.unsqueeze(3).expand(-1, -1, -1, 5)
            
            # Masked loss
            loss = criterion(preds, targets)
            loss = loss.sum(dim=2)  # Sum xyz
            loss = (loss * mask.unsqueeze(2).float()).sum() / (mask.sum() * 5 + 1e-6)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                log(f"E{epoch+1} B{i}/{len(loader)}: {loss.item():.4f}")
        
        if epoch >= CONFIG['warmup_epochs']:
            scheduler.step()
        
        avg_loss = total_loss / len(loader)
        log(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model': model.state_dict(), 'mean': COORD_MEAN, 'std': COORD_STD}, 'model.pth')
            log("Saved best model!")
    
    return best_loss

best = train_model(model, train_loader, CONFIG['epochs'])
log(f"Training done! Best loss: {best:.4f}")""")

# Cell 9: Inference
add_code("""log("Loading best model for inference...")
ckpt = torch.load('model.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
COORD_MEAN = ckpt['mean']
COORD_STD = ckpt['std']
model.eval()

log("Generating predictions...")
test_dataset = RNADataset(test_seq, None, CONFIG['max_len'], is_test=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

results = []
with torch.no_grad():
    for i, (seq, mask, orig_len, target_id) in enumerate(test_loader):
        if i % 200 == 0:
            log(f"Progress: {i}/{len(test_loader)}")
        
        seq, mask = seq.to(device), mask.to(device)
        preds = model(seq, mask)  # (1, L, 3, 5)
        
        # Denormalize
        preds = preds[0, :orig_len[0]].cpu().numpy()  # (L, 3, 5)
        preds = preds * COORD_STD.reshape(1, 3, 1) + COORD_MEAN.reshape(1, 3, 1)
        
        for resid in range(orig_len[0]):
            row = {'ID': f"{target_id[0]}_{resid+1}"}
            for p in range(5):
                row[f'x_{p+1}'] = float(preds[resid, 0, p])
                row[f'y_{p+1}'] = float(preds[resid, 1, p])
                row[f'z_{p+1}'] = float(preds[resid, 2, p])
            results.append(row)

log(f"Generated {len(results)} predictions")""")

# Cell 10: Create submission
add_code("""submission = pd.DataFrame(results)
log(f"Submission shape: {submission.shape}")
print(submission.head())

# Check NaN
nan_count = submission.isna().sum().sum()
log(f"NaN values: {nan_count}")

submission.to_csv('submission.csv', index=False)
log(f"Saved submission.csv ({os.path.getsize('submission.csv')/1e6:.2f} MB)")

# Verify format
sample = pd.read_csv(os.path.join(CONFIG['data_dir'], 'sample_submission.csv'))
log(f"Sample shape: {sample.shape}, Our shape: {submission.shape}")
log(f"Columns match: {list(submission.columns) == list(sample.columns)}")

log("DONE!")""")

# Save notebook
with open('train_and_submit.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Created train_and_submit.ipynb")
