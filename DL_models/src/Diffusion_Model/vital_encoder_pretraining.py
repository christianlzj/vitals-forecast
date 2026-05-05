from model import VitalsEncoder
from pretraining_dataset import process_dataset, VitalEncoderPretrainingDataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import joblib
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from scaler import VitalScaler

def random_block_mask(values, mask_ratio=0.5, block_size=10):
    B, T, C = values.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=values.device)

    num_blocks = int(T * mask_ratio / block_size)

    for b in range(B):
        for _ in range(num_blocks):
            start = torch.randint(0, T - block_size, (1,))
            mask[b, start:start+block_size] = True

    mask = mask.unsqueeze(-1)
    corrupted = values.clone()
    corrupted[mask] = 0

    return corrupted, mask


def mae_loss(recon, target, mae_mask, obs_mask):
    """
    recon: [B, T, 1]
    target: [B, T, 1]
    mae_mask: [B, T, 1] (True where masked)
    """

    valid_mask = mae_mask & (obs_mask.bool())

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=recon.device)

    loss = (recon - target) ** 2

    loss = loss[valid_mask]  # only masked positions

    return loss.mean()

class VitalsMAE(nn.Module):
    def __init__(self, num_vitals, embed_dim, n_head=4, n_layers=4):
        super().__init__()

        self.vital_encoder = VitalsEncoder(num_vitals, embed_dim)

        # lightweight decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1) 
        )
    
    def forward(self, values, masks, deltas):
        """
        values: [B, T, 1]
        masks:  [B, T, 1]
        deltas: [B, T, 1]
        """

        # Step 1: apply random masking
        corrupted_values, mae_mask = random_block_mask(values)

        # Step 2: encode corrupted input
        tokens = self.vital_encoder(corrupted_values, masks, deltas)  # [B, T, D]

        # Step 3: decode
        recon = self.decoder(tokens)  # [B, T, 1]

        # Step 4: compute loss
        loss = mae_loss(recon, values, mae_mask, obs_mask=masks)

        return recon, loss


def pretrain_vital_encoders(vital_mae_model, vital_loader, vital_optimizer, model_save_path, train_state_save_path, epoch, epochs, device):
    total_loss = 0.0

    for idx, batch in tqdm(enumerate(vital_loader), total=len(vital_loader), desc=f'Epoch {epoch+1}/{epochs}'):
        values, masks, deltas = batch # [B, T, 1]

        values = values.to(device)
        masks = masks.to(device)
        deltas = deltas.to(device)

        _, loss = vital_mae_model(values, masks, deltas)

        total_loss += loss.item()

        vital_optimizer.zero_grad()
        loss.backward()
        vital_optimizer.step()
    
    print(f'epoch {epoch} training checkpoint created')
    torch.save({
        'epoch': epoch,
        'model_state_dict': vital_mae_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss / len(vital_loader),
        }, 
    train_state_save_path)

    print('saving model to {}...'.format(model_save_path))
    torch.save(vital_mae_model.state_dict(), model_save_path)

    return total_loss / len(vital_loader)


#######################
#       SETUP
#######################

#training variables
vital = 'RESP' #CHANGE
test_num = 1
save_path = f'../../models/Diffusion/pretrained_vital_encoders/test_{test_num}/{vital}'
if not os.path.exists(save_path):
  os.makedirs(f'{save_path}', exist_ok=True)
train_state_save_path = f'{save_path}/train_checkpoint.pth'
model_save_path = f'{save_path}/model.pth'
batch_size = 512 #2048
num_epochs = 50

log_path = f'../../logs/Diffusion/pretrained_vital_encoders/test_{test_num}/{vital}'
if not os.path.exists(log_path):
  os.makedirs(f'{log_path}', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. Load and Process Data
df = process_dataset(vital=vital) 

#Normalize Data and Save Scaler
print("loading datasets...")
scaler = VitalScaler()
scaler.fit(df, cols=[vital])
joblib.dump(scaler, f'{save_path}/vital_scaler.pkl')

df = scaler.transform(df)

#Create Datasets
dataset = VitalEncoderPretrainingDataset(df, vital=vital)


print(f'Training Examples: {len(dataset.windows)}')

#Create DataLoaders
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#2. Create Model
model = VitalsMAE(num_vitals=1, embed_dim=256) #256, 4, 4
model.to(device)

optimizer = AdamW(
  model.parameters(), 
  lr=3e-6, 
  weight_decay=1e-4, 
  betas=(0.9, 0.999)
) 

#Load Weights if Resuming
if os.path.exists(train_state_save_path):
    checkpoint = torch.load(train_state_save_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    model.to(device)
    print(f'Resuming from Epoch {start_epoch}...')
else:
    start_epoch = 0
    with open(f'{log_path}/metric_log.txt', "w") as file:
      file.write('')

#3. Train
print('training...')
for epoch in range(start_epoch, num_epochs):
  #train step
  train_loss = pretrain_vital_encoders(vital_mae_model=model, vital_loader=loader, vital_optimizer=optimizer, model_save_path=model_save_path, train_state_save_path=train_state_save_path, epoch=epoch, epochs=num_epochs, device=device)

  log_string = f'Epoch {epoch}: Train Loss - {train_loss}'

  print(log_string)

  with open(f'{log_path}/metric_log.txt', 'a') as f:
      f.write(f'{log_string}\n')