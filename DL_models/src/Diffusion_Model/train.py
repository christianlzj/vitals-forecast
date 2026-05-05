import numpy as np
import pandas as pd
from dataset import process_dataset, DiffusionTimeSeriesDataset, process_waveform_dataset, DiffusionTimeSeriesWaveformDataset, process_clinical_dataset, DiffusionTimeSeriesClinicalConditionedDataset
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from scaler import VitalScaler, WaveformScaler
import joblib
from tqdm import tqdm
from model import DiffusionForecaster
from noise_scheduler import DiffusionScheduler
import torch.nn.functional as F
from utils import compute_mae, compute_crps
import matplotlib.pyplot as plt
import os
import random

#######################
#       TRAIN
#######################
def train(model, train_loader, waveform_conditioning, clinical_conditioning, optimizer, diff_scheduler, model_save_path, train_state_save_path, epoch, epochs, device):
  model.train()
  total_loss = 0.0

  # train model for 1 epoch

  for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
    if waveform_conditioning:
      encoder_target, encoder_mask, encoder_delta, decoder_target, decoder_mask, decoder_delta, waveform_values = batch
    elif clinical_conditioning:
      encoder_target, encoder_mask, encoder_delta, decoder_target, decoder_mask, decoder_delta, clinical_embeddings = batch
    else:
      encoder_target, encoder_mask, encoder_delta, decoder_target, decoder_mask, decoder_delta = batch

    encoder_target = encoder_target.to(device)
    encoder_mask = encoder_mask.to(device)
    encoder_delta = encoder_delta.to(device)
    decoder_target = decoder_target.to(device)
    decoder_mask = decoder_mask.to(device)
    decoder_delta = decoder_delta.to(device)

    if waveform_conditioning:
      waveform_values = waveform_values.to(device)
    else:
      waveform_values = None

    if clinical_conditioning:
      clinical_embeddings = clinical_embeddings.to(device)
    else:
      clinical_embeddings = None


    B = encoder_target.shape[0]

    #1. Sample diffusion timestep
    t = diff_scheduler.sample_timesteps(B, device)

    #2. Sample noise
    noise = torch.randn_like(decoder_target, device=device)

    #3. Create noisy future
    noisy_decoder_target = diff_scheduler.add_noise(decoder_target, t, noise)

    # 4. Predict noise
    eps_hat = model(noisy_decoder_target, encoder_target, encoder_mask, encoder_delta, t, waveform_values, clinical_embeddings)

    # 4.2 Predict x_0
    alpha_bar = diff_scheduler.alpha_cumprod[t].view(-1,1,1)

    x0_pred = (
        noisy_decoder_target - torch.sqrt(1 - alpha_bar) * eps_hat
    ) / torch.sqrt(alpha_bar)

    # 5. loss
    valid_mask = 1 - decoder_mask
    noise_loss = ((eps_hat - noise) ** 2) * valid_mask
    noise_loss = noise_loss.sum() / (valid_mask.sum() + 1e-8)

    x_0_loss = F.mse_loss(x0_pred, decoder_target)

    loss = noise_loss + (0.5 * x_0_loss)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    total_loss += loss

  print(f'epoch {epoch} training checkpoint created')
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, 
  train_state_save_path)

  print('saving model to {}...'.format(model_save_path))
  torch.save(model.state_dict(), model_save_path)

  return total_loss / len(train_loader)


#######################
#      EVALUATE
#######################
def sample_future_ddim(
    model, scaler, encoder_target, encoder_mask, encoder_delta, diff_scheduler, waveform_values=None, clinical_embeddings=None, num_steps=50, eta=0.2):
    """
    DDIM sampler: can be deterministic (eta=0) or partially stochastic (0<eta<=1)
    """
    with torch.no_grad():
      alpha_cum = diff_scheduler.alpha_cumprod

      B = encoder_target.shape[0]
      T_future = model.prediction_length
      D = model.num_vitals

      # start from pure noise
      future = torch.randn(B, T_future, D, device=device)

      # augment starting point with last observed point
      alpha = 0.75
      last_val = encoder_target[:, -1, :].unsqueeze(1)
      for t in range(T_future):
        decay = alpha * (0.8 ** t)  # decay over time
        future[:, t, :] = decay * last_val[:, 0, :] + (1 - decay) * future[:, t, :]
      
      
      # DDIM timesteps (descending)
      ddim_timesteps = torch.linspace(diff_scheduler.T - 1, 0, num_steps, dtype=torch.long)

      for i, t in enumerate(ddim_timesteps):
          t_next = ddim_timesteps[i+1] if i+1 < len(ddim_timesteps) else 0

          t_batch = torch.full((B,), t, device=device, dtype=torch.long)
          
          # predict noise
          eps_hat = model(future, encoder_target, encoder_mask, encoder_delta, t_batch, waveform_values, clinical_embeddings)
          
          # predicted x_0
          alpha_t = alpha_cum[t]
          alpha_next = alpha_cum[t_next]
          x0_hat = (future - torch.sqrt(1 - alpha_t) * eps_hat) / torch.sqrt(alpha_t)
          
          # compute sigma for partial stochasticity
          sigma_t = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
          
          # next sample
          noise = torch.zeros_like(future) if eta == 0 else torch.randn_like(future)
          future = torch.sqrt(alpha_next) * x0_hat + torch.sqrt(1 - alpha_next - sigma_t**2) * eps_hat + sigma_t * noise

    #move to cpu + numpy
    future = future.cpu().numpy()

    #inverse transform
    future_inv = scaler.inverse(future)

    future_inv = torch.from_numpy(future_inv).to(device)

    return future_inv


def sample_future(model, scaler, encoder_target, encoder_mask, encoder_delta, diff_scheduler, waveform_values=None, clinical_embeddings=None, num_steps=50, eta=0.0):
    """
    encoder_target: [B, T_past, num_vitals]
    encoder_mask: [B, T_past, num_vitals]
    encoder_deltas: [B, T_past, num_vitals]
    returns: [B, T_future, num_vitals]
    """

    model.eval()
    device = encoder_target.device
    with torch.no_grad():

      B = encoder_target.shape[0]
      T_future = model.prediction_length
      D = model.num_vitals

      # start from pure noise
      future = torch.randn(B, T_future, D, device=device)

      for t in reversed(range(num_steps)):
          t_batch = torch.full((B,), t, device=device, dtype=torch.long)

          # predict noise
          eps_hat = model(future, encoder_target, encoder_mask, encoder_delta, t_batch, waveform_values, clinical_embeddings)

          alpha = diff_scheduler.alphas[t].to(device)
          alpha_bar = diff_scheduler.alpha_cumprod[t].to(device)

          # DDPM update step
          future = (1 / torch.sqrt(alpha)) * (
              future - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_hat
          )

          if t > 0:
              noise = torch.randn_like(future)
              beta = diff_scheduler.betas[t].to(device)
              future = future + torch.sqrt(beta) * noise


    #move to cpu + numpy
    future = future.cpu().numpy()

    #inverse transform
    future_inv = scaler.inverse(future)

    future_inv = torch.from_numpy(future_inv).to(device)

    return future_inv

def plot_forecast(x_past, samples, target, save_path):

    """
    x_past: [T_past]
    samples: [S, T_future]
    target: [T_future]
    """

    T_past = x_past.shape[0]
    T_future = target.shape[0]

    future_x = range(T_past, T_past + T_future)

    plt.figure(figsize=(10,5))

    # past
    plt.plot(range(T_past), x_past, label="Past", color="black")

    # samples
    for i in range(samples.shape[0]):
        plt.plot(future_x, samples[i].detach().cpu().numpy(), alpha=0.2)

    # median prediction
    plt.plot(future_x, samples.median(axis=0).values.detach().cpu().numpy(), label="Pred Median")

    # ground truth
    plt.plot(future_x, target, label="Ground Truth")

    plt.legend(loc="upper left")
    plt.title("Forecast")
    plt.savefig(save_path)
    plt.close()

def evaluate(model, scaler, test_loader, waveform_conditioning, clinical_conditioning, diff_scheduler, epoch, log_path, vitals, device, num_samples=10): #5

  model.eval()

  total_maes = {}
  total_crpss = {}
  for vital in vitals:
    total_maes[vital] = 0
    total_crpss[vital] = 0

  count = 0


  rand_idx = random.randint(1, len(test_loader)) - 1

  for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Evaluating: "):
    if idx > rand_idx or idx < rand_idx:
      continue

    if waveform_conditioning:
      encoder_target, encoder_mask, encoder_delta, decoder_target, decoder_mask, decoder_delta, waveform_values = batch
    elif clinical_conditioning:
      encoder_target, encoder_mask, encoder_delta, decoder_target, decoder_mask, decoder_delta, clinical_embeddings = batch
    else:
      encoder_target, encoder_mask, encoder_delta, decoder_target, decoder_mask, decoder_delta = batch

    encoder_target = encoder_target.to(device)
    encoder_mask = encoder_mask.to(device)
    encoder_delta = encoder_delta.to(device)
    decoder_target = decoder_target.to(device)
    decoder_mask = decoder_mask.to(device)
    decoder_delta = decoder_delta.to(device)

    if waveform_conditioning:
      waveform_values = waveform_values.to(device)
    else:
      waveform_values = None

    if clinical_conditioning:
      clinical_embeddings = clinical_embeddings.to(device)
    else:
      clinical_embeddings = None


    B = encoder_target.shape[0]

    # === Generate multiple samples ===
    samples = []
    for _ in range(num_samples):
        # pred = sample_future(model, scaler, encoder_target, encoder_mask, encoder_delta, diff_scheduler, num_steps=diff_scheduler.T)
        pred = sample_future_ddim(model, scaler, encoder_target, encoder_mask, encoder_delta, diff_scheduler, waveform_values, clinical_embeddings, num_steps=50, eta=0.0)
        samples.append(pred)

    samples = torch.stack(samples)  # [S, B, T, D]

    # === Median prediction ===
    median_pred = samples.median(dim=0).values

    # === Metrics ===
    decoder_target = decoder_target.cpu().numpy()
    decoder_target_inv = scaler.inverse(decoder_target)
    mae = compute_mae(median_pred.detach().cpu().numpy(), decoder_target_inv, vitals=vitals)
    crps = compute_crps(samples.detach().cpu().numpy(), decoder_target_inv, vitals=vitals)

    for vital in vitals:
      total_maes[vital] += np.mean(mae[vital])
      total_crpss[vital] += np.mean(crps[vital])

    count += 1

    # ==== Plot ====
    enocder_target = encoder_target[0].detach().cpu().numpy()
    enocder_target_inv = scaler.inverse(enocder_target)
    for i in range(len(vitals)):
      vital = vitals[i]
      x_past = enocder_target_inv[:, i]
      plot_forecast(x_past=x_past, samples=samples[:, 0, :, i], target=decoder_target_inv[0, :, i], save_path=f'{log_path}/figures/{vital}/epoch_{epoch}.png')

  mean_maes = {}
  mean_crpss = {}
  for vital in vitals:
    mean_maes[vital] = total_maes[vital] / count
    mean_crpss[vital] = total_crpss[vital] / count

  metrics = {
    'MAE': mean_maes,
    'CRPS': mean_crpss
  }
  return metrics


#######################
#       SETUP
#######################

#training variables
use_pretrained_vital_encoder_weights = True # CHANGE

use_waveform_data = False # CHANGE
waveform_conditioning = False # CHANGE

use_clinical_data = False  # CHANGE
clinical_conditioning = False  # CHANGE

modifier = ""
if use_waveform_data:
  modifier += "/waveform_data"
  if waveform_conditioning:
    modifier += "/waveform_conditioned"
  else:
    modifier += "/non_waveform_conditioned"
if use_clinical_data:
  modifier += "/clinical_data"
  if clinical_conditioning:
    modifier += "/clinical_conditioned"
  else:
    modifier += "/non_clinical_conditioned"
  
if use_pretrained_vital_encoder_weights:
  modifier += "/forecasing_with_pretrained_vital_encoders"

test_num = 2 #CHANGE - Vital Only: 7, Pretrained: 2, Clincal Data + Conditioned: 2, Clincal Data + Not Conditioned: 2
save_path = f'../../models/Diffusion{modifier}/test_{test_num}'
if not os.path.exists(save_path):
  os.makedirs(f'{save_path}', exist_ok=True)
train_state_save_path = f'{save_path}/train_checkpoint.pth'
model_save_path = f'{save_path}/model.pth'
batch_size = 512 #2048
if use_waveform_data:
  batch_size = 256
num_epochs = 50
vitals = ['HR', 'RESP', 'SpO2']

log_path = f'../../logs/Diffusion{modifier}/test_{test_num}'
if not os.path.exists(log_path):
  os.makedirs(f'{log_path}/figures', exist_ok=True)
  os.makedirs(f'{log_path}/figures/HR', exist_ok=True)
  os.makedirs(f'{log_path}/figures/RESP', exist_ok=True)
  os.makedirs(f'{log_path}/figures/SpO2', exist_ok=True)

#Pretrained vital encoder paths
hr_encoder_pretrained_weights = '../../models/Diffusion/pretrained_vital_encoders/test_1/HR/model.pth'
resp_encoder_pretrained_weights = '../../models/Diffusion/pretrained_vital_encoders/test_1/RESP/model.pth'
spO2_encoder_pretrained_weights = '../../models/Diffusion/pretrained_vital_encoders/test_1/SpO2/model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. Load and Process Data
if use_waveform_data:

  vital_df, waveform_df = process_waveform_dataset(waveform_conditioning=waveform_conditioning, test_mode=False) 
  
  #Normalize Data and Save Scaler
  print("loading datasets...")
  vital_scaler = VitalScaler()
  vital_scaler.fit(vital_df)
  scaler = vital_scaler
  joblib.dump(vital_scaler, f'{save_path}/vital_scaler.pkl')
  waveform_scaler = WaveformScaler()
  waveform_scaler.fit(waveform_df)
  joblib.dump(waveform_scaler, f'{save_path}/waveform_scaler.pkl')

  #Create Datasets
  train_dataset = DiffusionTimeSeriesWaveformDataset(vital_scaler, waveform_scaler, include_waveform_data=waveform_conditioning, test=False)
  test_dataset = DiffusionTimeSeriesWaveformDataset(vital_scaler, waveform_scaler, include_waveform_data=waveform_conditioning, test=True)
elif use_clinical_data:
  train_vital_df, test_vital_df, train_clinical_df, test_clinical_df = process_clinical_dataset(test_mode=False) 

  #Normalize Data and Save Scaler
  print("loading datasets...")
  scaler = VitalScaler()
  scaler.fit(train_vital_df)
  joblib.dump(scaler, f'{save_path}/vital_scaler.pkl')

  train_vital_df = scaler.transform(train_vital_df)
  test_vital_df = scaler.transform(test_vital_df)

  #Create Datasets
  train_dataset = DiffusionTimeSeriesClinicalConditionedDataset(train_vital_df, train_clinical_df, include_clinical_data=clinical_conditioning)
  test_dataset = DiffusionTimeSeriesClinicalConditionedDataset(test_vital_df, test_clinical_df, include_clinical_data=clinical_conditioning)
else:
  train_df, test_df = process_dataset(test_mode=False) 

  #Normalize Data and Save Scaler
  print("loading datasets...")
  scaler = VitalScaler()
  scaler.fit(train_df)
  joblib.dump(scaler, f'{save_path}/vital_scaler.pkl')

  train_df = scaler.transform(train_df)
  test_df = scaler.transform(test_df)

  #Create Datasets
  train_dataset = DiffusionTimeSeriesDataset(train_df)
  test_dataset = DiffusionTimeSeriesDataset(test_df)


print(f'Training Examples: {len(train_dataset.windows)}')

#Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#2. Create Model
model = DiffusionForecaster(num_vitals=3, prediction_length=10, embed_dim=256, num_heads=4, num_layers=4, waveform_conditioning=waveform_conditioning, clinical_conditioning=clinical_conditioning, use_pretrained_vital_encoder_weights=use_pretrained_vital_encoder_weights, hr_encoder_pretrained_weights=hr_encoder_pretrained_weights, resp_encoder_pretrained_weights=resp_encoder_pretrained_weights, spO2_encoder_pretrained_weights=spO2_encoder_pretrained_weights) #256, 4, 4
model.to(device)

optimizer = AdamW(
  model.parameters(), 
  lr=1e-4, 
  weight_decay=1e-4, 
  betas=(0.9, 0.999)
) 
if use_pretrained_vital_encoder_weights:
  non_vital_encoder_parameters = []
  for name, param in model.named_parameters():
      if 'hr_encoder' not in name and 'resp_encoder' not in name and 'spO2_encoder' not in name:
          non_vital_encoder_parameters.append(param)

  optimizer = AdamW([
    {
      "params":non_vital_encoder_parameters, 
      "lr":1e-4, 
      "weight_decay":1e-4, 
      "betas":(0.9, 0.999)
    },
    {
      "params":model.hr_encoder.parameters(), 
      "lr":1e-5, 
      "weight_decay":1e-4, 
      "betas":(0.9, 0.999)
    },
    {
      "params":model.resp_encoder.parameters(), 
      "lr":1e-5, 
      "weight_decay":1e-4, 
      "betas":(0.9, 0.999)
    },
    {
      "params":model.spO2_encoder.parameters(), 
      "lr":1e-5, 
      "weight_decay":1e-4, 
      "betas":(0.9, 0.999)
    }
  ]) 
else:
  optimizer = AdamW(
    model.parameters(), 
    lr=1e-4, 
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
diff_scheduler = DiffusionScheduler(device, T=400)

print('training...')
for epoch in range(start_epoch, num_epochs):
  #train step
  train_loss = train(model=model, train_loader=train_loader, waveform_conditioning=waveform_conditioning, clinical_conditioning=clinical_conditioning, optimizer=optimizer, diff_scheduler=diff_scheduler, model_save_path=model_save_path, train_state_save_path=train_state_save_path, epoch=epoch, epochs=num_epochs, device=device)

  #test step
  test_metrics = evaluate(model=model, scaler=scaler, test_loader=test_loader, waveform_conditioning=waveform_conditioning, clinical_conditioning=clinical_conditioning, diff_scheduler=diff_scheduler, epoch=epoch, log_path=log_path, vitals=vitals, device=device)


  log_string = f'Epoch {epoch}: Train Loss - {train_loss}'
  for metric in test_metrics.keys():
    for vital in vitals:
      log_string += f' || Test {vital} {metric} - {test_metrics[metric][vital]}'

  print(log_string)

  with open(f'{log_path}/metric_log.txt', 'a') as f:
      f.write(f'{log_string}\n')







