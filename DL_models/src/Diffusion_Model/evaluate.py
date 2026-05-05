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
import json

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
        plt.plot(future_x, samples[i].detach().cpu().numpy(), alpha=0.3)

    # median prediction
    plt.plot(future_x, samples.median(axis=0).values.detach().cpu().numpy(), label="Pred Median")

    # ground truth
    plt.plot(future_x, target, label="Ground Truth")

    plt.legend(loc="upper left")
    plt.title("Forecast")
    plt.savefig(save_path)
    plt.close()

def evaluate(model, scaler, test_loader, waveform_conditioning, clinical_conditioning, diff_scheduler, outputs_path, vitals, device, num_samples=10): #5

  model.eval()

  count = 0

  for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Evaluating: "):
    if os.path.exists(f'{outputs_path}/batch_{idx}'):
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

    os.makedirs(f'{outputs_path}/batch_{idx}', exist_ok=True)
    os.makedirs(f'{outputs_path}/batch_{idx}/figures', exist_ok=True)
    np.save(f'{outputs_path}/batch_{idx}/encoder_target.npy', encoder_target.detach().cpu().numpy())
    np.save(f'{outputs_path}/batch_{idx}/decoder_target.npy', decoder_target.detach().cpu().numpy())
    np.save(f'{outputs_path}/batch_{idx}/samples.npy', samples.detach().cpu().numpy())

    # === Median prediction ===
    median_pred = samples.median(dim=0).values

    # === Metrics ===
    decoder_target = decoder_target.cpu().numpy()
    decoder_target_inv = scaler.inverse(decoder_target)
    mae = compute_mae(median_pred.detach().cpu().numpy(), decoder_target_inv, vitals=vitals)
    crps = compute_crps(samples.detach().cpu().numpy(), decoder_target_inv, vitals=vitals)

    metric_dict = {}
    for vital in vitals:
      metric_dict[vital] = {
        "maes": mae[vital].tolist(),
        'crpss': crps[vital].tolist()
      }


    with open(f'{outputs_path}/batch_{idx}/metrics.json', 'w') as f:
      json.dump(metric_dict, f)

    # ==== Plot ====
    enocder_target = encoder_target[0].detach().cpu().numpy()
    enocder_target_inv = scaler.inverse(enocder_target)
    for i in range(len(vitals)):
      vital = vitals[i]
      x_past = enocder_target_inv[:, i]
      plot_forecast(x_past=x_past, samples=samples[:, 0, :, i], target=decoder_target_inv[0, :, i], save_path=f'{outputs_path}/batch_{idx}/figures/{vital}.png')




#######################
#       SETUP
#######################

#testing variables
use_pretrained_vital_encoder_weights = False # CHANGE

use_waveform_data = True # CHANGE
waveform_conditioning = False # CHANGE

use_clinical_data = False  # CHANGE
clinical_conditioning = False  # CHANGE

test_num = 7
modifier = ""
if use_waveform_data:
  modifier += "/waveform_data"
  if waveform_conditioning:
    modifier += "/waveform_conditioned"
    test_num = 1
  else:
    modifier += "/non_waveform_conditioned"
    test_num = 2
if use_clinical_data:
  modifier += "/clinical_data"
  if clinical_conditioning:
    modifier += "/clinical_conditioned"
    test_num = 2
  else:
    modifier += "/non_clinical_conditioned"
    test_num = 2
  
if use_pretrained_vital_encoder_weights:
  modifier += "/forecasing_with_pretrained_vital_encoders"
  test_num = 2


save_path = f'../../models/Diffusion{modifier}/test_{test_num}'
model_save_path = f'{save_path}/model.pth'

outputs_path = f'../../outputs/Diffusion{modifier}'
if modifier == "":
  outputs_path += "/non_pretrained_base_model"

batch_size = 512 #2048
if use_waveform_data:
  batch_size = 256
vitals = ['HR', 'RESP', 'SpO2']

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
  test_dataset = DiffusionTimeSeriesWaveformDataset(vital_scaler, waveform_scaler, include_waveform_data=waveform_conditioning, test=True)
elif use_clinical_data:
  train_vital_df, test_vital_df, train_clinical_df, test_clinical_df = process_clinical_dataset(test_mode=False) 

  #Normalize Data and Save Scaler
  print("loading datasets...")
  scaler = VitalScaler()
  scaler.fit(train_vital_df)
  joblib.dump(scaler, f'{save_path}/vital_scaler.pkl')

  test_vital_df = scaler.transform(test_vital_df)

  #Create Datasets
  test_dataset = DiffusionTimeSeriesClinicalConditionedDataset(test_vital_df, test_clinical_df, include_clinical_data=clinical_conditioning)
else:
  train_df, test_df = process_dataset(test_mode=False) 

  #Normalize Data and Save Scaler
  print("loading datasets...")
  scaler = VitalScaler()
  scaler.fit(train_df)
  joblib.dump(scaler, f'{save_path}/vital_scaler.pkl')

  test_df = scaler.transform(test_df)

  #Create Datasets
  test_dataset = DiffusionTimeSeriesDataset(test_df)


print(f'Test Examples: {len(test_dataset.windows)}')

#Create DataLoaders
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#2. Create Model
model = DiffusionForecaster(num_vitals=3, prediction_length=10, embed_dim=256, num_heads=4, num_layers=4, waveform_conditioning=waveform_conditioning, clinical_conditioning=clinical_conditioning, use_pretrained_vital_encoder_weights=use_pretrained_vital_encoder_weights, hr_encoder_pretrained_weights=hr_encoder_pretrained_weights, resp_encoder_pretrained_weights=resp_encoder_pretrained_weights, spO2_encoder_pretrained_weights=spO2_encoder_pretrained_weights) #256, 4, 4

#Load Weights
model_weights = torch.load(model_save_path, map_location=torch.device('cpu'))
if use_waveform_data and not waveform_conditioning:
  cleaned_model_weights = {}
  for key in model_weights.keys():
    if 'cross_waveform_fusion' not in key:
        cleaned_model_weights[key] = model_weights[key] 

  model_weights = cleaned_model_weights
  
model.load_state_dict(model_weights)
model.to(device)

#3. Evaluate
diff_scheduler = DiffusionScheduler(device, T=400)

#test step
evaluate(model=model, scaler=scaler, test_loader=test_loader, waveform_conditioning=waveform_conditioning, clinical_conditioning=clinical_conditioning, diff_scheduler=diff_scheduler, outputs_path=outputs_path, vitals=vitals, device=device)
