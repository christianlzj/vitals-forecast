import numpy as np
import pandas as pd
from dataset import process_dataset, DiffusionTimeSeriesDataset
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from scaler import VitalScaler
import joblib
from tqdm import tqdm
from model import DiffusionForecaster
from noise_scheduler import DiffusionScheduler
import torch.nn.functional as F
from utils import compute_mae, compute_crps
import matplotlib.pyplot as plt
import os

#######################
#      EVALUATE
#######################
def sample_future_ddim(
    model, scaler, encoder_target, encoder_mask, encoder_delta, diff_scheduler, num_steps=50, eta=0.2):
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
      alpha = 0.75 #0.4
      last_val = encoder_target[:, -1, :].unsqueeze(1)
      for t in range(T_future):
        decay = alpha * (0.8 ** t)  # decay over time, 0.7
        future[:, t, :] = decay * last_val[:, 0, :] + (1 - decay) * future[:, t, :]
      
      
      # DDIM timesteps (descending)
      ddim_timesteps = torch.linspace(diff_scheduler.T - 1, 0, num_steps, dtype=torch.long)

      for i, t in enumerate(ddim_timesteps):
          t_next = ddim_timesteps[i+1] if i+1 < len(ddim_timesteps) else 0

          t_batch = torch.full((B,), t, device=device, dtype=torch.long)
          
          # predict noise
          eps_hat = model(future, encoder_target, encoder_mask, encoder_delta, t_batch)
          
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

def evaluate(model, scaler, test_loader, diff_scheduler, log_path, vitals, device, num_samples=10): #5

  model.eval()

  total_maes = {}
  total_crpss = {}
  for vital in vitals:
    total_maes[vital] = 0
    total_crpss[vital] = 0

  count = 0

  for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Evaluating: "):
    encoder_target, encoder_mask, encoder_delta, decoder_target, decoder_mask, decoder_delta = batch

    encoder_target = encoder_target.to(device)
    encoder_mask = encoder_mask.to(device)
    encoder_delta = encoder_delta.to(device)
    decoder_target = decoder_target.to(device)
    decoder_mask = decoder_mask.to(device)
    decoder_delta = decoder_delta.to(device)

    B = encoder_target.shape[0]

    # === Generate multiple samples ===
    samples = []
    for _ in range(num_samples):
        # pred = sample_future(model, scaler, encoder_target, encoder_mask, encoder_delta, diff_scheduler, num_steps=diff_scheduler.T)
        pred = sample_future_ddim(model, scaler, encoder_target, encoder_mask, encoder_delta, diff_scheduler, num_steps=50, eta=0.0)
        samples.append(pred)

    samples = torch.stack(samples)  # [S, B, T, D]

    os.makedirs(f'../../outputs/Diffusion/batch_{idx}', exist_ok=True)
    np.save(f'../../outputs/Diffusion/batch_{idx}/encoder_target.npy', encoder_target.detach().cpu().numpy())
    np.save(f'../../outputs/Diffusion/batch_{idx}/decoder_target.npy', decoder_target.detach().cpu().numpy())
    np.save(f'../../outputs/Diffusion/batch_{idx}/samples.npy', samples.detach().cpu().numpy())

    # === Median prediction ===
    median_pred = samples.median(dim=0).values

    # === Metrics ===
    decoder_target = decoder_target.cpu().numpy()
    decoder_target_inv = scaler.inverse(decoder_target)
    mae = compute_mae(median_pred.detach().cpu().numpy(), decoder_target_inv, vitals=vitals)
    crps = compute_crps(samples.detach().cpu().numpy(), decoder_target_inv, vitals=vitals)

    for vital in vitals:
      total_maes[vital] += mae[vital]
      total_crpss[vital] += crps[vital]

    count += 1

  # enocder_target = encoder_target[0].detach().cpu().numpy()
  # enocder_target_inv = scaler.inverse(enocder_target)
  # for i in range(len(vitals)):
  #   vital = vitals[i]
  #   x_past = enocder_target_inv[:, i]
  #   plot_forecast(x_past=x_past, samples=samples[:, 0, :, i], target=decoder_target_inv[0, :, i], save_path=f'{log_path}/figures/{vital}/pred.png')

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
save_path = '../../models/Diffusion/epoch_29'
model_save_path = f'{save_path}/model.pth'
batch_size = 512 #2048
vitals = ['HR', 'RESP', 'SpO2']

log_path = '../../logs/Diffusion'
os.makedirs(f'{log_path}/figures', exist_ok=True)
os.makedirs(f'{log_path}/figures/HR', exist_ok=True)
os.makedirs(f'{log_path}/figures/RESP', exist_ok=True)
os.makedirs(f'{log_path}/figures/SpO2', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. Load and Process Data
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
model = DiffusionForecaster(num_vitals=3, prediction_length=10, embed_dim=256, num_heads=4, num_layers=4) #256, 4, 4

#Load Weights
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
model.to(device)

#3. Train
diff_scheduler = DiffusionScheduler(device, T=400)


#test step
test_metrics = evaluate(model=model, scaler=scaler, test_loader=test_loader, diff_scheduler=diff_scheduler, log_path=log_path, vitals=vitals, device=device)


log_string = 'Results'
for metric in test_metrics.keys():
    for vital in vitals:
        log_string += f' || Test {vital} {metric} - {test_metrics[metric][vital]}'

print(log_string)
