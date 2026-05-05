import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import os


def process_dataset(vital):

    dfs = []
    for parquet_file_name in os.listdir(f'../../datasets/pretraining/{vital}'):
        if 'parquet' in parquet_file_name:
            df_path = f'../../datasets/pretraining/{vital}/{parquet_file_name}'

            df = pd.read_parquet(df_path)
            dfs.append(df)

    full_df = pd.concat(dfs)

    filter_cols = [vital] + ["Minute", "Record"]

    df = full_df[filter_cols]

    #drop first 30 minutes from each record
    df = df[df["Minute"] >= 30].copy()

    #Set 0's to NaN so they are filtered out
    df[vital] = df[vital].replace(0, np.nan)

    #Handle Missing Values
    df[f"{vital}_missing"] = df[vital].isna().astype(int) #Add mask column
    # df[f"{vital}_delta"] = df.groupby("Record")[f"{vital}_missing"].cumsum() # add delta column
    df[f"{vital}_delta"] = (
        df.groupby("Record")
        .apply(lambda g: (
            g["Minute"] - g["Minute"].where(g[vital].notna()).ffill()
        ).fillna(0))
        .reset_index(level=0, drop=True)
        )

    df[vital] = df.groupby("Record")[vital].ffill() #forward fill missing values within each record
    df[vital] = df[vital].fillna(df[vital].mean()) #fill any remaining missing values with column mean

    #reset indices
    df = df.reset_index(drop=True)

    train_df = df.sort_values(by=['Record', 'Minute'])
    print(f'Train df: {train_df.shape}')

    return train_df


class VitalEncoderPretrainingDataset(Dataset):
    def __init__(self, data, vital, encoder_length=60, stride= 5, group_ids='Record', time_idx='Minute'):
        
        self.data = data
        self.vital = vital
        self.vital_mask = f'{self.vital}_missing'
        self.vital_delta = f'{self.vital}_delta'
        self.encoder_length = encoder_length
        self.stride = stride
        self.group_ids = group_ids
        self.time_idx = time_idx

        self.windows = []

        for _, group in self.data.groupby(self.group_ids):
            target_values = group[self.vital].values
            target_mask_values = group[self.vital_mask].values
            target_delta_values = group[self.vital_delta].values

            for i in range(0, len(target_values) - self.encoder_length, self.stride):
                window_values = target_values[i:i+self.encoder_length]
                window_mask_values = target_mask_values[i:i+self.encoder_length]
                window_delta_values = target_delta_values[i:i+self.encoder_length]

                if np.mean(window_mask_values) > 0.5:
                    continue

                self.windows.append((window_values, window_mask_values, window_delta_values))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_values, window_mask_values, window_delta_values = self.windows[idx]

        return (
            torch.tensor(window_values, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(window_mask_values, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(window_delta_values, dtype=torch.float32).unsqueeze(-1),
        )
        

