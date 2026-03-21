import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


def process_dataset(vitals=['HR', 'RESP', 'SpO2'], test_mode=False):
    if test_mode:
        full_df = pd.read_parquet('../../datasets/test/test.parquet')
    else:
        df_00 = pd.read_parquet('../../datasets/mimic_vitals_p00.parquet')
        df_01 = pd.read_parquet('../../datasets/mimic_vitals_p01.parquet')
        df_02 = pd.read_parquet('../../datasets/mimic_vitals_p02.parquet')
        df_03 = pd.read_parquet('../../datasets/mimic_vitals_p03.parquet')
        df_04 = pd.read_parquet('../../datasets/mimic_vitals_p04.parquet')
        df_05 = pd.read_parquet('../../datasets/mimic_vitals_p05.parquet')
        df_06 = pd.read_parquet('../../datasets/mimic_vitals_p06.parquet')
        df_07 = pd.read_parquet('../../datasets/mimic_vitals_p07.parquet')
        df_08 = pd.read_parquet('../../datasets/mimic_vitals_p08.parquet')
        df_09 = pd.read_parquet('../../datasets/mimic_vitals_p09.parquet')

        full_df = pd.concat([
            df_00, 
            df_01, 
            df_02, 
            df_03, 
            df_04, 
            df_05, 
            df_06, 
            df_07, 
            df_08, 
            df_09
        ])

    filter_cols = vitals + ["Minute", "Record"]

    df = full_df[filter_cols]

    #drop first 30 minutes from each record
    df = df[df["Minute"] >= 30].copy()

    #Set 0's to NaN so they are filtered out
    df[vitals] = df[vitals].replace(0, np.nan)

    #Handle Missing Values
    for col in vitals:
        df[f"{col}_missing"] = df[col].isna().astype(int) #Add mask column
        # df[f"{col}_delta"] = df.groupby("Record")[f"{col}_missing"].cumsum() # add delta column
        df[f"{col}_delta"] = (
            df.groupby("Record")
            .apply(lambda g: (
                g["Minute"] - g["Minute"].where(g[col].notna()).ffill()
            ).fillna(0))
            .reset_index(level=0, drop=True)
        )

    df[vitals] = df.groupby("Record")[vitals].ffill() #forward fill missing values within each record
    df[vitals] = df[vitals].fillna(df[vitals].mean()) #fill any remaining missing values with column mean

    #reset indices
    df = df.reset_index(drop=True)

    #create new splits if create_new_splits=True
    # splits_path = '../../datasets/splits/'
    # if create_new_splits:
    #     train_ids = np.random.choice(df['Record'].unique(), size=int(0.8 * df['Record'].nunique()), replace=False)
    #     test_val_ids = np.setdiff1d(df['Record'].unique(), train_ids)
    #     test_ids = np.random.choice(test_val_ids, size=int(0.5 * len(test_val_ids)), replace=False)
    #     val_ids = np.setdiff1d(test_val_ids, test_ids)
    #     np.save(f'{splits_path}/train_ids.npy', train_ids)
    #     np.save(f'{splits_path}/test_ids.npy', test_ids)
    #     np.save(f'{splits_path}/val_ids.npy', val_ids)
    #     print(f'Train IDs: {len(train_ids)}')
    #     print(f'Test IDs: {len(test_ids)}')
    #     print(f'Validation IDs: {len(val_ids)}')

    #load split ids, create split dfs
    if test_mode:
        test_ids = np.load('../../datasets/test/test_ids.npy', allow_pickle=True)
    else:
        test_ids = np.load('../../datasets/test_records.npy', allow_pickle=True)
    train_ids = np.setdiff1d(df['Record'].unique(), test_ids)
    train_df = df[df['Record'].isin(train_ids)].sort_values(by=['Record', 'Minute'])
    test_df = df[df['Record'].isin(test_ids)].sort_values(by=['Record', 'Minute'])
    # print(f'Train df: {train_df.shape}')
    # print(f'Test df: {test_df.shape}')

    return train_df, test_df


class DiffusionTimeSeriesDataset(Dataset):
    def __init__(self, data, encoder_length=60, prediction_length=10, stride= 5, group_ids='Record', time_idx='Minute', targets=['HR', 'RESP', 'SpO2'], 
    target_masks=["HR_missing", "RESP_missing", "SpO2_missing"], target_deltas=["HR_delta", "RESP_delta", "SpO2_delta"]):
        
        self.data = data
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.group_ids = group_ids
        self.time_idx = time_idx
        self.targets = targets
        self.target_masks = target_masks
        self.target_deltas = target_deltas

        self.windows = []

        for _, group in self.data.groupby(self.group_ids):
            target_values = group[self.targets].values
            target_mask_values = group[self.target_masks].values
            target_delta_values = group[self.target_deltas].values

            for i in range(0, len(target_values) - self.encoder_length - self.prediction_length, self.stride):
                encoder_target_values = target_values[i:i+self.encoder_length]
                encoder_target_mask_values = target_mask_values[i:i+self.encoder_length]
                encoder_target_delta_values = target_delta_values[i:i+self.encoder_length]

                decoder_target_values = target_values[i+self.encoder_length:i+self.encoder_length+self.prediction_length]
                decoder_target_mask_values = target_mask_values[i+self.encoder_length:i+self.encoder_length+self.prediction_length]
                decoder_target_delta_values = target_delta_values[i+self.encoder_length:i+self.encoder_length+self.prediction_length]

                if np.mean(encoder_target_mask_values) > 0.5:
                    continue
                if np.mean(decoder_target_mask_values) > 0.5:
                    continue

                self.windows.append((encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values = self.windows[idx]

        return (
            torch.tensor(encoder_target_values, dtype=torch.float32),
            torch.tensor(encoder_target_mask_values, dtype=torch.float32),
            torch.tensor(encoder_target_delta_values, dtype=torch.float32),
            torch.tensor(decoder_target_values, dtype=torch.float32),
            torch.tensor(decoder_target_mask_values, dtype=torch.float32),
            torch.tensor(decoder_target_delta_values, dtype=torch.float32)
        )
        






