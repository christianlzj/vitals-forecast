import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from tqdm import tqdm
import random


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


def process_waveform_dataset(waveform_conditioning=True, test_mode=False):
    window_vital_dfs = []
    window_waveform_dfs = []

    random_folders = random.sample(os.listdir('../../datasets/waveform_data'), len(os.listdir('../../datasets/waveform_data')) // 100)
    for window_folder in tqdm(random_folders):
        if "window" in window_folder:
            vital_encoder_targets = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_encoder_targets.parquet').iloc[:5]
            vital_encoder_delta_values = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_encoder_delta_values.parquet').iloc[:5]
            vital_encoder_delta_values = vital_encoder_delta_values.add_suffix('_delta')
            vital_df = pd.concat([vital_encoder_targets, vital_encoder_delta_values], axis=1)
            window_vital_dfs.append(vital_df)

            waveform_data = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/waveform_data.parquet')
            waveform_data = waveform_data.interpolate().bfill().ffill()
            # waveform_data = waveform_data.sample(frac=0.1)
            window_waveform_dfs.append(waveform_data)
    
    full_vital_df = pd.concat(window_vital_dfs)
    full_waveform_df = pd.concat(window_waveform_dfs)

    return full_vital_df, full_waveform_df


def process_clinical_dataset(vitals=['HR', 'RESP', 'SpO2'], test_mode=False):
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
        train_ids = np.setdiff1d(df['Record'].unique(), test_ids)
    else:
        try:
            test_ids = np.load('../../datasets/clinical_test_records.npy', allow_pickle=True)
        except:
            train_ids = np.random.choice(df['Record'].unique(), size=int(0.8 * df['Record'].nunique()), replace=False)
            test_ids = np.setdiff1d(df['Record'].unique(), train_ids)
            np.save('../../datasets/clinical_test_records.npy', test_ids)
    
    train_ids = np.setdiff1d(df['Record'].unique(), test_ids)
    train_vital_df = df[df['Record'].isin(train_ids)].sort_values(by=['Record', 'Minute'])
    test_vital_df = df[df['Record'].isin(test_ids)].sort_values(by=['Record', 'Minute'])


    #######################
    #   CLINICAL EVENTS
    #######################

    input_events_df = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_INPUTEVENTS.parquet')
    chart_events_df_00 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0000.parquet')
    chart_events_df_01 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0001.parquet')
    chart_events_df_02 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0002.parquet')
    chart_events_df_03 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0003.parquet')
    chart_events_df_04 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0004.parquet')
    chart_events_df_05 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0005.parquet')
    chart_events_df_06 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0006.parquet')
    chart_events_df_07 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0007.parquet')
    chart_events_df_08 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0008.parquet')
    chart_events_df_09 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0009.parquet')
    # chart_events_df_10 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0010.parquet')
    # chart_events_df_11 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0011.parquet')
    # chart_events_df_12 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0012.parquet')
    # chart_events_df_13 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0013.parquet')
    # chart_events_df_14 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0014.parquet')
    # chart_events_df_15 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0015.parquet')
    # chart_events_df_16 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0016.parquet')
    # chart_events_df_17 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0017.parquet')
    # chart_events_df_18 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0018.parquet')
    # chart_events_df_19 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0019.parquet')
    # chart_events_df_20 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0020.parquet')
    # chart_events_df_21 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0021.parquet')
    # chart_events_df_22 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0022.parquet')
    # chart_events_df_23 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0023.parquet')
    # chart_events_df_24 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0024.parquet')
    # chart_events_df_25 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0025.parquet')
    # chart_events_df_26 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0026.parquet')
    # chart_events_df_27 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0027.parquet')
    # chart_events_df_28 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0028.parquet')
    # chart_events_df_29 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0029.parquet')
    # chart_events_df_30 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0030.parquet')
    # chart_events_df_31 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0031.parquet')
    # chart_events_df_32 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0032.parquet')
    # chart_events_df_33 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0033.parquet')
    # chart_events_df_34 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0034.parquet')
    # chart_events_df_35 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0035.parquet')
    # chart_events_df_36 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0036.parquet')
    # chart_events_df_37 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0037.parquet')
    # chart_events_df_38 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0038.parquet')
    # chart_events_df_39 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0039.parquet')
    # chart_events_df_40 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0040.parquet')
    # chart_events_df_41 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0041.parquet')
    # chart_events_df_42 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0042.parquet')
    # chart_events_df_43 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0043.parquet')
    # chart_events_df_44 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0044.parquet')
    # chart_events_df_45 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0045.parquet')
    # chart_events_df_46 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0046.parquet')
    # chart_events_df_47 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0047.parquet')
    # chart_events_df_48 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0048.parquet')
    # chart_events_df_49 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0049.parquet')
    # chart_events_df_50 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0050.parquet')
    # chart_events_df_51 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0051.parquet')
    # chart_events_df_52 = pd.read_parquet('../../datasets/clinical_data/EMBEDDED_CHARTEVENTS_0052.parquet')

    chart_events_df = pd.concat([
        chart_events_df_00, 
        chart_events_df_01, 
        chart_events_df_02, 
        chart_events_df_03, 
        chart_events_df_04, 
        chart_events_df_05, 
        chart_events_df_06, 
        chart_events_df_07, 
        chart_events_df_08, 
        chart_events_df_09, 
        # chart_events_df_10, 
        # chart_events_df_11, 
        # chart_events_df_12, 
        # chart_events_df_13, 
        # chart_events_df_14, 
        # chart_events_df_15, 
        # chart_events_df_16, 
        # chart_events_df_17, 
        # chart_events_df_18, 
        # chart_events_df_19, 
        # chart_events_df_20,
        # chart_events_df_21,
        # chart_events_df_22,
        # chart_events_df_23,
        # chart_events_df_24,
        # chart_events_df_25,
        # chart_events_df_26,
        # chart_events_df_27,
        # chart_events_df_28,
        # chart_events_df_29,
        # chart_events_df_30,
        # chart_events_df_31,
        # chart_events_df_32,
        # chart_events_df_33,
        # chart_events_df_34,
        # chart_events_df_35,
        # chart_events_df_36,
        # chart_events_df_37,
        # chart_events_df_38,
        # chart_events_df_39,
        # chart_events_df_40,
        # chart_events_df_41,
        # chart_events_df_42,
        # chart_events_df_43,
        # chart_events_df_44,
        # chart_events_df_45,
        # chart_events_df_46,
        # chart_events_df_47,
        # chart_events_df_48,
        # chart_events_df_49,
        # chart_events_df_50,
        # chart_events_df_51,
        # chart_events_df_52
    ])

    chart_events_records_list = chart_events_df["Record"].unique()

    filtered_input_events_df = input_events_df[input_events_df['Record'].isin(chart_events_records_list)]

    full_clinical_df = pd.concat([
        filtered_input_events_df, 
        chart_events_df
    ])

    clinical_df = full_clinical_df[["Record", "Minute", "Embedding_256"]]

    #reset indices
    clinical_df = clinical_df.reset_index(drop=True)

    train_clinical_df = clinical_df[clinical_df['Record'].isin(train_ids)].sort_values(by=['Record', 'Minute'])
    test_clinical_df = clinical_df[clinical_df['Record'].isin(test_ids)].sort_values(by=['Record', 'Minute'])
    

    return train_vital_df, test_vital_df, train_clinical_df, test_clinical_df
    


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


class DiffusionTimeSeriesWaveformDataset(Dataset):
    def __init__(self, vital_scaler, waveform_scaler, include_waveform_data=True, test=False):

        self.include_waveform_data = include_waveform_data
        self.windows = []

        folders = os.listdir('../../datasets/waveform_data')

        split_index = int(len(folders) * 0.8)

        if test:
            folder_list = np.sort(folders)[split_index:]
        else:
            folder_list = np.sort(folders)[:split_index]


        for window_folder in tqdm(folder_list):
            if "window" in window_folder:
                vital_encoder_targets = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_encoder_targets.parquet')
                vital_encoder_mask_values = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_encoder_mask_values.parquet')
                vital_encoder_delta_values = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_encoder_delta_values.parquet')
                vital_encoder_delta_values = vital_encoder_delta_values.add_suffix('_delta')
                vital_encoder_targets_and_deltas = pd.concat([vital_encoder_targets, vital_encoder_delta_values], axis=1)
                vital_encoder_targets_and_deltas = vital_scaler.transform(vital_encoder_targets_and_deltas)
                vital_encoder_targets = vital_encoder_targets_and_deltas[vital_encoder_targets.columns.tolist()]
                vital_encoder_delta_values = vital_encoder_targets_and_deltas[vital_encoder_delta_values.columns.tolist()]

                vital_decoder_targets = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_decoder_targets.parquet')
                vital_decoder_mask_values = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_decoder_mask_values.parquet')
                vital_decoder_delta_values = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/vital_decoder_delta_values.parquet')
                vital_decoder_delta_values = vital_decoder_delta_values.add_suffix('_delta')
                vital_decoder_targets_and_deltas = pd.concat([vital_decoder_targets, vital_decoder_delta_values], axis=1)
                vital_decoder_targets_and_deltas = vital_scaler.transform(vital_decoder_targets_and_deltas)
                vital_decoder_targets = vital_decoder_targets_and_deltas[vital_decoder_targets.columns.tolist()]
                vital_decoder_delta_values = vital_decoder_targets_and_deltas[vital_decoder_delta_values.columns.tolist()]

                window = [vital_encoder_targets.to_numpy(), vital_encoder_mask_values.to_numpy(), vital_encoder_delta_values.to_numpy(), vital_decoder_targets.to_numpy(), vital_decoder_mask_values.to_numpy(), vital_decoder_delta_values.to_numpy()]

                if self.include_waveform_data:
                    waveform_data = pd.read_parquet(f'../../datasets/waveform_data/{window_folder}/waveform_data.parquet')
                    waveform_data = waveform_data.interpolate().bfill().ffill()
                    waveform_data = waveform_scaler.transform(waveform_data)
                    
                    window.append(waveform_data.to_numpy())
                
                self.windows.append(window)
    
        
                    

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if self.include_waveform_data:
            encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values, waveform_values = self.windows[idx]

            return (
                torch.tensor(encoder_target_values, dtype=torch.float32),
                torch.tensor(encoder_target_mask_values, dtype=torch.float32),
                torch.tensor(encoder_target_delta_values, dtype=torch.float32),
                torch.tensor(decoder_target_values, dtype=torch.float32),
                torch.tensor(decoder_target_mask_values, dtype=torch.float32),
                torch.tensor(decoder_target_delta_values, dtype=torch.float32),
                torch.tensor(waveform_values, dtype=torch.float32)
            )
        else:
            encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values = self.windows[idx]

            return (
                torch.tensor(encoder_target_values, dtype=torch.float32),
                torch.tensor(encoder_target_mask_values, dtype=torch.float32),
                torch.tensor(encoder_target_delta_values, dtype=torch.float32),
                torch.tensor(decoder_target_values, dtype=torch.float32),
                torch.tensor(decoder_target_mask_values, dtype=torch.float32),
                torch.tensor(decoder_target_delta_values, dtype=torch.float32)
            )
        


class DiffusionTimeSeriesClinicalConditionedDataset(Dataset):
    def __init__(self, vital_data, clinical_data=None, include_clinical_data=False, encoder_length=60, prediction_length=10, stride= 5, group_ids='Record', time_idx='Minute', targets=['HR', 'RESP', 'SpO2'], 
    target_masks=["HR_missing", "RESP_missing", "SpO2_missing"], target_deltas=["HR_delta", "RESP_delta", "SpO2_delta"]):
        
        self.vital_data = vital_data
        self.clinical_data = clinical_data
        self.include_clinical_data = include_clinical_data
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.group_ids = group_ids
        self.time_idx = time_idx
        self.targets = targets
        self.target_masks = target_masks
        self.target_deltas = target_deltas

        self.windows = []

        clincal_data_records_list = self.clinical_data[self.group_ids].unique()

        for _, group in self.vital_data.groupby(self.group_ids):
            record_name = group[self.group_ids].iloc[0]
            if record_name not in clincal_data_records_list:
                continue
            target_values = group[self.targets].values
            target_mask_values = group[self.target_masks].values
            target_delta_values = group[self.target_deltas].values

            if self.include_clinical_data:
                group_clinical_data = self.clinical_data[self.clinical_data[self.group_ids] == record_name]

                group_clinical_embs = np.stack(group_clinical_data['Embedding_256'].values)
                group_clinical_times = group_clinical_data[self.time_idx].values

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

                if self.include_clinical_data:
                    mask = group_clinical_times < i + self.encoder_length
                    if not np.any(mask):
                        clinical_embeddings = np.zeros(256, dtype=np.float32)
                    else:
                        embs = group_clinical_embs[mask]
                        deltas = (i + self.encoder_length - group_clinical_times[mask])

                        weights = np.log(deltas)
                        clinical_embeddings = (weights[:, None] * embs).sum(axis=0) / np.sum(mask)


                if self.include_clinical_data:
                    self.windows.append((encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values, clinical_embeddings))
                else:
                    self.windows.append((encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values))


    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if self.include_clinical_data:
            encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values, clinical_embeddings = self.windows[idx]
            
            return (
                torch.tensor(encoder_target_values, dtype=torch.float32),
                torch.tensor(encoder_target_mask_values, dtype=torch.float32),
                torch.tensor(encoder_target_delta_values, dtype=torch.float32),
                torch.tensor(decoder_target_values, dtype=torch.float32),
                torch.tensor(decoder_target_mask_values, dtype=torch.float32),
                torch.tensor(decoder_target_delta_values, dtype=torch.float32),
                torch.tensor(clinical_embeddings, dtype=torch.float32).unsqueeze(0)
            )
        else:
            encoder_target_values, encoder_target_mask_values, encoder_target_delta_values, decoder_target_values, decoder_target_mask_values, decoder_target_delta_values = self.windows[idx]

            return (
                torch.tensor(encoder_target_values, dtype=torch.float32),
                torch.tensor(encoder_target_mask_values, dtype=torch.float32),
                torch.tensor(encoder_target_delta_values, dtype=torch.float32),
                torch.tensor(decoder_target_values, dtype=torch.float32),
                torch.tensor(decoder_target_mask_values, dtype=torch.float32),
                torch.tensor(decoder_target_delta_values, dtype=torch.float32),
            )






