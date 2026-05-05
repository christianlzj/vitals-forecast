from sklearn.preprocessing import StandardScaler

class VitalScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.delta_scaler = StandardScaler()
        self.delta_cols = []

    def fit(self, df, cols=['HR', 'RESP', 'SpO2']):
        self.cols = cols
        self.scaler.fit(df[cols])

        for col in cols:
            self.delta_cols.append(f'{col}_delta')

        self.delta_scaler.fit(df[self.delta_cols])
    
    def transform(self, df):
        df = df.copy()
        df[self.cols] = self.scaler.transform(df[self.cols])

        df[self.delta_cols] = self.delta_scaler.transform(df[self.delta_cols])
        return df
    
    def inverse(self, arr):
        # arr: (..., num_features)
        shape = arr.shape
        arr_flat = arr.reshape(-1, len(self.cols))
        arr_inv = self.scaler.inverse_transform(arr_flat)
        return arr_inv.reshape(shape)

class WaveformScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, df, cols=['II', 'RESP', 'PLETH']):
        self.cols = cols
        self.scaler.fit(df[cols])
    
    def transform(self, df):
        df = df.copy()
        df[self.cols] = self.scaler.transform(df[self.cols])

        return df
    
    def inverse(self, arr):
        # arr: (..., num_features)
        shape = arr.shape
        arr_flat = arr.reshape(-1, len(self.cols))
        arr_inv = self.scaler.inverse_transform(arr_flat)
        return arr_inv.reshape(shape)