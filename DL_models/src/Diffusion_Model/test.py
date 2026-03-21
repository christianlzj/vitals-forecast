import pandas as pd
import numpy as np

df00 = pd.read_parquet('../../datasets/mimic_vitals_p00.parquet')

record_ids = df00['Record'].unique()
test_ids = record_ids[:10]

test_df = df00[df00['Record'].isin(test_ids)]
test_df.to_parquet("../../datasets/test/test.parquet")

test_test_record_ids = record_ids[8:10]
np.save("../../datasets/test/test_ids.npy", np.array(test_test_record_ids))
