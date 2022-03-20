import pandas as pd

from data_assembler import DataAssembler
from transcribe import Transcribe

BASE_FILES = ["train-ws96-i", "train-ws97-i"]

# Initialize DataAssembler class
da = DataAssembler()

# Get ground truth from each base file
df = pd.concat(
    objs=[da.extract_labels(base_file) for base_file in BASE_FILES],
    ignore_index=True,
)

# Initialize Trancribe class
trsc = Transcribe()

# Make reduced dataset for testing
n_reduced = 2  # number of rows
df_red = df.iloc[:n_reduced, :]  # work on reduced dataframe
print(
    trsc.predict(
        df=df_red,
        benchmark=True,
        # save_file=True,
    )
)
