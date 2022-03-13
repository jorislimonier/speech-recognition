import os
import glob
import pandas as pd


class DataAssembler:
    RAW_PATH = "./data/raw/"
    INTERIM_PATH = "./data/interim/"

    @staticmethod
    def list_wav_files():
        path_pattern = "data/raw/train-ws96-i/wav/**/**/*.wav"
        return glob.glob(path_pattern)


assembler = DataAssembler()

# get sorted list of wav files
wav_files = assembler.list_wav_files()
wav_files.sort()
# get wrd counterparts
wrd_files = [f.replace("wav", "wrd") for f in wav_files]


wrd_file = wrd_files[3]


# with open(wrd_file, "r") as f:
#     txt = "".join(f.readlines())
#     # print(txt)
#     print(txt.split("\n#\n")[1])


trans_file = "data/raw/train-ws96-i/trans/train-ws96-i-trans.text,v"

# read data
with open(trans_file, "r") as f:
    full_txt = "".join(f.readlines())

# prep data
trans_text = full_txt.split("@")[7]
trans_text = trans_text.split("\n")  # split lines
trans_text = [s.split("\t") for s in trans_text]  # split columns
df = pd.DataFrame(trans_text)

# remove columns with unknown data and last row
df = df.drop(columns=[1, 2], index=[len(df)-1])

# write data
df.to_csv("data/interim/labels.csv")
print(df)
