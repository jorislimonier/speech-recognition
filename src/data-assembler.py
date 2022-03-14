import os
import glob
import pandas as pd


class DataAssembler:
    RAW_PATH = "./data/raw/"
    INTERIM_PATH = "./data/interim/"

    def list_wav_files(self):
        """
        List all ".wav" files in a subdirectory of the data/raw directory.
        """
        path_pattern = f"{self.RAW_PATH}train-ws96-i/wav/**/**/*.wav"
        wav_files = glob.glob(path_pattern)
        wav_files.sort()
        return wav_files

    @staticmethod
    def extract_labels(trans_file):
        """
        Read data from `trans_file` and return a dataframe with wav file location and its label.
        """
        # read data
        with open(trans_file, "r") as f:
            full_txt = "".join(f.readlines())

        # prep data
        trans_text = full_txt.split("@")[7]
        trans_text = trans_text.split("\n")  # split lines
        trans_text = [s.split("\t") for s in trans_text]  # split columns
        df = pd.DataFrame(trans_text)

        # remove columns with unknown data and remove last row
        df = df.drop(columns=[1, 2], index=[len(df) - 1])

        df.columns = ["parent_folder", "text"]

        # extract filename
        df["filename"] = df["text"].apply(lambda txt: txt.split(" (")[1])
        df["filename"] = df["filename"].apply(lambda fn: fn[:-1])

        # extract text
        df["text"] = df["text"].apply(lambda txt: txt.split(" (")[0])

        # write data
        # df.to_csv("data/interim/labels.csv")
        return df[["parent_folder", "filename", "text"]]


assembler = DataAssembler()
trans_file = "data/raw/train-ws96-i/trans/train-ws96-i-trans.text,v"

# get sorted list of wav files
wav_files = assembler.list_wav_files()
print(wav_files)
print(assembler.extract_labels(trans_file))
