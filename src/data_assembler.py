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

    def extract_labels(self, base_file):
        """
        Read data from `trans_file` and return a dataframe with wav file location and its label.
        """
        trans_file = f"data/raw/{base_file}/trans/{base_file}-trans.text,v"

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

        df.columns = ["parent_folder", "ground_truth"]

        # extract filename
        df["filename"] = df["ground_truth"].apply(lambda txt: txt.split(" (")[1])
        df["filename"] = df["filename"].apply(lambda fn: fn[:-1])

        # extract text
        df["ground_truth"] = df["ground_truth"].apply(lambda txt: txt.split(" (")[0])

        # remove leading and trailing whitespace
        df = df.apply(lambda x: x.str.strip())
        # write data
        # df.to_csv("data/interim/labels.csv")
        df = df[["parent_folder", "filename", "ground_truth"]]
        # main_dir = "train-ws96-i/"

        # construct full path of wav files
        start_digits = df["parent_folder"].apply(lambda x: x[:2])
        print(start_digits)
        df["path"] = (
            f"""{self.RAW_PATH}{base_file}/wav/"""
            + start_digits
            + "/"
            + df["parent_folder"]
            + "/"
            + df["filename"]
            + ".wav"
        )
        return df[["path", "ground_truth"]]
