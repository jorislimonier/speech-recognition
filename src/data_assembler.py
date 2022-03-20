import glob

import pandas as pd


class DataAssembler:
    RAW_PATH = "./data/raw/"
    INTERIM_PATH = "./data/interim/"

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

        # rename columns
        df.columns = ["parent_folder", "ground_truth"]

        # extract filename
        df["filename"] = df["ground_truth"].apply(lambda txt: txt.split(" (")[1])
        df["filename"] = df["filename"].apply(lambda fn: fn[:-1])

        # extract text
        df["ground_truth"] = df["ground_truth"].apply(lambda txt: txt.split(" (")[0])

        # remove leading and trailing whitespace
        df = df.apply(lambda x: x.str.strip())

        # write data
        df = df[["parent_folder", "filename", "ground_truth"]]

        # construct full path of wav files
        start_digits = df["parent_folder"].apply(lambda x: x[:2])
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
