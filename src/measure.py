import re

import jiwer
import pandas as pd


class Measure:
    def __init__(self, results_file) -> None:
        self.df_res = pd.read_csv(results_file)

    @staticmethod
    def clean_ground_truth(txt):
        """apply successive cleanings to ground truth"""
        txt = txt.replace("H#", "")
        txt = re.sub("\[.*?\]", "", txt)  # remove words between square brackets
        txt = re.sub("  ", " ", txt)  # make multiple whitespaces to single
        txt = txt.strip()

        return txt

    def compute_measures(self):
        """
        Computes wer, mer, wil, wip, hits, substitutions, deletions, insertions
        on `self.df_res` and adds a column for each metric.
        """
        df = self.df_res.copy()
        # drop na
        df = df.dropna().reset_index(drop=True)

        # clean
        df["ground_truth"] = df["ground_truth"].apply(self.clean_ground_truth)

        # transformations for metrics computation
        transformation = jiwer.Compose(
            [
                jiwer.ToUpperCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
            ]
        )

        # iterate over rows and set value for each metric
        for row_nb in range(len(df)):
            ground_truth = df.loc[row_nb, "ground_truth"]
            pred = df.loc[row_nb, "pred"]

            # compute dictionary with all metrics
            measures = jiwer.compute_measures(
                truth=ground_truth,
                hypothesis=pred,
                truth_transform=transformation,
                hypothesis_transform=transformation,
            )

            # set value for each metric
            for measure_name, measure_val in measures.items():
                df.loc[row_nb, measure_name] = measure_val

        return df
