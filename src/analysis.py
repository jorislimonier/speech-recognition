import re
from itertools import chain

import jiwer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from jiwer.measures import _preprocess
from sklearn.linear_model import LinearRegression


class Analysis:
    def __init__(self, results_file) -> None:
        self.df_res = pd.read_csv(results_file)

    @staticmethod
    def clean_ground_truth(txt):
        """apply successive cleanings to ground truth"""
        txt = txt.replace("H#", "")
        txt = txt.replace("h#", "")
        txt = txt.replace("_#", "")
        txt = txt.replace("_!", "")
        txt = txt.replace("_?", "")
        txt = txt.replace('"', "")
        txt = txt.replace("_", " ")
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

        # transformations for metrics computation
        transformation = jiwer.Compose(
            [
                jiwer.ToUpperCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveKaldiNonWords(),
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
            ]
        )

        # clean
        df["clean_ground_truth"] = df["ground_truth"].apply(self.clean_ground_truth)

        # iterate over rows and set value for each metric
        for row_nb in range(len(df)):
            clean_ground_truth = df.loc[row_nb, "clean_ground_truth"]
            pred = df.loc[row_nb, "pred"]

            # skip if isna
            if pd.isna(pred):
                continue

            # compute dictionary with all metrics
            measures = jiwer.compute_measures(
                truth=clean_ground_truth,
                hypothesis=pred,
                truth_transform=transformation,
                hypothesis_transform=transformation,
            )

            # set value for each metric
            for measure_name, measure_val in measures.items():
                df.loc[row_nb, measure_name] = measure_val

        return df[["clean_ground_truth"] + list(measures.keys())]

    def count_keywords(self):
        """
        Count the number of keywords in the gorund truth column. \\
        Keywords are between square brackets.
        """
        df = self.df_res.copy()

        get_n_kw = lambda txt: len(re.findall(pattern="\[.*?\]", string=txt))

        return pd.DataFrame({"n_keywords": df["ground_truth"].apply(get_n_kw)})

    def count_words(self):
        """
        Count the number of words in the gorund truth column.
        """
        df = self.df_res.copy()

        get_n_words = lambda txt: len(txt.split(" "))

        return pd.DataFrame({"n_words": df["ground_truth"].apply(get_n_words)})

    def transcr_time_per_word(self, df_res, savefig=False):
        """
        Plot the transcription time as a function of the number of words in the ground truth. \\
        Then, fit a regression line.
        """
        lr = LinearRegression()
        lr.fit(df_res[["n_words"]], df_res["time"])
        slope = lr.coef_[0]
        intercept = lr.intercept_

        fig = px.scatter(
            df_res,
            x="n_words",
            y="time",
            opacity=0.7,
            trendline="ols",
            trendline_color_override="red",
        )
        fig.update_layout(
            title=f"Slope: {round(slope, 2)}, Intercept: {round(intercept, 2)}",
            template="plotly_white",
            font_family="Lato",
            font_color="Black",
            font_size=14,
            xaxis_title="Number of words in ground truth",
            yaxis_title="Transcription time (s)",
        )

        if savefig:
            fig.write_image("transcription_time.png", width=1200, height=800, scale=2)
        return fig

    def corr_heatmap(
        self,
        df_res,
        corr_cols=["time", "n_keywords", "wer"],
        savefig=False
    ):
        """
        Plot the correlation heatmap over the chosen `corr_cols`.
        """
        df_corr = df_res[corr_cols].corr()

        fig = sns.heatmap(
            df_corr,
            annot=True,
            cmap="coolwarm",
        )

        if savefig:
            fig.get_figure().savefig("data/processed/corr_mat.png")
        return fig

    def mean_wer_minus_worst(self, df_res):
        """
        Plot the WER (word error rate) if the worst samples were removed. \\
        The variable is the number of words that would be removed.
        """
        px.line(
            [
                df_res.sort_values("wer", ascending=False).iloc[n_rem:]["wer"].mean()
                for n_rem in range(len(df_res))
            ]
        )
