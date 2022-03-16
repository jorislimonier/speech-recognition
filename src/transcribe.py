import os
import wave
from time import time

import deepspeech
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from data_assembler import DataAssembler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # remove message "This TensorFlow binary..."


class Transcribe:
    BASE_FILES = ["train-ws96-i", "train-ws97-i"]

    def __init__(
        self,
        model_path="models/deepspeech-0.9.3-models.pbmm",
    ):
        self.model_path = model_path
        self.model = deepspeech.Model(self.model_path)

    def rewrite_wav(self, path):
        """
        Read a wav file and rewrite it to a temporary wav file in order to avoid the RIFF id error
        (i.e. wav files previously couldn't be read)
        """
        x, _ = librosa.load(path, sr=16000)
        tmp = "data/interim/tmp.wav"
        sf.write(tmp, x, 16000)
        return tmp

    def batch_transcribe(self, path, benchmark=False):
        """
        Use the batch api of DeepSpeech to perform Speech-to-text (STT).
        """
        tmp = self.rewrite_wav(path)

        if benchmark:
            start_time = time()

        # read wav file
        w = wave.open(tmp, "r")
        frames = w.getnframes()
        buffer = w.readframes(frames)
        data16 = np.frombuffer(buffer, dtype=np.int16)

        # perform stt
        text = self.model.stt(data16)

        if benchmark:
            end_time = time()
            transcription_time = end_time - start_time
            return text, transcription_time
        return text

    def predict(self, df, benchmark=False, save_file=False):
        """
        Predicts a column of the df, adds the prediction to a "pred" column and returns the df.

        If `benchmark=True`, also times the transcription and adds it to a "time" column.
        """
        df = df.copy()  # prevent SettingWithCopyWarning

        n_samples = len(df)
        df["pred"] = np.nan

        if benchmark:
            df["time"] = np.nan

        for row_nb in tqdm(range(n_samples)):
            wav_file = df.loc[row_nb, "path"]

            try:
                if benchmark:
                    pred, time = self.batch_transcribe(
                        path=wav_file,
                        benchmark=benchmark,
                    )
                    df.loc[row_nb, "pred"] = pred
                    df.loc[row_nb, "time"] = time

                else:
                    df.loc[row_nb, "pred"] = self.batch_transcribe(
                        path=wav_file,
                        benchmark=benchmark,
                    )
            except Exception as e:
                print(f"EXCEPTION: {e}")

        if save_file:
            str_n_samples = str(n_samples).zfill(4)
            df.to_csv(
                f"data/processed/results_{str_n_samples}_samples.csv", index=False
            )
        return df


BASE_FILES = ["train-ws96-i", "train-ws97-i"]

base_file = BASE_FILES[1]
da = DataAssembler()

df = pd.concat(
    objs=[da.extract_labels(base_file) for base_file in BASE_FILES],
    ignore_index=True,
)
print(df)
trsc = Transcribe()

df_red = df.iloc[:, :]  # work on reduced dataframe
print(trsc.predict(df_red, benchmark=True, save_file=True))
