import os
import wave

import deepspeech
import librosa
import numpy as np
import soundfile as sf

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
        Read a wav file and rewrite it to a temporary wav file in order to avoid the RIFF id error (i.e. wav files previously couldn't be read)
        """
        x, _ = librosa.load(path, sr=16000)
        tmp = "data/interim/tmp.wav"
        sf.write(tmp, x, 16000)
        return tmp

    def batch_transcribe(self, path):
        """
        Use the batch api of DeepSpeech to perform Speech-to-text (STT).
        """
        tmp = self.rewrite_wav(path)

        # read wav file
        w = wave.open(tmp, "r")
        frames = w.getnframes()
        buffer = w.readframes(frames)
        data16 = np.frombuffer(buffer, dtype=np.int16)

        # perform stt
        text = self.model.stt(data16)
        return text

    def stream_transcribe(self, path):
        """
        Use the stream api of DeepSpeech to perform Speech-to-text (STT).
        May get deprecated soon.
        """
        tmp = self.rewrite_wav(path)
        w = wave.open(tmp, "r")
        frames = w.getnframes()
        buffer = w.readframes(frames)

        context = self.model.createStream()

        buffer_len = len(buffer)
        offset = 0
        batch_size = 16384
        text = ""
        while offset < buffer_len:
            end_offset = offset + batch_size
            chunk = buffer[offset:end_offset]
            data16 = np.frombuffer(chunk, dtype=np.int16)
            context.feedAudioContent(data16)
            text = context.intermediateDecode()
            print(text)
            offset = end_offset


trsc = Transcribe()
BASE_FILES = ["train-ws96-i", "train-ws97-i"]

base_file = BASE_FILES[1]
da = DataAssembler()

df = da.extract_labels(base_file)


wav_file = df.loc[0, "path"]
df_red = df.iloc[:5, :] # work on reduced dataframe
df_red["pred"] = df_red["path"].apply(trsc.batch_transcribe)
print(df_red)


# trsc.batch_transcribe(path)
# trsc.stream_transcribe(path)
