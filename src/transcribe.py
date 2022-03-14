import wave

import deepspeech
import numpy as np

import librosa
import soundfile as sf


class Transcribe:
    def __init__(
        self,
        model_filepath="models/deepspeech-0.9.3-models.pbmm",
    ):
        self.model_filepath = model_filepath
        self.model = deepspeech.Model(self.model_filepath)

    def rewrite_wav(self, filepath):
        """
        Read a wav file and rewrite it to a temporary wav file in order to avoid the RIFF id error (i.e. wav files previously couldn't be read)
        """
        x, _ = librosa.load(filepath, sr=16000)
        tmp = "data/interim/tmp.wav"
        sf.write(tmp, x, 16000)
        return tmp

    def batch_transcribe(self, filepath):
        """
        Use the batch api of DeepSpeech to perform Speech-to-text (STT).
        """
        tmp = self.rewrite_wav(filepath)

        w = wave.open(tmp, "r")
        frames = w.getnframes()
        buffer = w.readframes(frames)
        data16 = np.frombuffer(buffer, dtype=np.int16)

        text = self.model.stt(data16)
        print(text)

    def stream_transcribe(self, filepath):
        """
        Use the stream api of DeepSpeech to perform Speech-to-text (STT).
        May get deprecated soon.
        """
        tmp = self.rewrite_wav(filepath)
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
filepath = "data/raw/train-ws96-i/wav/20/2073B/sw2073B-ws96-i-0025.wav"
# filename = "data/external/audio/2830-3980-0043.wav"


trsc.batch_transcribe(filepath)
# trsc.stream_transcribe(filepath)
