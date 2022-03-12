import wave

import deepspeech
import numpy as np

import librosa
import soundfile as sf


model_file_path = "models/deepspeech-0.9.3-models.pbmm"
beam_width = 500
model = deepspeech.Model(model_file_path)

filename = "data/raw/train-ws96-i/wav/20/2073B/sw2073B-ws96-i-0025.wav"
# filename = "data/external/audio/2830-3980-0043.wav"

def rewrite_wav():
    x, _ = librosa.load(filename, sr=16000)
    tmp = "data/interim/tmp.wav"
    sf.write(tmp, x, 16000)
    return tmp

def batch_transcribe(filepath):
    w = wave.open(filepath, "r")
    rate = w.getframerate()
    frames = w.getnframes()
    buffer = w.readframes(frames)
    # print(rate)
    # print(model.sampleRate())


    data16 = np.frombuffer(buffer, dtype=np.int16)

    text = model.stt(data16)
    print(text)
batch_transcribe()
def stream_transcribe():
    # stream api
    tmp = rewrite_wav()
    w = wave.open(tmp, "r")
    frames = w.getnframes()
    buffer = w.readframes(frames)

    context = model.createStream()

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
