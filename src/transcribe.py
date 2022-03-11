import wave

import deepspeech
import numpy as np


model_file_path = "models/deepspeech-0.9.3-models.pbmm"
beam_width = 500
model = deepspeech.Model(model_file_path)

filename = "data/external/audio/2830-3980-0043.wav"
w = wave.open(filename, "r")
rate = w.getframerate()
frames = w.getnframes()
buffer = w.readframes(frames)
# print(rate)
# print(model.sampleRate())


data16 = np.frombuffer(buffer, dtype=np.int16)

# text = model.stt(data16)
# print(text)
# print(text)

# stream api
context = model.createStream()

buffer_len = len(buffer)
offset = 0
batch_size = 16384
text = ''
while offset < buffer_len:
    end_offset = offset + batch_size
    chunk = buffer[offset:end_offset]
    data16 = np.frombuffer(chunk, dtype=np.int16)
    context.feedAudioContent(data16)
    text = context.intermediateDecode()
    print(text)
    offset = end_offset