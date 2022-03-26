# Personal notes

- Focus: Speech to text convertion using DeepSpeech model
- Qualitative & Quantitative analysis
- Evaluate the performance of DeepSpeech on Swithcboard data
  - What goes right?
  - What goes wrong?
- Can we reproduce research-level metrics?
- Evaluate the results using Word Rate Error

## Docs

- [Getting started tutorial](https://www.slanglabs.in/blog/how-to-build-python-transcriber-using-mozilla-deepspeech)
- [Official docs](https://deepspeech.readthedocs.io/en/r0.9/)

## TO DO

Analyse the results and observe what affected to the wrong predictions. Some ideas:

- The length of the sentences and speech.
- The length of the words (correctly predicted vs wrongly).
- Analyzing structure of the wrongly predicted words. For example: Words that start with W are predicted wrongly in general. We have to make an analyze for this
- Cem:
  - Deadline for writing the report is Saturday 23:59
    - Report
        - Explain what deepspeech does
        - Mention why we remove 2 samples (ground truth is wrong)
        - Explain that we don't know what exact dataset they use in the paper (we have no label in our data about easy or hard examples)
  - Add:
    - Explanation about paper:
      - Dataset and network architecture
      - Couple sentences per section
    - Table 3 from paper
    - Graph Joris made
    - Add "questions answered" part
    - Explain RIFF error and how we fixed it
    - Explain WER (denominator is total number of words in ground truth)
- Joris:
    - Compute seconds of transcription per number of words
    - Add hardware specs: Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz
    - Check report on sunday


# Questions answered

### How complicated is it to deploy a working pre-trained speech recognition engine?
Very easy: Install model and library. Only issue is that we had to rewrite WAV file (but this wouldn't occur on deployement of the app).

```bash
# Create and activate a virtualenv
virtualenv -p python3 $HOME/tmp/deepspeech-venv/
source $HOME/tmp/deepspeech-venv/bin/activate

# Install DeepSpeech
pip3 install deepspeech

# Download pre-trained English model files
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# Download example audio files
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/audio-0.9.3.tar.gz
tar xvf audio-0.9.3.tar.gz

# Transcribe an audio file
deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio audio/2830-3980-0043.wav
```

### Is the engine able to approach near real-time performance? (lag less than 10 seconds)
Yes (add seconds of transcription per number of words).

### What is the hardware chosen?
`Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz`

### What is the CPU and memory cost of this recognition on the hardware they chose to use?
Can probably be inferred from hardware and transciption time.

### How good is the recognition?

#### Qualitatively, can you illustrate performance with well-chosen perfect recognitions and illustrate typical mistakes?
Further analysis? *e.g.* Words/Letters that are mostly missclassified

#### Quantitatively, what is the Word Error Rate? How does it compare with state-of-the-art engines on the same dataset?
Metric: word error rate
```python
WER = float(S + D + I) / float(H + S + D)
```
Based on substitutions, deletions, insertions and hits (explain each term)


## Presentation

Maybe add [this](https://serpwatch.io/blog/voice-search-statistics/) for intro? (some statistics about voice recognition)

### Part 1. DeepSpeech paper
- Quick explanation of the model (presented by Cem)
  - architecture
  - ideas behind the model
  - what's innovative in this model?

### Part 2. Our solution
- Presentation of the way we solved the problem (presented by Joris)
  - [Code architecture](https://github.com/jorislimonier/speech-recognition/blob/main/report/images/speech-recognition.drawio.png) (one slide per box)
  - RIFF error (explanation + solution)

### Part 3. Our results
- Quantitative results (WER, ...etc) (Cem)
- Qualitative results (Type of words that go right/wrong, ...etc) (Joris)

### Conclusion & Opinion
