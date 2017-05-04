# speechless
Speech recognizer based on [wav2letter architecture](https://arxiv.org/pdf/1609.03193v2.pdf) built with [Keras](https://keras.io/).

Supports CTC loss, KenLM and greedy decoding and transfer learning between different languages. ASG loss is currently not supported.

Training for English with the [1000h LibriSpeech corpus](http://www.openslr.org/12) works out of the box, 
while training for the German language requires downloading data manually.

## Installation

Python 3.4+ and [TensorFlow](https://www.tensorflow.org/install/) are required.

    pip3 install git+git@github.com:JuliusKunze/speechless.git

will install speechless together with minimal requirements.

If you want to use the KenLM decoder, [this modified version](https://github.com/timediv/tensorflow-with-kenlm) of TensorFlow needs to be installed first.

You need to have an audio backend available, for example ffmpeg (run `brew install ffmpeg` on Mac OS).  

## Training

```python
from speechless.configuration import Configuration

Configuration.minimal_english().train_from_beginning()
```
    
will automatically download a small English example corpus (337MB), 
train a net based on it while giving you updated loss and predictions.
If you use a strong consumer-grade GPU, you should observe training predictions become similar to the input after ~12h, e. g.
```
Expected:  "just thrust and parry and victory to the stronger"
Predicted: "jest thcrus and pary and bettor o the stronter"
Errors: 10 letters (20%), 6 words (67%), loss: 37.19.
```

All data (corpus, nets, logs) will be stored in `~/speechless-data`.

This directory can be changed:
```python
from pathlib import Path

from speechless import configuration
from speechless.configuration import Configuration, DataDirectories

configuration.default_data_directories = DataDirectories(Path("/your/data/path"))

Configuration.minimal_english().train_from_beginning()
```

To download and train on the full 1000h LibriSpeech corpus, replace `mininal_english` with `english`.

`main.py` contains various other functions that were executed to train and use models.

## Loading & Testing

By default, all trained models are stored in the `~/speechless-data/nets` directory after every 100 batches of training. 
To load a previously trained model `load_model` there, use e. g. 

```python
from speechless.configuration import Configuration
from speechless.english_corpus import english_frequent_characters

german = Configuration.german()

wav2letter = german.load_model(
    load_name="20170314-134351-adam-small-learning-rate-complete-95",
    load_epoch=1689, allowed_characters_for_loaded_model=english_frequent_characters)

german.test_model_grouped_by_loaded_corpus_name(wav2letter)
```

If the language was originally trained with a different character set (e. g. a corpus of another language),
specifying `allowed_characters_for_loaded_model` still allows you to use that model for training, thereby allowing transfer learning. 

Testing will write to the standard output and a log to `~/speechless-data/test-results` by default.

## Recording

You can record your own audio with a microphone and get a prediction for it:
```python
from speechless.recording import record_plot_and_save
from speechless.configuration import Configuration

label = record_plot_and_save()

wav2letter = Configuration.german().load_model(load_name="some_model", load_epoch=42)
wav2letter.predict(label)
```

Three seconds of silence will automatically end the recording and silence will automatically be truncated.
By default, this will store a `wav`-file and a spectrogram plot into `~/speechless-data/recordings`.

### Input plotting

Plotting labeled audio examples from the corpus like this one [here](https://docs.google.com/presentation/d/1X30IcB-CzCxnGt780ze0qOrbsRtDrxbWrZ_zQ91TOZQ/edit#slide=id.g1b9173e933_0_15) can be done with `LabeledExamplePlotter.save_spectrogram`.