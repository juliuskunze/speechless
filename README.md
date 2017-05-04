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

## Loading

By default, all trained models are stored in the `~/speechless-data/nets` directory. 
You use models from [here](https://drive.google.com/drive/folders/0B0Azt-a50ylyal9JVDJnbXJJd2c?usp=sharing) by downloading them into this folder (keep the subfolder from Google Drive).
To load a such a model use `load_model` or `load_german_model` e. g.

```python
from speechless.configuration import Configuration

german = Configuration.german()

wav2letter = german.load_german_model(load_name="20170420-001258-adam-small-learning-rate-transfer-to-German-freeze-0", load_epoch=2066)
```

If the language was originally trained with a different character set (e. g. a corpus of another language),
specifying the `allowed_characters_for_loaded_model` parameter of `load_model` still allows you to use that model for training, 
thereby allowing transfer learning. 

## Recording

You can record your own audio with a microphone and get a prediction for it:
```python
# ... after loading a model, see above

from speechless.recording import record_plot_and_save

label = record_plot_and_save()

print(wav2letter.predict(label))
```

Three seconds of silence will end the recording and silence will be truncated.
By default, this will generate a `wav`-file and a spectrogram plot in `~/speechless-data/recordings`.


## Testing

Given that you downloaded the German corpus into the corpus directory, you can evaluate the German model on the test set:

```python
german.test_model_grouped_by_loaded_corpus_name(wav2letter)
```

Testing will write to the standard output and a log to `~/speechless-data/test-results` by default.

## Plotting

Plotting labeled audio examples from the corpus like this one [here](https://docs.google.com/presentation/d/1X30IcB-CzCxnGt780ze0qOrbsRtDrxbWrZ_zQ91TOZQ/edit#slide=id.g1b9173e933_0_15) can be done with `LabeledExamplePlotter.save_spectrogram`.