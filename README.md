# speechless
Speech recognizer based on wav2letter architecture built with Keras

# Installation

[Install keras](https://keras.io/#installation) if not yet done.

Then run 

    pip3 install -r requirements.txt

Now

    python3 main.py
    
will automatically download the smallest English example corpus (322MB), 
and train a net based on it. Everything (corpus, nets, logs) will be stored in `~/speechless-data`.
You can change this directory by adapting `base_directory` in `main.py`. 

`main.py` currently contains other configurations that were executed to train and use models.