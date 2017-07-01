# language_models

(in progress) Individual summer project at Imperial College London

The (initial) goal of the project is to create a personalised language model, i.e. a language model which can generate dialogue that resembles a specific speaker. Benchmark language models are typically trained on a very large text corpus (Wikipedia, for example). A challenge of this project is to build a robust, personalised language model despite having a relatively small amount of text specific to the individual speaker.

### Data
Television show scripts, e.g. Seinfeld, which are long running and have consistent characters. As stated above, this is still a very small text corpus relative to that used in most language models. This is supplemented with:  
-word2vec, GLoVe word vectors  
-pretraining on larger (related) text corpus, e.g. other characters in the TV show, movie scripts, etc.  
  
All data is stored in the 'data/' folder. Seinfeld scripts are stored in an SQL database. The data is preprocessed and train/test text files were created using preprocessing.py.

### Model
Word language model using recurrent neural networks (LSTM, GRU, basic RNN). 

Two main model types:
1. specific to an individual TV character, so each character has his/her own separate model  
2. shared model among multiple TV characters. In this case, the word embeddings and RNN parameters are shared between characters but the projection layer parameters are specific (different) for each character.  

Model specifications include:  
-word embeddings  
-dropout  
-multiple layers  
-LSTM, GRU, RNN cell  
-padding/masking for variable length sequences  
-gradient clipping  

### Training/Testing/Generating Text
Model can be trained directly using train.py. Once a model is trained and saved, it can be tested using test.py and can generate text using generate.py.  
run.py can instead be used to either train/test/generate text. Furthermore, it can be used to easily set command line arguments, as well as use random hyperparameters and run monte carlo cross validation. Refer to comments in these files for full details on how to run them.

### Other
Models are trained using python 3.5+ and TensorFlow 1.1.  
Initial code based on TensorFlow RNN tutorial https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb  
Seinfeld scripts from https://github.com/colinpollock/seinfeld-scripts

