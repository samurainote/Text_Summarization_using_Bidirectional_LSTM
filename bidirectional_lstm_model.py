
from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split
import logging

import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import pydot


import keras
from keras import backend as k
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# ========================================================================
# model params
# ========================================================================

en_shape = np.shape(train_data["article"][0])
den_shape = np.shape(train_data["summary"][0])

MAX_ART_LEN =
MAX_SUM_LEN =
EMBEDDING_DIM = 1000
HIDDEN_UNITS =

LEARNING_RATE = 0.002
BATCH_SIZE = 32
EPOCHS = 5

rmsprop = RMSprop(lr=LEARNING_RATE, clipnorm=1.0)


# ========================================================================
# model helpers
# ========================================================================

def bidirectional_lstm(data):
    """
    Encoder-Decoder-seq2seq
    """
    # encoder
    encoder_inputs = Input(shape=en_shape[1])
    encoder_LSTM = LSTM(HIDDEN_UNITS, dropout_U = 0.2, dropout_W = 0.2 ,return_state=True)
    rev_encoder_LSTM = LSTM(HIDDEN_UNITS, return_state=True, go_backwards=True)
    #
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)
    rev_encoder_outputs, rev_state_h, rev_state_c = rev_encoder_LSTM(encoder_inputs)
    #
    final_state_h = Add()([state_h, rev_state_h])
    final_state_c = Add()([state_c, rev_state_c])

    encoder_states = [final_state_h, final_state_c]

    # decoder
    decoder_inputs = Input(shape=(None, de_shape[1]))
    decoder_LSTM = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _, = decoder_LSTM(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(units=de_shape[1], activation="linear")
    decoder_outputs = decoder_dense(decoder_outputs)

    # modeling
    model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
    model.compile(optimizer=rmsprop, loss="mse", metrics=["accuracy"])

    # return model
    print(model.summary())

    x_train, x_test, y_train, y_test = train_test_split(data["article"], data["summaries"], test_size=0.2)
    model.fit([x_train, y_train], y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1, validation_data=([x_test, y_test], y_test))

    """
    推論モデル
    """
    encoder_model_inf = Model(encoder_inputs, encoder_states)

    decoder_state_input_H = Input(shape=(HIDDEN_UNITS,))
    decoder_state_input_C = Input(shape=(HIDDEN_UNITS,))

    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs, initial_state=decoder_state_inputs)

    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model_inf = Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)

    scores = model.evaluate([x_test, y_test], y_test, verbose=0)

    print('LSTM test scores:', scores)
    print('\007')

    return model, encoder_model_inf, decoder_model_inf

"""
model._make_predict_function()
"""
