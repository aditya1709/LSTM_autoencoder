import numpy as np
import matplotlib.pyplot as plt
import h5py

import os,glob
from sklearn import preprocessing
from lstm_vae import create_lstm_ae
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, advanced_activations, BatchNormalization, Dense
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Masking
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import os
import tensorflow as tf
from random import*
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

np.set_printoptions(threshold=np.nan)

K.set_learning_phase(1)


# Standardise and scale data to range [0,1]
def preprocess_data(input, clip_val = 3.0):
    input = preprocessing.scale(np.transpose(input))
    _, I = np.where(np.transpose(input) < -1.0*clip_val)
    input[I,0]=-1.0*clip_val
    _, I = np.where(np.transpose(input) > 1.0*clip_val)
    input[I,0]=1.0*clip_val
    input = (input - np.min(input)) / (np.max(input) - np.min(input))
    return np.transpose(input)

# Extract data and split into training and testing set
# HD5F/ .mat file -v7.3 is accepted
def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
"""Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)

    N, D = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:], data[ind[ind_cut:], 1:], data[ind[:ind_cut], 0], data[ind[ind_cut:], 0]


def repeat(x):
    stepMatrix = K.ones_like(x[0][:,:,:1]) #matrix with ones, shaped as (batch, steps, 1)
    latentMatrix = K.expand_dims(x[1],axis=1) #latent vars, shaped as (batch, 1, latent_dim)
    return K.batch_dot(stepMatrix,latentMatrix)

def cropOutputs(x):
    padding =  K.cast( K.not_equal(x[1],0.0), dtype=K.floatx())
    return x[0]*padding

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = open_data('./UCR_TS_Archive_2015')
    input_dim = X_train.shape[-1] # 13
    timesteps = X_train.shape[1] # 3
    batch_size = 32

    # Create graph structure.
    input_placeholder = Input(shape=(None, input_dim))

    # Encoder.
    masked_input = Masking(mask_value=0.0, input_shape=(None,input_dim), name = 'masking_layer')(input_placeholder)
    encoded = LSTM(60, return_sequences=True, dropout = 0.2,unit_forget_bias=True)(masked_input)
    encoded = advanced_activations.ELU(alpha=.5)(encoded)
    encoded = LSTM(60,return_sequences=True, dropout = 0.2, unit_forget_bias=True)(encoded)
    encoded = advanced_activations.ELU(alpha=.5)(encoded)
    encoded = LSTM(60,dropout = 0.2, unit_forget_bias=True)(encoded)
    encoded = advanced_activations.ELU(alpha=.5)(encoded)
    encoded = Dense(5)(encoded)
    encoded = advanced_activations.ELU(alpha=.5)(encoded)
    encoded_final = BatchNormalization(name='embedding')(encoded)

    # Decoder.
    decoded = Lambda(repeat)([masked_input,encoded_final])
    decoded = LSTM(60, return_sequences=True, dropout = 0.2, unit_forget_bias=True)(decoded)
    decoded = advanced_activations.ELU(alpha=.5)(decoded)
    decoded = LSTM(60, return_sequences=True, dropout = 0.2, unit_forget_bias=True)(decoded)
    decoded = advanced_activations.ELU(alpha=.5)(decoded)
    decoded = LSTM(input_dim, return_sequences=True, dropout = 0.2, unit_forget_bias=True)(decoded)
    decoded = advanced_activations.ELU(alpha=.5)(decoded)
    decoded_final = Lambda(cropOutputs,output_shape=(None,input_dim))([decoded,input_placeholder])

    autoencoder = Model(inputs=input_placeholder, outputs=decoded_final)
    encoder = Model(inputs=input_placeholder, outputs=encoded_final)
    print(autoencoder.summary())

    def vae_loss(input_placeholder, decoded_final):
        loss = objectives.mse(input_placeholder, decoded_final)
        return loss


    adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=adam, loss=vae_loss)

    outputFolder = './output_V2V_exp'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    filepath=outputFolder+"/weights_-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                                 save_best_only=False, save_weights_only=True, \
                                 mode='auto', period=20)

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10,verbose=1, mode='auto')
    callbacks_list = [earlystop,checkpoint]

    #Fit the model
    autoencoder.fit(X_train, X_train, batch_size = batch_size, epochs=2000, callbacks=callbacks_list,validation_split=0.25 )

    #Output the autoencoder model
    decode_data = autoencoder.predict(X_train)

    #Output the decoder model
    latent_data_train = encoder.predict(X_train)

    #Save the output and the latent dimension representation for further processing.
    f1 = h5py.File("Latent_representation_exp.hdf5", "w")
    g1 = f1.create_group('group1')
    g1.create_dataset("Input_data", data = X_train)
    g1.create_dataset("L_rep", data = latent_data_train)
    g1.create_dataset("Decoded_data", data = decode_data)
    f1.close()