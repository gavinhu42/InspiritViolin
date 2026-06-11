# -*- coding: utf-8 -*-
import dataset_prepare as vfp
from custom_layers import Sampling, KLDivergenceLayer, GumbelSoftmaxLayer, GumbelKLDivergenceLayer           

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import random as python_random
from tensorflow.keras import regularizers

# Configure GPU memory
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Model configuration
vae_enc_units = 64 
vae_dec_units = 128 
latent_dim = 16 
rnn_units = 128   
n_dim1 = 64 
n_dim2 = n_dim1 
epochs = 350 
batch_size = 32
validation_split = 0.05
both = 1          
savemodel = True  

n_pitch = 47 
n_start = 96 
n_duration = 32 
n_pure_pitch = 22 
n_pure_octave = 5 
n_spf = 241

# Dictionary Preparation
spf = np.zeros(240)
ct = 0 
for xs in range(1,5):
    for xp in range(1,13):
        for xf in range(5):         
            spf[ct] = 1000*xs + 10*xp + xf
            ct +=1            
spf = np.int32(spf)
spf_unique = np.concatenate(([10,11,12,13,14],spf,[0]))
spf_list = np.concatenate(([0,0,0,0,0],range(1,241),[0]))
spf_dict = {spf_unique[k] : spf_list[k] for k in range(len(spf_unique))}

def make_data(training_corpus, training_key_list, validation_split=0):
    Xtrain, Ytrain = vfp.split_data(training_corpus, training_key_list)
    Xrest = Xtrain['test']
    Xtrain = Xtrain['train']
    Ytrain = Ytrain['train']
        
    start_unique = np.array(np.unique(np.concatenate((Xtrain['start'],Xrest['start'])))*256 , dtype='int')
    start_dict = {start_unique[k] : k for k in range(len(start_unique))}
    
    duration_unique = np.array(np.unique(np.concatenate((Xtrain['duration'],Xrest['duration'])))*256 , dtype='int')
    duration_dict = {duration_unique[k] : k for k in range(len(duration_unique))}
    
    spft = (Ytrain['string']+1)*1000 + (Ytrain['position']+1)*10 + Ytrain['finger']
    
    validation_index = list(range(len(spft)-np.int32(np.ceil(len(spft)*validation_split)),len(spft))) 
    training_index = np.setdiff1d(range(len(spft)),validation_index)
    
    training_data = [Xtrain['pitch'][training_index,:] - 54, 
                      np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')),
                      np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')),
                      Xtrain['pure_pitch'][training_index,:], 
                      Xtrain['pure_octave'][training_index,:] -3, 
                      ]
    training_vae_labels = [keras.utils.to_categorical(Xtrain['pitch'][training_index,:] - 54, n_pitch), 
                           keras.utils.to_categorical(np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')), n_start),
                           keras.utils.to_categorical(np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')), n_duration), 
                           ]
    training_classifier_labels = keras.utils.to_categorical(np.vectorize(spf_dict.__getitem__)(spft[training_index,:]), n_spf)

    if validation_split != 0:
        validation_data = [Xtrain['pitch'][validation_index,:] - 54, 
                           np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][validation_index,:]*256, dtype='int')),
                           np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][validation_index,:]*256, dtype='int')),
                           ]
        validation_vae_labels = [keras.utils.to_categorical(Xtrain['pitch'][validation_index,:] - 54, n_pitch), 
                                 keras.utils.to_categorical(np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][validation_index,:]*256, dtype='int')), n_start), 
                                 keras.utils.to_categorical(np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][validation_index,:]*256, dtype='int')), n_duration),  
                                 ]
        validation_classifier_labels = keras.utils.to_categorical(np.vectorize(spf_dict.__getitem__)(spft[validation_index,:]), n_spf)
    else: 
        validation_data = validation_vae_labels = validation_classifier_labels = None

    class output: None    
    output = output()
    output.training_data = training_data
    output.training_vae_labels = training_vae_labels
    output.training_classifier_labels = training_classifier_labels
    output.validation_data = validation_data
    output.validation_vae_labels = validation_vae_labels
    output.validation_classifier_labels = validation_classifier_labels
    return output

# Loading Datasets
training_full = vfp.load_data('vf_dataset_window32_gap16.pickle')
training_corpus = {k: v for k, v in training_full.items() if 'vio2_' in k} 
seq_len = 32
l1l2 = regularizers.l1_l2(l1=1e-5, l2=1e-4)
l2 = regularizers.l2(1e-4)
    
# --- Architecture Definitions ---
in_pitch = keras.Input(shape=(seq_len,), name='pitch')  
in_start = keras.Input(shape=(seq_len,), name='start')
in_duration = keras.Input(shape=(seq_len,), name='duration')

emb_in = [in_pitch, in_start, in_duration]
x = layers.Concatenate()([layers.Embedding(n_pitch, 16)(in_pitch), layers.Embedding(n_start, 32)(in_start), layers.Embedding(n_duration, 8)(in_duration)])
out_emb = layers.LayerNormalization()(layers.PReLU()(layers.Dense(n_dim1)(x)))
embedder = keras.Model(emb_in, [out_emb], name='embedder')

in_enc = keras.Input(shape=(seq_len, n_dim2, ), name='encoder in')
z = layers.Bidirectional(layers.LSTM(vae_enc_units, return_sequences=True))(in_enc)        
z_mean, z_log_var = KLDivergenceLayer()([layers.Dense(latent_dim)(z), layers.Dense(latent_dim)(z)])
out_enc = Sampling()([z_mean, z_log_var])
encoder = keras.Model([in_enc], [out_enc], name='encoder')

in_cla = keras.Input(shape=(seq_len, n_dim2, ), name='classifier in')
x_cla = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(in_cla)        
out_cla = layers.Activation('softmax')(layers.Dense(n_spf)(x_cla))
classifier = keras.Model([in_cla], [out_cla], name='classifier')   
classifier_gumbel = keras.Model([in_cla], GumbelSoftmaxLayer()(GumbelKLDivergenceLayer()(x_cla)), name='classifier_gumbel')   

in_dec_z = keras.Input(shape=(seq_len, latent_dim, ), name='decoder z input')
in_dec_spf = keras.Input(shape=(seq_len, n_spf, ), name='decoder spf input')
dec_rnn = layers.Bidirectional(layers.LSTM(vae_dec_units, return_sequences=True))(layers.Concatenate()([in_dec_z, in_dec_spf]))
out_dec = [layers.Dense(n_pitch, activation='softmax')(dec_rnn), layers.Dense(n_start, activation='softmax')(dec_rnn), layers.Dense(n_duration, activation='softmax')(dec_rnn)]
decoder = keras.Model([in_dec_z, in_dec_spf], out_dec, name="decoder")

# --- Training Setup ---
emb_inl = [keras.Input(shape=(seq_len,)) for _ in range(3)]
emb_inu = [keras.Input(shape=(seq_len,)) for _ in range(3)]

M2l = keras.Model(emb_inl, [decoder([encoder(embedder(emb_inl)), classifier(embedder(emb_inl))]), classifier(embedder(emb_inl))], name='labelled')
M2u = keras.Model(emb_inu, decoder([encoder(embedder(emb_inu)), classifier_gumbel(embedder(emb_inu))]), name='unlabelled')
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, clipnorm=0.001)

# Logic for data loading
tekl = 0 
tkl, ukl = [1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]
keys_list = list(training_corpus.keys())
labelled_key_list = [keys_list[i] for i in tkl]
unlabelled_key_list = [keys_list[i] for i in ukl]

if both == 1:
    La = make_data(training_corpus, labelled_key_list, validation_split) 
    Un = make_data(training_corpus, unlabelled_key_list, validation_split=0) 
    
    # Corrected Flattened Model and Inputs
    M2 = keras.Model(
        inputs=[emb_inl, emb_inu],
        outputs=[
            M2l(emb_inl)[0][0], M2l(emb_inl)[0][1], M2l(emb_inl)[0][2], M2l(emb_inl)[1], 
            M2u(emb_inu)[0], M2u(emb_inu)[1], M2u(emb_inu)[2]
        ],
        name='M2'
    )
    M2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

    La_inputs = [La.training_data[0], La.training_data[1], La.training_data[2]]
    Un_inputs = [Un.training_data[0], Un.training_data[1], Un.training_data[2]]

    history = M2.fit(
        x = La_inputs + Un_inputs,
        y = [
            La.training_vae_labels[0], La.training_vae_labels[1], La.training_vae_labels[2], 
            La.training_classifier_labels,
            Un.training_vae_labels[0], Un.training_vae_labels[1], Un.training_vae_labels[2]
        ],
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(
            [La.validation_data[0], La.validation_data[1], La.validation_data[2], 
             La.validation_data[0], La.validation_data[1], La.validation_data[2]], 
            [
                La.validation_vae_labels[0], La.validation_vae_labels[1], La.validation_vae_labels[2], 
                La.validation_classifier_labels,
                La.validation_vae_labels[0], La.validation_vae_labels[1], La.validation_vae_labels[2]
            ]
        )
    )
else:
    La = make_data(training_corpus, labelled_key_list, validation_split) 
    M2l.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
    history = M2l.fit(x = La.training_data[0:3], y = [La.training_vae_labels[0], La.training_vae_labels[1], La.training_vae_labels[2], La.training_classifier_labels], batch_size=batch_size, epochs=epochs, verbose=2)

print("End training")
keras.backend.clear_session()