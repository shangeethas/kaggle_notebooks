#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# It demonstrats how to utilize [the unified Wi-Fi dataset](https://www.kaggle.com/kokitanisaka/indoorunifiedwifids).<br>
# The Neural Net model is not optimized, there's much space to improve the score. 
# 
# In this notebook, I refer these two excellent notebooks.
# * [wifi features with lightgbm/KFold](https://www.kaggle.com/hiro5299834/wifi-features-with-lightgbm-kfold) by [@hiro5299834](https://www.kaggle.com/hiro5299834/)<br>
#  I took some code fragments from his notebook.
# * [Simple 👌 99% Accurate Floor Model 💯](https://www.kaggle.com/nigelhenry/simple-99-accurate-floor-model) by [@nigelhenry](https://www.kaggle.com/nigelhenry/)<br>
#  I use his excellent work, the "floor" prediction.
# 
# It takes much much time to finish learning. <br>
# And even though I enable the GPU, it doesn't help. <br>
# If anybody knows how to make it better, can you please make a comment? <br>
# 
# Thank you!

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import glob
import pickle

import random
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


# ### options
# We can change the way it learns with these options. <br>
# Especialy **NUM_FEATS** is one of the most important options. <br>
# It determines how many features are used in the training. <br>
# We have 100 Wi-Fi features in the dataset, but 100th Wi-Fi signal sounds not important, right? <br>
# So we can use top Wi-Fi signals if we think we need to. 

# In[2]:


# options

N_SPLITS = 10

SEED = 2021

NUM_FEATS = 20 # number of features that we use. there are 100 feats but we don't need to use all of them

base_path = '/kaggle'


# In[3]:


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    
def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat-x, 2) + np.power(yhat-y, 2)) + 15 * np.abs(fhat-f)
    return intermediate.sum()/xhat.shape[0]


# In[4]:


feature_dir = f"{base_path}/input/indoorunifiedwifids"
train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, '*_test.csv')))
subm = pd.read_csv(f'{base_path}/input/indoor-location-navigation/sample_submission.csv', index_col=0)


# In[5]:


with open(f'{feature_dir}/train_all.pkl', 'rb') as f:
  data = pickle.load( f)

with open(f'{feature_dir}/test_all.pkl', 'rb') as f:
  test_data = pickle.load(f)


# In[6]:


# training target features

BSSID_FEATS = [f'bssid_{i}' for i in range(NUM_FEATS)]
RSSI_FEATS  = [f'rssi_{i}' for i in range(NUM_FEATS)]


# In[7]:


# get numbers of bssids to embed them in a layer

wifi_bssids = []
for i in range(100):
    wifi_bssids.extend(data.iloc[:,i].values.tolist())
wifi_bssids = list(set(wifi_bssids))

wifi_bssids_size = len(wifi_bssids)
print(f'BSSID TYPES: {wifi_bssids_size}')

wifi_bssids_test = []
for i in range(100):
    wifi_bssids_test.extend(test_data.iloc[:,i].values.tolist())
wifi_bssids_test = list(set(wifi_bssids_test))

wifi_bssids_size = len(wifi_bssids_test)
print(f'BSSID TYPES: {wifi_bssids_size}')

wifi_bssids.extend(wifi_bssids_test)
wifi_bssids_size = len(wifi_bssids)


# In[8]:


# preprocess

le = LabelEncoder()
le.fit(wifi_bssids)
le_site = LabelEncoder()
le_site.fit(data['site_id'])

ss = StandardScaler()
ss.fit(data.loc[:,RSSI_FEATS])


# In[9]:


data.loc[:,RSSI_FEATS] = ss.transform(data.loc[:,RSSI_FEATS])
for i in BSSID_FEATS:
    data.loc[:,i] = le.transform(data.loc[:,i])
    data.loc[:,i] = data.loc[:,i] + 1
    
data.loc[:, 'site_id'] = le_site.transform(data.loc[:, 'site_id'])

data.loc[:,RSSI_FEATS] = ss.transform(data.loc[:,RSSI_FEATS])


# In[10]:


test_data.loc[:,RSSI_FEATS] = ss.transform(test_data.loc[:,RSSI_FEATS])
for i in BSSID_FEATS:
    test_data.loc[:,i] = le.transform(test_data.loc[:,i])
    test_data.loc[:,i] = test_data.loc[:,i] + 1
    
test_data.loc[:, 'site_id'] = le_site.transform(test_data.loc[:, 'site_id'])

test_data.loc[:,RSSI_FEATS] = ss.transform(test_data.loc[:,RSSI_FEATS])


# In[11]:


site_count = len(data['site_id'].unique())
data.reset_index(drop=True, inplace=True)


# In[12]:


set_seed(SEED)


# ## The model
# The first Embedding layer is very important. <br>
# Thanks to the layer, we can make sense of these BSSID features. <br>
# <br>
# We concatenate all the features and put them into LSTM. <br>
# <br>
# If something is theoritically wrong, please correct me. Thank you in advance. 

# In[13]:


def create_model(input_data):

    # bssid feats
    input_dim = input_data[0].shape[1]

    input_embd_layer = L.Input(shape=(input_dim,))
    x1 = L.Embedding(wifi_bssids_size, 64)(input_embd_layer)
    x1 = L.Flatten()(x1)

    # rssi feats
    input_dim = input_data[1].shape[1]

    input_layer = L.Input(input_dim, )
    x2 = L.BatchNormalization()(input_layer)
    x2 = L.Dense(NUM_FEATS * 64, activation='relu')(x2)

    # site
    input_site_layer = L.Input(shape=(1,))
    x3 = L.Embedding(site_count, 2)(input_site_layer)
    x3 = L.Flatten()(x3)


    # main stream
    x = L.Concatenate(axis=1)([x1, x3, x2])

    x = L.BatchNormalization()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(256, activation='relu')(x)

    x = L.Reshape((1, -1))(x)
    x = L.BatchNormalization()(x)
    x = L.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, activation='relu')(x)
    x = L.LSTM(16, dropout=0.1, return_sequences=False, activation='relu')(x)

    
    output_layer_1 = L.Dense(2, name='xy')(x)
    output_layer_2 = L.Dense(1, activation='softmax', name='floor')(x)

    model = M.Model([input_embd_layer, input_layer, input_site_layer], 
                    [output_layer_1, output_layer_2])

    model.compile(optimizer=tf.optimizers.Adam(lr=0.001),
                  loss='mse', metrics=['mse'])

    return model


# In[14]:


score_df = pd.DataFrame()
oof = list()
predictions = list()

oof_x, oof_y, oof_f = np.zeros(data.shape[0]), np.zeros(data.shape[0]), np.zeros(data.shape[0])
preds_x, preds_y = 0, 0
preds_f_arr = np.zeros((test_data.shape[0], N_SPLITS))

for fold, (trn_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(data.loc[:, 'path'], data.loc[:, 'path'])):
    X_train = data.loc[trn_idx, BSSID_FEATS + RSSI_FEATS + ['site_id']]
    y_trainx = data.loc[trn_idx, 'x']
    y_trainy = data.loc[trn_idx, 'y']
    y_trainf = data.loc[trn_idx, 'floor']

    tmp = pd.concat([y_trainx, y_trainy], axis=1)
    y_train = [tmp, y_trainf]

    X_valid = data.loc[val_idx, BSSID_FEATS + RSSI_FEATS + ['site_id']]
    y_validx = data.loc[val_idx, 'x']
    y_validy = data.loc[val_idx, 'y']
    y_validf = data.loc[val_idx, 'floor']

    tmp = pd.concat([y_validx, y_validy], axis=1)
    y_valid = [tmp, y_validf]

    model = create_model([X_train.loc[:,BSSID_FEATS], X_train.loc[:,RSSI_FEATS], X_train.loc[:,'site_id']])
    model.fit([X_train.loc[:,BSSID_FEATS], X_train.loc[:,RSSI_FEATS], X_train.loc[:,'site_id']], y_train, 
                validation_data=([X_valid.loc[:,BSSID_FEATS], X_valid.loc[:,RSSI_FEATS], X_valid.loc[:,'site_id']], y_valid), 
                batch_size=128, epochs=1000,
                callbacks=[
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')
                , ModelCheckpoint(f'{base_path}/RNN_{SEED}_{fold}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
                , EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, mode='min', baseline=None, restore_best_weights=True)
            ])

    model.load_weights(f'{base_path}/RNN_{SEED}_{fold}.hdf5')
    val_pred = model.predict([X_valid.loc[:,BSSID_FEATS], X_valid.loc[:,RSSI_FEATS], X_valid.loc[:,'site_id']])

    oof_x[val_idx] = val_pred[0][:,0]
    oof_y[val_idx] = val_pred[0][:,1]
    oof_f[val_idx] = val_pred[1][:,0].astype(int)

    pred = model.predict([test_data.loc[:,BSSID_FEATS], test_data.loc[:,RSSI_FEATS], test_data.loc[:,'site_id']]) # test_data.iloc[:, :-1])
    preds_x += pred[0][:,0]
    preds_y += pred[0][:,1]
    preds_f_arr[:, fold] = pred[1][:,0].astype(int)

    score = comp_metric(oof_x[val_idx], oof_y[val_idx], oof_f[val_idx],
                        y_validx.to_numpy(), y_validy.to_numpy(), y_validf.to_numpy())
    print(f"fold {fold}: mean position error {score}")

    break # for demonstration, run just one fold as it takes much time.

preds_x /= (fold + 1)
preds_y /= (fold + 1)
    
print("*+"*40)
# as it breaks in the middle of cross-validation, the score is not accurate at all.
score = comp_metric(oof_x, oof_y, oof_f, data.iloc[:, -5].to_numpy(), data.iloc[:, -4].to_numpy(), data.iloc[:, -3].to_numpy())
oof.append(score)
print(f"mean position error {score}")
print("*+"*40)

preds_f_mode = stats.mode(preds_f_arr, axis=1)
preds_f = preds_f_mode[0].astype(int).reshape(-1)
test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
test_preds.columns = subm.columns
test_preds.index = test_data["site_path_timestamp"]
test_preds["floor"] = test_preds["floor"].astype(int)
predictions.append(test_preds)


# In[15]:


all_preds = pd.concat(predictions)
all_preds = all_preds.reindex(subm.index)


# ## Fix the floor prediction
# So far, it is not successfully make the "floor" prediction part with this dataset. <br>
# To make it right, we can incorporate [@nigelhenry](https://www.kaggle.com/nigelhenry/)'s [excellent work](https://www.kaggle.com/nigelhenry/simple-99-accurate-floor-model). <br>

# In[16]:


simple_accurate_99 = pd.read_csv('../input/simple-99-accurate-floor-model/submission.csv')

all_preds['floor'] = simple_accurate_99['floor'].values


# In[17]:


all_preds.to_csv('submission.csv')


# That's it. 
# 
# Thank you for reading all of it.
# 
# I hope it helps!
# 
# Please make comments if you found something to point out, insights or suggestions. 
