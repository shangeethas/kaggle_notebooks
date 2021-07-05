#!/usr/bin/env python
# coding: utf-8

# <h1><center>INGV - Volcanic Eruption Prediction. Data Analysis.</center></h1>
# 
# <center><img src="https://images.ctfassets.net/81iqaqpfd8fy/3Wp4SEgzagcICaSqcIMOQM/5721655abf93a19521dad8a35d747f2d/Erupting_Volcano.jpg?h=620&w=1440"></center>

# In[1]:


import numpy as np
import pandas as pd
import glob
import math

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import optuna
from optuna.samplers import TPESampler

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


pd.set_option('display.max_columns', None)


# <a id="1"></a>
# <h2 style='background:black; border:0; color:white'><center>1. Data overview<center><h2>

# In[3]:


train = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")
sample_submission = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv")


# **train.csv** Metadata for the train files.
# 
# * segment_id: ID code for the data segment. Matches the name of the associated data file.
# 
# * time_to_eruption: The target value, the time until the next eruption.

# In[4]:


train


# In[5]:


fig = px.histogram(
    train, 
    x="time_to_eruption",
    width=800,
    height=500,
    nbins=100,
    title='Time to eruption distribution'
)

fig.show()


# In[6]:


fig = px.line(
    train, 
    y="time_to_eruption",
    width=800,
    height=500,
    title='Time to eruption for all volcanos'
)

fig.show()


# In[7]:


train['time_to_eruption'].describe()


# In[8]:


print('Median:', train['time_to_eruption'].median())
print('Skew:', train['time_to_eruption'].skew())
print('Std:', train['time_to_eruption'].std())
print('Kurtosis:', train['time_to_eruption'].kurtosis())
print('Mean:', train['time_to_eruption'].mean())


# In[9]:


sample_submission


# **[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# <a id="2"></a>
# <h2 style='background:black; border:0; color:white'><center>2. Train & test fragments<center><h2>

# In[10]:


train_frags = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/train/*")
len(train_frags)


# In[11]:


test_frags = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/test/*")
len(test_frags)


# In[12]:


train_frags[0]


# In[13]:


check = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train/2037160701.csv')
check


# We can see that we have data from 10 sensors and 60001 observation for each of them.

# Let's check number of observations and number of sensors for every sample in train directory.

# In[14]:


sensors = set()
observations = set()
nan_columns = list()
missed_groups = list()
for_df = list()

for item in train_frags:
    name = int(item.split('.')[-2].split('/')[-1])
    at_least_one_missed = 0
    frag = pd.read_csv(item)
    missed_group = list()
    missed_percents = list()
    for col in frag.columns:
        missed_percents.append(frag[col].isnull().sum() / len(frag))
        if pd.isnull(frag[col]).all() == True:
            at_least_one_missed = 1
            nan_columns.append(col)
            missed_group.append(col)
    if len(missed_group) > 0:
        missed_groups.append(missed_group)
    sensors.add(len(frag.columns))
    observations.add(len(frag))
    for_df.append([name, at_least_one_missed] + missed_percents)


# In[15]:


print('Unique number of sensors: ', sensors)
print('Unique number of observations: ', observations)


# Let's see how many missed sensors we have in training set.

# In[16]:


print('Number of totaly missed sensors:', len(nan_columns))

absent_sensors = dict()

for item in nan_columns:
    if item in absent_sensors:
        absent_sensors[item] += 1
    else:
        absent_sensors[item] = 0


# In[17]:


absent_df = pd.DataFrame(absent_sensors.items(), columns=['Sensor', 'Missed sensors'])

fig = px.bar(
    absent_df, 
    x="Sensor",
    y='Missed sensors',
    width=800,
    height=500,
    title='Number of missed sensors in training dataset'
)

fig.show()


# As we can see we don't have missed sensors for all volcanos for sensor_4 and sensor_6 in trainig set.

# Let's see now missed combinations of sensors.

# In[18]:


absent_groups = dict()

for item in missed_groups:
    if str(item) in absent_groups:
        absent_groups[str(item)] += 1
    else:
        absent_groups[str(item)] = 0


# In[19]:


absent_df = pd.DataFrame(absent_groups.items(), columns=['Group', 'Missed number'])
absent_df = absent_df.sort_values('Missed number')

fig = px.bar(
    absent_df, 
    y="Group",
    x='Missed number',
    orientation='h',
    width=800,
    height=600,
    title='Number of missed sensor groups in training dataset'
)

fig.show()


# In[20]:


for_df = pd.DataFrame(
    for_df, 
    columns=[
        'segment_id', 'has_missed_sensors', 'missed_percent_sensor1', 
        'missed_percent_sensor2', 'missed_percent_sensor3', 'missed_percent_sensor4', 
        'missed_percent_sensor5', 'missed_percent_sensor6', 'missed_percent_sensor7', 
        'missed_percent_sensor8', 'missed_percent_sensor9', 'missed_percent_sensor10'
    ]
)

for_df


# In[21]:


train = pd.merge(train, for_df)
train


# Let's check time_to_eruption for segments with and without missed sensors.

# In[22]:


fig = make_subplots(rows=1, cols=2)
traces = [
    go.Histogram(
        x=train[train['has_missed_sensors']==1]['time_to_eruption'], 
        nbinsx=100, 
        name='Has missed sensors'
    ),
    go.Histogram(
        x=train[train['has_missed_sensors']==0]['time_to_eruption'], 
        nbinsx=100, 
        name="Doesn't have missed sensors"
    )
]

for i in range(len(traces)):
    fig.append_trace(
        traces[i], 
        (i // 2) + 1, 
        (i % 2) + 1
    )

fig.update_layout(
    title_text='Time to erruption distribution for segments with / without missed sensors',
    height=600,
    width=1000
)
fig.show()


# Let's do the same for test set.

# In[23]:


sensors = set()
observations = set()
nan_columns = list()
missed_groups = list()
for_test_df = list()

for item in test_frags:
    name = int(item.split('.')[-2].split('/')[-1])
    at_least_one_missed = 0
    frag = pd.read_csv(item)
    missed_group = list()
    missed_percents = list()
    for col in frag.columns:
        missed_percents.append(frag[col].isnull().sum() / len(frag))
        if pd.isnull(frag[col]).all() == True:
            at_least_one_missed = 1
            nan_columns.append(col)
            missed_group.append(col)
    if len(missed_group) > 0:
        missed_groups.append(missed_group)
    sensors.add(len(frag.columns))
    observations.add(len(frag))
    for_test_df.append([name, at_least_one_missed] + missed_percents)


# In[24]:


for_test_df = pd.DataFrame(
    for_test_df, 
    columns=[
        'segment_id', 'has_missed_sensors', 'missed_percent_sensor1', 'missed_percent_sensor2', 'missed_percent_sensor3', 
        'missed_percent_sensor4', 'missed_percent_sensor5', 'missed_percent_sensor6', 'missed_percent_sensor7', 
        'missed_percent_sensor8', 'missed_percent_sensor9', 'missed_percent_sensor10'
    ]
)

for_test_df


# In[25]:


print('Unique number of sensors: ', sensors)
print('Unique number of observations: ', observations)


# In[26]:


print('Number of totaly missed sensors:', len(nan_columns))

absent_sensors = dict()

for item in nan_columns:
    if item in absent_sensors:
        absent_sensors[item] += 1
    else:
        absent_sensors[item] = 0


# In[27]:


absent_df = pd.DataFrame(absent_sensors.items(), columns=['Sensor', 'Missed sensors'])

fig = px.bar(
    absent_df, 
    x="Sensor",
    y='Missed sensors',
    width=800,
    height=500,
    title='Number of missed sensors in test dataset'
)

fig.show()


# In[28]:


absent_groups = dict()

for item in missed_groups:
    if str(item) in absent_groups:
        absent_groups[str(item)] += 1
    else:
        absent_groups[str(item)] = 0


# In[29]:


absent_df = pd.DataFrame(absent_groups.items(), columns=['Group', 'Missed number'])
absent_df = absent_df.sort_values('Missed number')

fig = px.bar(
    absent_df, 
    y="Group",
    x='Missed number',
    orientation='h',
    width=800,
    height=600,
    title='Number of missed sensor groups in test dataset'
)

fig.show()


# Let's check data from all sensors for current volcano.

# In[30]:


fig = make_subplots(rows=5, cols=2)
traces = [
    go.Histogram(
        x=check[col], 
        nbinsx=100, 
        name=col
    ) for col in check.columns
]

for i in range(len(traces)):
    fig.append_trace(
        traces[i], 
        (i // 2) + 1, 
        (i % 2) + 1
    )

fig.update_layout(
    title_text='Data from sensors distribution',
    height=800,
    width=1200
)

fig.show()


# In[31]:


fig = make_subplots(rows=5, cols=2)
traces = [
    go.Scatter(
        x=[i for i in range(60002)], 
        y=check[col], 
        mode='lines', 
        name=col
    ) for col in check.columns
]

for i in range(len(traces)):
    fig.append_trace(
        traces[i], 
        (i // 2) + 1, 
        (i % 2) + 1
    )

fig.update_layout(
    title_text='Data from sensors',
    height=800,
    width=1200
)

fig.show()


# <a id="3"></a>
# <h2 style='background:black; border:0; color:white'><center>3. Modeling<center><h2>

# In[32]:


def build_features(signal, ts, sensor_id):
    X = pd.DataFrame()
    f = np.fft.fft(signal)
    f_real = np.real(f)
    X.loc[ts, f'{sensor_id}_sum']       = signal.sum()
    X.loc[ts, f'{sensor_id}_mean']      = signal.mean()
    X.loc[ts, f'{sensor_id}_std']       = signal.std()
    X.loc[ts, f'{sensor_id}_var']       = signal.var() 
    X.loc[ts, f'{sensor_id}_max']       = signal.max()
    X.loc[ts, f'{sensor_id}_min']       = signal.min()
    X.loc[ts, f'{sensor_id}_skew']      = signal.skew()
    X.loc[ts, f'{sensor_id}_mad']       = signal.mad()
    X.loc[ts, f'{sensor_id}_kurtosis']  = signal.kurtosis()
    X.loc[ts, f'{sensor_id}_quantile99']= np.quantile(signal, 0.99)
    X.loc[ts, f'{sensor_id}_quantile95']= np.quantile(signal, 0.95)
    X.loc[ts, f'{sensor_id}_quantile85']= np.quantile(signal, 0.85)
    X.loc[ts, f'{sensor_id}_quantile75']= np.quantile(signal, 0.75)
    X.loc[ts, f'{sensor_id}_quantile55']= np.quantile(signal, 0.55)
    X.loc[ts, f'{sensor_id}_quantile45']= np.quantile(signal, 0.45) 
    X.loc[ts, f'{sensor_id}_quantile25']= np.quantile(signal, 0.25) 
    X.loc[ts, f'{sensor_id}_quantile15']= np.quantile(signal, 0.15) 
    X.loc[ts, f'{sensor_id}_quantile05']= np.quantile(signal, 0.05)
    X.loc[ts, f'{sensor_id}_quantile01']= np.quantile(signal, 0.01)
    X.loc[ts, f'{sensor_id}_fft_real_mean']= f_real.mean()
    X.loc[ts, f'{sensor_id}_fft_real_std'] = f_real.std()
    X.loc[ts, f'{sensor_id}_fft_real_max'] = f_real.max()
    X.loc[ts, f'{sensor_id}_fft_real_min'] = f_real.min()

    return X


# In[33]:


train_set = list()
j=0
for seg in train.segment_id:
    signals = pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/{seg}.csv')
    train_row = []
    if j%500 == 0:
        print(j)
    for i in range(0, 10):
        sensor_id = f'sensor_{i+1}'
        train_row.append(build_features(signals[sensor_id].fillna(0), seg, sensor_id))
    train_row = pd.concat(train_row, axis=1)
    train_set.append(train_row)
    j+=1

train_set = pd.concat(train_set)


# In[34]:


train_set = train_set.reset_index()
train_set = train_set.rename(columns={'index': 'segment_id'})
train_set = pd.merge(train_set, train, on='segment_id')
train_set


# In[35]:


drop_cols = list()
for col in train_set.columns:
    if col == 'segment_id':
        continue
    if abs(train_set[col].corr(train_set['time_to_eruption'])) < 0.01:
        drop_cols.append(col)


# In[36]:


not_to_drop_cols = list()

for col1 in train_set.columns:
    for col2 in train_set.columns:
        if col1 == col2:
            continue
        if col1 == 'segment_id' or col2 == 'segment_id': 
            continue
        if col1 == 'time_to_eruption' or col2 == 'time_to_eruption':
            continue
        if abs(train_set[col1].corr(train_set[col2])) > 0.98:
            if col2 not in drop_cols and col1 not in not_to_drop_cols:
                drop_cols.append(col2)
                not_to_drop_cols.append(col1)


# In[37]:


train = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
y = train_set['time_to_eruption']


# In[38]:


reduced_y = y.copy()
reduced_train = train.copy()
reduced_train = reduced_train.drop(drop_cols, axis=1)
reduced_train


# In[39]:


train, val, y, y_val = train_test_split(train, y, random_state=666, test_size=0.2, shuffle=True)
reduced_train, reduced_val, reduced_y, reduced_y_val = train_test_split(reduced_train, reduced_y, random_state=666, test_size=0.2, shuffle=True)


# In[40]:


lgb = LGBMRegressor(
    random_state=666, 
    max_depth=7, 
    n_estimators=250, 
    learning_rate=0.12
)

lgb.fit(train, y)
preds = lgb.predict(val)


# In[41]:


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


# In[42]:


print('Simple LGB model rmse: ', rmse(y_val, preds))


# In[43]:


sampler = TPESampler(seed=666)

def create_model(trial):
    num_leaves = trial.suggest_int("num_leaves", 2, 31)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 8)
    min_child_samples = trial.suggest_int('min_child_samples', 100, 1200)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 90)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.0001, 1.0)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.0001, 1.0)
    model = LGBMRegressor(
        num_leaves=num_leaves,
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_child_samples=min_child_samples, 
        min_data_in_leaf=min_data_in_leaf,
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        random_state=666
    )
    return model

def objective(trial):
    model = create_model(trial)
    model.fit(train, y)
    preds = model.predict(val)
    score = rmse(y_val, preds)
    return score

# To use optuna uncomment it 
# study = optuna.create_study(direction="minimize", sampler=sampler)
# study.optimize(objective, n_trials=5000)
# params = study.best_params
# params['random_state'] = 666

params = {
    'num_leaves': 29,
    'n_estimators': 289,
    'max_depth': 8,
    'min_child_samples': 507,
    'learning_rate': 0.0812634327662599,
    'min_data_in_leaf': 13,
    'bagging_fraction': 0.020521665677937423,
    'feature_fraction': 0.05776459974779927,
    'random_state': 666
}

lgb = LGBMRegressor(**params)
lgb.fit(train, y)


# In[44]:


preds = lgb.predict(val)
print('Optimized LGB model rmse: ', rmse(y_val, preds))


# In[45]:


parms = {
    'num_leaves': 31, 
    'n_estimators': 138, 
    'max_depth': 8, 
    'min_child_samples': 182, 
    'learning_rate': 0.16630987899513125, 
    'min_data_in_leaf': 24, 
    'bagging_fraction': 0.8743237361979733, 
    'feature_fraction': 0.45055692472636766,
    'random_state': 666
}

rfe_lgb = RFE(
    estimator=DecisionTreeRegressor(
        random_state=666
    ), 
    n_features_to_select=83
)

pipe_lgb = Pipeline(
    steps=[
        ('s', rfe_lgb), 
        ('m', LGBMRegressor(**parms))
    ]
)

pipe_lgb.fit(train, y)
preds = pipe_lgb.predict(val)  
print('LGB rmse', rmse(y_val, preds))


# In[46]:


params = {
    'max_depth': 11, 
    'n_estimators': 245, 
    'learning_rate': 0.0925872303097654, 
    'gamma': 0.6154687206061559,
    'random_state': 666
}

rfe_estimator = RFE(estimator=DecisionTreeRegressor(random_state=666), n_features_to_select=60)
pipe = Pipeline(
    steps=[
        ('s', rfe_estimator),
        ('m', XGBRegressor(**params))
    ]
)

pipe.fit(train, y)
preds = pipe.predict(val)  
print('XGBoost rmse', rmse(y_val, preds))


# In[47]:


params = {
    'max_depth': 6,
    'n_estimators': 189,
    'learning_rate': 0.09910718143795864,
    'gamma': 0.787986320220815,
    'random_state': 666
}

xgb_short = XGBRegressor(
    **params
)
xgb_short.fit(reduced_train, reduced_y)
preds = xgb_short.predict(reduced_val)  
print('XGBoost rmse', rmse(reduced_y_val, preds))


# In[48]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((241,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1000, activation="sigmoid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    
    model.compile(
        loss=root_mean_squared_error, 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model


# In[49]:


yy = np.log1p(y)


# In[50]:


models = list()
preds = list()

for n, (tr, te) in enumerate(KFold(
    n_splits=3, 
    random_state=666, 
    shuffle=True).split(yy)):
    
    early_stopping = EarlyStopping(
        patience=10, 
        verbose=0
    )
    
    print(f'Fold {n}')
    
    model = create_model()
    
    model.fit(
        train.values[tr],
        yy.values[tr],
        epochs=4000,
        batch_size=128,
        verbose=0,
        callbacks=[early_stopping]
    )
    
    pred = model.predict(val)
    pred = np.expm1(pred).reshape((pred.shape[0], ))
    preds.append(pred)
    print('Fold rmse', rmse(yy.values[te], model.predict(train.values[te]))) 
    models.append(model)


# In[51]:


predictions = preds[0]
for i in range(1, 3):
    predictions += preds[i]
predictions /= 3

print('NN rmse', rmse(y_val, predictions))


# In[52]:


sample_submission


# In[53]:


test_set = list()
j=0
for seg in sample_submission.segment_id:
    signals = pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{seg}.csv')
    test_row = []
    if j%500 == 0:
        print(j)
    for i in range(0, 10):
        sensor_id = f'sensor_{i+1}'
        test_row.append(build_features(signals[sensor_id].fillna(0), seg, sensor_id))
    test_row = pd.concat(test_row, axis=1)
    test_set.append(test_row)
    j+=1
test_set = pd.concat(test_set)


# In[54]:


test_set = test_set.reset_index()
test_set = test_set.rename(columns={'index': 'segment_id'})
test_set = pd.merge(test_set, for_test_df, on='segment_id')
test = test_set.drop(['segment_id'], axis=1)
test


# In[55]:


reduced_test = test.copy()
reduced_test = reduced_test.drop(drop_cols, axis=1)
reduced_test


# In[56]:


preds1 = lgb.predict(test)
preds1


# In[57]:


preds2 = pipe_lgb.predict(test)
preds2


# In[58]:


preds3 = pipe.predict(test)
preds3


# In[59]:


predictions = list()
for model in models:
    pred = model.predict(test)
    pred = np.expm1(pred).reshape((pred.shape[0], ))
    predictions.append(pred)

preds4 = predictions[0]
for i in range(1, 3):
    preds4 += predictions[i]
preds4 /= 3

preds4


# In[60]:


preds5 = xgb_short.predict(reduced_test)
preds5


# In[61]:


test_set['time_to_eruption'] = preds1 * 0.5 + preds2 * 0.05 + preds3 * 0.3 + preds4 * 0.05 + preds5 * 0.1


# In[62]:


sample_submission = pd.merge(sample_submission, test_set[['segment_id', 'time_to_eruption']], on='segment_id')


# In[63]:


sample_submission = sample_submission.drop(['time_to_eruption_x'], axis=1)
sample_submission.columns = ['segment_id', 'time_to_eruption']
sample_submission


# In[64]:


sample_submission.to_csv('submission.csv', index=False)


# ## WORK IN PROGRESS

# In[ ]:




