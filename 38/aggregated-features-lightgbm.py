#!/usr/bin/env python
# coding: utf-8

# In this kernel we will take a look at engineering aggregated features. Specifically, we will engineer 3 aggregated features using `train_active.csv`, `test_active.csv`, `periods_test.csv` and `periods_test.csv`. Those features will be:
# - `avg_times_up_user` - how often the average item of the user has been put up for sale.
# - `avg_days_up_user` - the average number of days an item from the user has been put up for sale.
# - `n_user_items` - the number of items the user has put up for sale.
# 
# Let's see if they help :)

# # Engineering aggregated features

# In[ ]:


import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import scipy
import lightgbm as lgb

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# Start with loading the data. To save some memory, we only load `item_id` and `user_id`, as that's all we need for the proposed features.

# In[ ]:


used_cols = ['item_id', 'user_id']

train = pd.read_csv('../input/train.csv', usecols=used_cols)
train_active = pd.read_csv('../input/train_active.csv', usecols=used_cols)
test = pd.read_csv('../input/test.csv', usecols=used_cols)
test_active = pd.read_csv('../input/test_active.csv', usecols=used_cols)

train_periods = pd.read_csv('../input/periods_train.csv', parse_dates=['date_from', 'date_to'])
test_periods = pd.read_csv('../input/periods_test.csv', parse_dates=['date_from', 'date_to'])

train.head()


# It's time for some visualizations. The following venn diagrams show the overlap of the user ID between the relevant dataframes. If this overlap is reasonably large, it might be a good idea to use the aggregated features.

# In[ ]:


fig, axarr = plt.subplots(2, 2, figsize=(14, 7))

def get_venn(axarr, feature):
    axarr[0, 0].set_title(f'Overlap between {feature} in train and train_active')
    venn2([
        set(train[feature].values), 
        set(train_active[feature].values)
    ], set_labels = ('train', 'train_active'), ax=axarr[0, 0])

    axarr[0, 1].set_title(f'Overlap between {feature} in test and test_active')
    venn2([
        set(test[feature].values), 
        set(test_active[feature].values)
    ], set_labels = ('test', 'test_active'), ax=axarr[0, 1])

    axarr[1, 0].set_title(f'Overlap between {feature} in train and test')
    venn2([
        set(train[feature].values), 
        set(test[feature].values)
    ], set_labels = ('train', 'test'), ax=axarr[1, 0])

    axarr[1, 1].set_title(f'Overlap between {feature} in train_active and test_active')
    venn2([
        set(train_active[feature].values), 
        set(test_active[feature].values)
    ], set_labels = ('train_active', 'test_active'), ax=axarr[1, 1])
    
get_venn(axarr, 'user_id')


# We're lucky! There is a huge overlap between the IDs of `train` / `train_active` and `test` / `test_active`. Out of curiosity, we'll also take a look at the overlap of item ID. This should (hopefully) not overlap at all.

# In[ ]:


fig, axarr = plt.subplots(2, 2, figsize=(14, 7))

get_venn(axarr, 'item_id')


# As suspected, there is no overlap between the dataframes. Except of `train_active` and `test_active`. These might be duplicated rows or items that have been put up for sale multiple times. We will have to filter these duplicated IDs for our engineered features.
# 
# Anyway, we will now merge the data into one dataframe and, as mentioned, drop the duplicate item IDs. At this point we can also delete `train_active` and `test_active` to free up some memory.

# In[ ]:


all_samples = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)
all_samples.drop_duplicates(['item_id'], inplace=True)

del train_active
del test_active
gc.collect()


# We will also concatenate the train and test period data to one dataframe for easier processing.

# In[ ]:


all_periods = pd.concat([
    train_periods,
    test_periods
])

del train_periods
del test_periods
gc.collect()

all_periods.head()


# Now the interesting part begins! For our feature `avg_days_up_user`, we first have to calculate the number of days every item has been put up. This can easily be done with pandas's `dt` API.

# In[ ]:


all_periods['days_up'] = all_periods['date_to'].dt.dayofyear - all_periods['date_from'].dt.dayofyear


# Because we want the sum of days one item has been put up for sale, we will group by `item_id` and sum the `days_up` column. We will also count the number of items in an item ID group for our second feature, `avg_times_up_user`.

# In[ ]:


gp = all_periods.groupby(['item_id'])[['days_up']]

gp_df = pd.DataFrame()
gp_df['days_up_sum'] = gp.sum()['days_up']
gp_df['times_put_up'] = gp.count()['days_up']
gp_df.reset_index(inplace=True)
gp_df.rename(index=str, columns={'index': 'item_id'})

gp_df.head()


# At this point, we have 2 scalars associated with every `item_id` appearing in train and test periods. We can now savely drop the duplicate item IDs in the `all_periods` dataframe and merge the features back into `all_periods`.

# In[ ]:


all_periods.drop_duplicates(['item_id'], inplace=True)
all_periods = all_periods.merge(gp_df, on='item_id', how='left')
all_periods.head()


# In[ ]:


del gp
del gp_df
gc.collect()


# We have an interesting but kind of useless feature now. As seen in the second venn diagram, there is no overlap at all between `train_active` (and with that `train_periods`) and `train` concerning *item* IDs. For the feature to become useful, we somehow have to associate an item ID with a user ID.
# 
# So the next step is to merge `all_samples` into `all_periods`. This will get us a user ID for every item ID in the periods dataframes. Now there is an overlap!

# In[ ]:


all_periods = all_periods.merge(all_samples, on='item_id', how='left')
all_periods.head()


# The next problem is that there are multiple features for a user if that user has put up more than one item that appears in `train_active` / `test_active`. We will have to somehow reduce this to one feature.
# 
# Here they are averaged, but you can try something else like median or modus too.

# In[ ]:


gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index()     .rename(index=str, columns={
        'days_up_sum': 'avg_days_up_user',
        'times_put_up': 'avg_times_up_user'
    })
gp.head()


# For our last feature, `n_user_items`, we just group by user ID and count the number of items. We have to be careful to use `all_samples` instead of `all_periods` here because the latter does not contain the `train.csv` and `test.csv` samples.

# In[ ]:


n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index()     .rename(index=str, columns={
        'item_id': 'n_user_items'
    })
gp = gp.merge(n_user_items, on='user_id', how='outer')

gp.head()


# I'll save the features to a CSV so you don't have to run the entire code yourself if you want to try them in your model.

# In[ ]:


gp.to_csv('aggregated_features.csv', index=False)


# In[ ]:


del all_samples
del all_periods
del train
del test

gc.collect()


# # Training a LightGBM model

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.merge(gp, on='user_id', how='left')
test = test.merge(gp, on='user_id', how='left')

agg_cols = list(gp.columns)[1:]

del gp
gc.collect()

train.head()


# One more thing about the approach that I haven't mentioned yet is that we will have quite some NaN values because not every ID in `train` and `test` occurs in `train_active` and `test_active`. Let's check how big that problem is.

# In[ ]:


train[agg_cols].isna().any(axis=1).sum() / len(train) * 100


# In[ ]:


test[agg_cols].isna().any(axis=1).sum() / len(test) * 100


# We have missing features for 22.41% of train and 24.35% of test data. That's not perfect but certainly acceptable. Onto some more basic feature engineering with ideas from [a great kernel](https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2241?scriptVersionId=3603709).

# In[ ]:


count = lambda l1,l2: sum([1 for x in l1 if x in l2])


for df in [train, test]:
    df['description'].fillna('unknowndescription', inplace=True)
    df['title'].fillna('unknowntitle', inplace=True)

    df['weekday'] = pd.to_datetime(df['activation_date']).dt.day
    
    for col in ['description', 'title']:
        df['num_words_' + col] = df[col].apply(lambda comment: len(comment.split()))
        df['num_unique_words_' + col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))

    df['words_vs_unique_title'] = df['num_unique_words_title'] / df['num_words_title'] * 100
    df['words_vs_unique_description'] = df['num_unique_words_description'] / df['num_words_description'] * 100
    
    df['city'] = df['region'] + '_' + df['city']
    df['num_desc_punct'] = df['description'].apply(lambda x: count(x, set(string.punctuation)))
    
    for col in agg_cols:
        df[col].fillna(-1, inplace=True)


# In[ ]:


count_vectorizer_title = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True, min_df=25)

title_counts = count_vectorizer_title.fit_transform(train['title'].append(test['title']))

train_title_counts = title_counts[:len(train)]
test_title_counts = title_counts[len(train):]


count_vectorizer_desc = TfidfVectorizer(stop_words=stopwords.words('russian'), 
                                        lowercase=True, ngram_range=(1, 2),
                                        max_features=15000)

desc_counts = count_vectorizer_desc.fit_transform(train['description'].append(test['description']))

train_desc_counts = desc_counts[:len(train)]
test_desc_counts = desc_counts[len(train):]

train_title_counts.shape, train_desc_counts.shape


# In[ ]:


target = 'deal_probability'
predictors = [
    'num_desc_punct', 
    'words_vs_unique_description', 'num_unique_words_description', 'num_unique_words_title', 'num_words_description', 'num_words_title',
    'avg_times_up_user', 'avg_days_up_user', 'n_user_items', 
    'price', 'item_seq_number'
]
categorical = [
    'image_top_1', 'param_1', 'param_2', 'param_3', 
    'city', 'region', 'category_name', 'parent_category_name', 'user_type'
]

predictors = predictors + categorical


# In[ ]:


for feature in categorical:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    encoder.fit(train[feature].append(test[feature]).astype(str))
    
    train[feature] = encoder.transform(train[feature].astype(str))
    test[feature] = encoder.transform(test[feature].astype(str))


# After some hyperparameter definitions and creating train / valid / test matrices, we can finally train the model. Let's see if the aggregated features helped.
# 
# *Note: For further feature engineering, I would recommend restricting the max_depth further (5 worked well for me) and increasing the learning rate (to ~ 0.1) so you don't have to wait forever for the training to finish.*

# In[ ]:


rounds = 16000
early_stop_rounds = 500
params = {
    'objective' : 'regression',
    'metric' : 'rmse',
    'num_leaves' : 32,
    'max_depth': 15,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.6,
    'verbosity' : -1
}

feature_names = np.hstack([
    count_vectorizer_desc.get_feature_names(),
    count_vectorizer_title.get_feature_names(),
    predictors
])
print('Number of features:', len(feature_names))


# In[ ]:


train_index, valid_index = train_test_split(np.arange(len(train)), test_size=0.1, random_state=42)

x_train = scipy.sparse.hstack([
        train_desc_counts[train_index],
        train_title_counts[train_index],
        train.loc[train_index, predictors]
], format='csr')
y_train = train.loc[train_index, target]

x_valid = scipy.sparse.hstack([
    train_desc_counts[valid_index],
    train_title_counts[valid_index],
    train.loc[valid_index, predictors]
], format='csr')
y_valid = train.loc[valid_index, target]

x_test = scipy.sparse.hstack([
    test_desc_counts,
    test_title_counts,
    test.loc[:, predictors]
], format='csr')

dtrain = lgb.Dataset(x_train, label=y_train,
                     feature_name=list(feature_names), 
                     categorical_feature=categorical)
dvalid = lgb.Dataset(x_valid, label=y_valid,
                     feature_name=list(feature_names), 
                     categorical_feature=categorical)


# In[ ]:


evals_result = {}
model = lgb.train(params, dtrain, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train', 'valid'],
                  num_boost_round=rounds, 
                  early_stopping_rounds=early_stop_rounds, 
                  verbose_eval=500)


# That looks good. But the model is kind of a black box. It is a good idea to plot the feature importances for our model now.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 14))
lgb.plot_importance(model, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")


# `avg_days_up`, `avg_times_up_user` and `n_user_items` are our most important engineered features! Looks like we were successful. Now we just have to predict the test matrix and submit!

# In[ ]:


subm = pd.read_csv('../input/sample_submission.csv')
subm['deal_probability'] = np.clip(model.predict(x_test), 0, 1)
subm.to_csv('submission.csv', index=False)


# I'll end this kernel with some ideas to improve it:
# - Use K-Fold cross validation.
# - Try other methods than mean for reducing the aggregated features to one per user (e. g. modus or median).
# - Try other gradient boosting libraries like CatBoost or XGBoost.
# - Add a temporal dimension to engineered features (e. g. # of items a user put up for sale *per day*).
# - Add more advanced text features like pretrained word embeddings.
# - Add image features. At the moment we completely ignore images! (as discussed [here](https://www.kaggle.com/c/avito-demand-prediction/discussion/56678), two promising approaches could be [NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424) and [Multimedia Features for Click Prediction](https://storage.googleapis.com/kaggle-forum-message-attachments/328059/9411/dimitri-clickadvert.pdf)).
# - Normalize text before creating the Tf-Idf matrix (e. g. using [stemming](http://www.nltk.org/howto/stem.html)).
# - ~~Learn russian and do in-depth text analysis.~~
# 
# Thanks for reading and have fun in this competition!
