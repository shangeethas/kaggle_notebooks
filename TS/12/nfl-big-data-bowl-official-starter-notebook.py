#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl 2020 Official Starter Notebook
# ## Introduction
# In this competition you will predict how many yards a team will gain on a rushing play in an NFL regular season game.  You will loop through a series of rushing plays; for each play, you'll receive the position, velocity, orientation, and more for all 22 players on the field at the moment of handing the ball off to the rusher, along with many other features such as teams, stadium, weather conditions, etc.  You'll use this information to predict how many yards the team will gain on the play as a [cumulative probability distribution](https://en.wikipedia.org/wiki/Cumulative_distribution_function).  Once you make that prediction, you can move on to the next rushing play.
# 
# This competition is different from most Kaggle Competitions in that:
# * You can only submit from Kaggle Notebooks, and you may not use other data sources, GPU, or internet access.
# * This is a **two-stage competition**.  In Stage One you can edit your Notebooks and improve your model, where Public Leaderboard scores are based on your predictions on rushing plays from the first few weeks of the 2019 regular season.  At the beginning of Stage Two, your Notebooks are locked, and we will re-run your Notebooks over the following several weeks, scoring them based on their predictions relative to live data as the 2019 regular season unfolds.
# * You must use our custom **`kaggle.competitions.nflrush`** Python module.  The purpose of this module is to control the flow of information to ensure that you are not using future data to make predictions for the current rushing play.  If you do not use this module properly, your code may fail when it is re-run in Stage Two.
# 
# ## In this Starter Notebook, we'll show how to use the **`nflrush`** module to get the training data, get test features and make predictions, and write the submission file.
# ## TL;DR: End-to-End Usage Example
# ```
# from kaggle.competitions import nflrush
# env = nflrush.make_env()
# 
# # Training data is in the competition dataset as usual
# train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# train_my_model(train_df)
# 
# for (test_df, sample_prediction_df) in env.iter_test():
#   predictions_df = make_my_predictions(test_df, sample_prediction_df)
#   env.predict(predictions_df)
#   
# env.write_submission_file()
# ```
# Note that `train_my_model` and `make_my_predictions` are functions you need to write for the above example to work.

# ## In-depth Introduction
# First let's import the module and create an environment.

# In[1]:


from kaggle.competitions import nflrush
import pandas as pd

# You can only call make_env() once, so don't lose it!
env = nflrush.make_env()


# ### Training data is in the competition dataset as usual

# In[2]:


train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train_df


# ## `iter_test` function
# 
# Generator which loops through each rushing play in the test set and provides the observations at `TimeHandoff` just like the training set.  Once you call **`predict`** to make your yardage prediction, you can continue on to the next play.
# 
# Yields:
# * While there are more rushing play(s) and `predict` was called successfully since the last yield, yields a tuple of:
#     * `test_df`: DataFrame with player and game observations for the next rushing play.
#     * `sample_prediction_df`: DataFrame with an example yardage prediction.  Intended to be filled in and passed back to the `predict` function.
# * If `predict` has not been called successfully since the last yield, prints an error and yields `None`.

# In[3]:


# You can only iterate through a result from `env.iter_test()` once
# so be careful not to lose it once you start iterating.
iter_test = env.iter_test()


# Let's get the data for the first test play and check it out.

# In[4]:


(test_df, sample_prediction_df) = next(iter_test)
test_df


# Note how our predictions need to take the form of a [cumulative probability distribution](https://en.wikipedia.org/wiki/Cumulative_distribution_function) over the range of possible yardages.  Each column indicates the probability that the team gains <= that many yards on the play.  For example, the value for `Yards-2` should be your prediction for the probability that the team gains at most -2 yards, and `Yard10` is the probability that the team gains at most 10 yards.  Theoretically, `Yards99` should equal `1.0`.

# In[5]:


sample_prediction_df


# The sample prediction here just predicts that exactly 3 yards were gained on the play.

# In[6]:


sample_prediction_df[sample_prediction_df.columns[98:108]]


# Note that we'll get an error if we try to continue on to the next test play without making our predictions for the current play.

# In[7]:


next(iter_test)


# ### **`predict`** function
# Stores your predictions for the current rushing play.  Expects the same format as you saw in `sample_prediction_df` returned from the `iter_test` generator.
# 
# Args:
# * `predictions_df`: DataFrame which must have the same format as `sample_prediction_df`.
# 
# This function will raise an Exception if not called after a successful iteration of the `iter_test` generator.

# Let's make a dummy prediction using the sample provided by `iter_test`.

# In[8]:


env.predict(sample_prediction_df)


# ## Main Loop
# Let's loop through all the remaining plays in the test set generator and make the default prediction for each.  The `iter_test` generator will simply stop returning values once you've reached the end.
# 
# When writing your own Notebooks, be sure to write robust code that makes as few assumptions about the `iter_test`/`predict` loop as possible.  For example, the number of iterations will change during Stage Two of the competition, since you'll be tested on rushing plays which hadn't even occurred when you wrote your code.  There may also be players in the updated test set who never appeared in any Stage One training or test data.
# 
# You may assume that the structure of `sample_prediction_df` will not change in this competition.

# In[9]:


for (test_df, sample_prediction_df) in iter_test:
    env.predict(sample_prediction_df)


# ## **`write_submission_file`** function
# 
# Writes your predictions to a CSV file (`submission.csv`) in the Notebook's output directory.
# 
# **You must call this function and not generate your own `submission.csv` file manually.**
# 
# Can only be called once you've completed the entire `iter_test`/`predict` loop.

# In[10]:


env.write_submission_file()


# In[11]:


# We've got a submission file!
import os
print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])


# As indicated by the helper message, calling `write_submission_file` on its own does **not** make a submission to the competition.  It merely tells the module to write the `submission.csv` file as part of the Notebook's output.  To make a submission to the competition, you'll have to **Commit** your Notebook and find the generated `submission.csv` file in that Notebook Version's Output tab (note this is _outside_ of the Notebook Editor), then click "Submit to Competition".  When we re-run your Notebook during Stage Two, we will run the Notebook Version(s) (generated when you hit "Commit") linked to your chosen Submission(s).

# ## Restart the Notebook to run your code again
# In order to combat cheating, you are only allowed to call `make_env` or iterate through `iter_test` once per Notebook run.  However, while you're iterating on your model it's reasonable to try something out, change the model a bit, and try it again.  Unfortunately, if you try to simply re-run the code, or even refresh the browser page, you'll still be running on the same Notebook execution session you had been running before, and the `nflrush` module will still throw errors.  To get around this, you need to explicitly restart your Notebook execution session, which you can do by **clicking "Run"->"Restart Session"** in the Notebook Editor's menu bar at the top.
