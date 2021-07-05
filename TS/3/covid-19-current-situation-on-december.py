#!/usr/bin/env python
# coding: utf-8

# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F518134%2F485aa04e87e4e45c91815101784c6d95%2Fcorona-4930541_1280.jpg?generation=1585438527494582&alt=media)
# 
# # COVID-19: current situation on December
# 
# The kernel is inspired by the great EDA kernel [COVID-19: Digging a Bit Deeper](https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper) by @abhinand05 in week1.<br/>
# Please upvote his kernel as well :)<br/>
# I will write an EDA including **the recent updates, recovering country analysis & sigmoid fitting convergence date estimation**.
# 
# [Note] It seems JHU has changed the data format, and stopped providing recovered cases. So I could not analyze recovered cases.
# 
# `plotly` visualization is heavy used in this kernel so that we can **interactively** see the figure, map etc.<br/>
# As a side effect, it might take a little bit more time to load the kernel, wait a minute please.
# 
# ### Version History
# 
# The data is updated daily-basis. **I will try to update the kernel DAILY basis too, so that you can refer latest information.**<br/>
# Below are the version history to see the information until specified date.<br/>
# 
# <details>
#     <summary>Version History (Expand by clicking here)</summary><br/>
#  - Version 15: Added **Daily NEW confirmed cases** analysis & **Asia** region EDA.
#  - [Version 18](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-march?scriptVersionId=31151381): Shows figure as of 2020/3/28.
#  - Version 19: Added <span style="color:red">sigmoid fitting to estimate when the coronavirus converge in each country</span>, jump to [When will it converge? - Estimation by sigmoid fitting](#id_converge).
#  - [Version 21](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31276251): Shows figure as of 2020/3/31.
#  - [Version 23](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31329249): Shows figure as of 2020/4/1, moved to use week3 competition data.
#  - [Version 29](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31384138): 2020/4/2
#  - [Version 30](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31436722): 2020/4/3
#  - [Version 31](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31489960): 2020/4/4
#  - [Version 32](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31547500): 2020/4/5
#  - [Version 33](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31604729): 2020/4/6
#  - [Version 35](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31670710): 2020/4/7, **Moved to week4 dataset**
#  - [Version 36](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31723129): 2020/4/8
#  - [Version 38](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31782521): 2020/4/9
#  - [Version 39](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31839132): 2020/4/10
#  - [Version 40](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31898610): 2020/4/11
#  - [Version 41](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=31957894): 2020/4/12
#  - [Version 42](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32023899): 2020/4/13
#  - [Version 43](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32082933): 2020/4/14
#  - [Version 49](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32266557): 2020/4/17, **<span style="color:red">Moved to use data from <a href="https://github.com/CSSEGISandData/COVID-19">2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository</a><span style="color:red"> by Johns Hopkins CSSE**. Internet connection is required, since it will download latest data from the repository.
#  - [Version 50](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32321582): 2020/4/18, removed unused data.
#  - [Version 51](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32382070): 2020/4/19
#  - [Version 52](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32623228): 2020/4/23
#  - [Version 53](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32751754): 2020/4/25
#  - [Version 54](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32821067): 2020/4/26
#  - [Version 55](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=32966399): 2020/4/28
#  - [Version 56](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april?scriptVersionId=33111789): 2020/4/30
#  - [Version 57](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33178751): 2020/5/1    
#  - [Version 58](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33246349): 2020/5/2   
#  - [Version 59](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33316049): 2020/5/3
#  - [Version 60](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33388532): 2020/5/4
#  - [Version 61](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33456431): 2020/5/5
#  - [Version 62](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33523879): 2020/5/6
#  - [Version 63](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33592887): 2020/5/7
#  - [Version 64](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33665835): 2020/5/8
#  - [Version 65](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33728654): 2020/5/9
#  - [Version 66](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33801408): 2020/5/10
#  - [Version 69](https://www.kaggle.com/corochann/covid-19-current-situation-on-may-daily-update?scriptVersionId=33943023): 2020/5/12
#  - [Version 70](https://www.kaggle.com/corochann/covid-19-current-situation-on-may?scriptVersionId=34085641): 2020/5/14
#  - [Version 71](https://www.kaggle.com/corochann/covid-19-current-situation-on-may?scriptVersionId=34273754): 2020/5/17
#  - [Version 72](https://www.kaggle.com/corochann/covid-19-current-situation-on-may?scriptVersionId=34415107): 2020/5/19
#  - [Version 73](https://www.kaggle.com/corochann/covid-19-current-situation-on-may?scriptVersionId=34661387): 2020/5/22, Added growth factor
#  - [Version 75](https://www.kaggle.com/corochann/covid-19-current-situation-on-may?scriptVersionId=34870894): 2020/5/25
#  - [Version 76](https://www.kaggle.com/corochann/covid-19-current-situation-on-may?scriptVersionId=35075854): 2020/5/28
#  - [Version 77](https://www.kaggle.com/corochann/covid-19-current-situation-on-june?scriptVersionId=35340282): 2020/6/1
#  - [Version 78](https://www.kaggle.com/corochann/covid-19-current-situation-on-june?scriptVersionId=35475709): 2020/6/3
#  - [Version 79](https://www.kaggle.com/corochann/covid-19-current-situation-on-june?scriptVersionId=35767313): 2020/6/7
#  - [Version 80](https://www.kaggle.com/corochann/covid-19-current-situation-on-june?scriptVersionId=36013977): 2020/6/10
#  - [Version 81](https://www.kaggle.com/corochann/covid-19-current-situation-on-june?scriptVersionId=37235122): 2020/6/21
#  - [Version 82](https://www.kaggle.com/corochann/covid-19-current-situation-on-june?scriptVersionId=37739722): 2020/6/28
#  - [Version 83](https://www.kaggle.com/corochann/covid-19-current-situation-on-july?scriptVersionId=38064445): 2020/7/2
#  - [Version 84](https://www.kaggle.com/corochann/covid-19-current-situation-on-july?scriptVersionId=38768566): 2020/7/13
#  - [Version 85](https://www.kaggle.com/corochann/covid-19-current-situation-on-july?scriptVersionId=39279123): 2020/7/21
#  - [Version 86](https://www.kaggle.com/corochann/covid-19-current-situation-on-august?scriptVersionId=39996289): 2020/8/1
#  - [Version 87](https://www.kaggle.com/corochann/covid-19-current-situation-on-august?scriptVersionId=40873956): 2020/8/15
#  - [Version 88](https://www.kaggle.com/corochann/covid-19-current-situation-on-august?scriptVersionId=41227754): 2020/8/22
#  - [Version 89](https://www.kaggle.com/corochann/covid-19-current-situation-on-september?scriptVersionId=41936057): 2020/9/2
#  - [Version 90](https://www.kaggle.com/corochann/covid-19-current-situation-on-september?scriptVersionId=43173201): 2020/9/20
#  - [Version 91](https://www.kaggle.com/corochann/covid-19-current-situation-on-october?scriptVersionId=44115812): 2020/10/4
#  - [Version 93](https://www.kaggle.com/corochann/covid-19-current-situation-on-october?scriptVersionId=45297457): 2020/10/21
#  - [Version 94](https://www.kaggle.com/corochann/covid-19-current-situation-on-october?scriptVersionId=46904410): 2020/11/13    
#     
# </details>
# 
#  - Latest Version: 2020/12/2

# ## Table of Contents
# 
# 
# **[Load Data](#id_load)**<br/>
# **[Worldwide trend](#id_ww)**<br/>
# **[Country-wise growth](#id_country)**<br/>
# **[Going into province](#id_province)**<br/>
# **[Zoom up to US: what is happening in US now??](#id_province)**<br/>
# **[Europe](#id_europe)**<br/>
# **[Asia](#id_asia)**<br/>
# **[Which country is recovering now?](#id_recover)**<br/>
# **[When will it converge? - Estimation by sigmoid fitting](#id_converge)**<br/>
# **[Further reading](#id_ref)**<br/>

# In[1]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# <a id="id_load"></a>
# # Load Data
# 
# Download latest data from Johns Hopkins University github repository: [https://github.com/CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19)

# In[2]:


# Input data files are available in the "../input/" directory.
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     filenames.sort()
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# Referenced: https://www.kaggle.com/benhamner/covid-19-forecasting-challenges-week-2-data-prep

# In[3]:


get_ipython().run_cell_magic('time', '', "import requests\n\nfor filename in ['time_series_covid19_confirmed_global.csv',\n                 'time_series_covid19_deaths_global.csv',\n                 'time_series_covid19_recovered_global.csv',\n                 'time_series_covid19_confirmed_US.csv',\n                 'time_series_covid19_deaths_US.csv']:\n    print(f'Downloading {filename}')\n    url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/{filename}'\n    myfile = requests.get(url)\n    open(filename, 'wb').write(myfile.content)")


# In[4]:


from datetime import datetime

def _convert_date_str(df):
    try:
        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]
    except:
        print('_convert_date_str failed with %y, try %Y')
        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%Y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]


confirmed_global_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
_convert_date_str(confirmed_global_df)

deaths_global_df = pd.read_csv('time_series_covid19_deaths_global.csv')
_convert_date_str(deaths_global_df)

recovered_global_df = pd.read_csv('time_series_covid19_recovered_global.csv')
_convert_date_str(recovered_global_df)


# In[5]:


# Filter out problematic data points (The West Bank and Gaza had a negative value, cruise ships were associated with Canada, etc.)
removed_states = "Recovered|Grand Princess|Diamond Princess"
removed_countries = "US|The West Bank and Gaza"

confirmed_global_df.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
deaths_global_df.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
recovered_global_df.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

confirmed_global_df = confirmed_global_df[~confirmed_global_df["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
deaths_global_df    = deaths_global_df[~deaths_global_df["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
recovered_global_df = recovered_global_df[~recovered_global_df["Province_State"].replace(np.nan, "nan").str.match(removed_states)]

confirmed_global_df = confirmed_global_df[~confirmed_global_df["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]
deaths_global_df    = deaths_global_df[~deaths_global_df["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]
recovered_global_df = recovered_global_df[~recovered_global_df["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]


# In[6]:


confirmed_global_melt_df = confirmed_global_df.melt(
    id_vars=['Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='ConfirmedCases')
deaths_global_melt_df = deaths_global_df.melt(
    id_vars=['Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='Deaths')
recovered_global_melt_df = deaths_global_df.melt(
    id_vars=['Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='Recovered')


# In[7]:


train = confirmed_global_melt_df.merge(deaths_global_melt_df, on=['Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])
train = train.merge(recovered_global_melt_df, on=['Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])


# In[8]:


# --- US ---
confirmed_us_df = pd.read_csv('time_series_covid19_confirmed_US.csv')
deaths_us_df = pd.read_csv('time_series_covid19_deaths_US.csv')

confirmed_us_df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Combined_Key'], inplace=True, axis=1)
deaths_us_df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Combined_Key', 'Population'], inplace=True, axis=1)

confirmed_us_df.rename({'Long_': 'Long'}, axis=1, inplace=True)
deaths_us_df.rename({'Long_': 'Long'}, axis=1, inplace=True)

_convert_date_str(confirmed_us_df)
_convert_date_str(deaths_us_df)

# clean
confirmed_us_df = confirmed_us_df[~confirmed_us_df.Province_State.str.match("Diamond Princess|Grand Princess|Recovered|Northern Mariana Islands|American Samoa")]
deaths_us_df = deaths_us_df[~deaths_us_df.Province_State.str.match("Diamond Princess|Grand Princess|Recovered|Northern Mariana Islands|American Samoa")]

# --- Aggregate by province state ---
#confirmed_us_df.groupby(['Country_Region', 'Province_State'])
confirmed_us_df = confirmed_us_df.groupby(['Country_Region', 'Province_State']).sum().reset_index()
deaths_us_df = deaths_us_df.groupby(['Country_Region', 'Province_State']).sum().reset_index()

# remove lat, long.
confirmed_us_df.drop(['Lat', 'Long'], inplace=True, axis=1)
deaths_us_df.drop(['Lat', 'Long'], inplace=True, axis=1)

confirmed_us_melt_df = confirmed_us_df.melt(
    id_vars=['Country_Region', 'Province_State'], value_vars=confirmed_us_df.columns[2:], var_name='Date', value_name='ConfirmedCases')
deaths_us_melt_df = deaths_us_df.melt(
    id_vars=['Country_Region', 'Province_State'], value_vars=deaths_us_df.columns[2:], var_name='Date', value_name='Deaths')

train_us = confirmed_us_melt_df.merge(deaths_us_melt_df, on=['Country_Region', 'Province_State', 'Date'])


# In[9]:


train = pd.concat([train, train_us], axis=0, sort=False)

train_us.rename({'Country_Region': 'country', 'Province_State': 'province', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Deaths': 'fatalities'}, axis=1, inplace=True)
train_us['country_province'] = train_us['country'].fillna('') + '/' + train_us['province'].fillna('')


# In[10]:


train


# In[11]:


get_ipython().run_cell_magic('time', '', "datadir = Path('/kaggle/input/covid19-global-forecasting-week-4')\n\n# Read in the data CSV files\n#train = pd.read_csv(datadir/'train.csv')\n#test = pd.read_csv(datadir/'test.csv')\n#submission = pd.read_csv(datadir/'submission.csv')")


# In[12]:


train


# In[13]:


#test


# In[14]:


#submission


# In[15]:


train.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Deaths': 'fatalities', 'Recovered': 'recovered'}, axis=1, inplace=True)
train['country_province'] = train['country'].fillna('') + '/' + train['province'].fillna('')

# test.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Fatalities': 'fatalities'}, axis=1, inplace=True)
# test['country_province'] = test['country'].fillna('') + '/' + test['province'].fillna('')


# <a id="id_ww"></a>
# # Worldwide trend

# In[16]:


ww_df = train.groupby('date')[['confirmed', 'fatalities']].sum().reset_index()
ww_df['new_case'] = ww_df['confirmed'] - ww_df['confirmed'].shift(1)
ww_df['growth_factor'] = ww_df['new_case'] / ww_df['new_case'].shift(1)
ww_df.tail()


# In[17]:


ww_melt_df = pd.melt(ww_df, id_vars=['date'], value_vars=['confirmed', 'fatalities', 'new_case'])
ww_melt_df


# When we see the confirmed cases in world wide, it just look like exponential growth curve. The number is increasing very rapidly especially recently. **the number almost doubled in last 1 week**...
# 
# <span style="color:red"><b>Confirmed cases reached 1M people, and 52K people already died on April 2</b></span>.<br/>
# <span style="color:red"><b>Confirmed cases reached 10M people, and 500K people already died on June 28</b></span>.<br/>
# <span style="color:red"><b>Confirmed cases reached 20M people, and 750K people already died on Aug 13</b></span>.<br/>
# <span style="color:red"><b>Confirmed cases reached 50M people, and 1.25M people already died on Nov 8</b></span>.
# 
# <span style="color:red"><b>Daily new confirmed cases started not increasing from April 4. After that flat trend continues so far.</b></span>.

# In[18]:


fig = px.line(ww_melt_df, x="date", y="value", color='variable', 
              title="Worldwide Confirmed/Death Cases Over Time")
fig.show()


# Moreover, when we check the growth in log-scale below figure, we can see that the speed of confirmed cases growth rate **slightly increases** when compared with the beginning of March and end of March.<br/>
# In spite of the Lockdown policy in Europe or US, the number was increasing rapidly around the beginning of April.

# In[19]:


fig = px.line(ww_melt_df, x="date", y="value", color='variable',
              title="Worldwide Confirmed/Death Cases Over Time (Log scale)",
             log_y=True)
fig.show()


# It looks like `fatalities` curve is just shifted the `confirmed` curve to below in log-scale, which means mortality rate is almost constant.
# 
# Is it true? Let's see mortality rate in detail.<br/>
# We see that mortality rate is kept almost 3%, however it is slightly **increasing gradually to go over 7%** at the end of April.<br/>
# 
# Why? I will show you later that Europe & US has more seriously infected by Coronavirus recently, and mortality rate is high in these regions.<br/>
# It might be because when too many people get coronavirus, the country cannot provide enough medical treatment.
# 
# **It seems mortality rate is start decreasing on May!!** Is it because enough medical care is started to reaching out to everyone or the number of inspection increased and many "hidden" confirmed cases are detected now?

# In[20]:


ww_df['mortality'] = ww_df['fatalities'] / ww_df['confirmed']

fig = px.line(ww_df, x="date", y="mortality", 
              title="Worldwide Mortality Rate Over Time")
fig.show()


# Let's check growth factor:
# 
# > Growth factor is the factor by which a quantity multiplies itself over time. The formula used is every day's new cases / new cases on the previous day.
# 
# When this number is more than 1. the number of confirmed cases will be increasing, and when it keeps below 1. the number of confirmed cases will decrease.
# So it is important to check growth factor is kept below 1. or not.
# 
# In worldwide, growth factor is around 1. from May, it means that confirmed cases does not increase, but not decreasing so far...
# 
# Reference:
#  - [Covid-19 Predictions, Growth Factor, and Calculus](https://www.kaggle.com/dferhadi/covid-19-predictions-growth-factor-and-calculus)
#  - [Coronavirus Cases](https://www.worldometers.info/coronavirus/coronavirus-cases/#cases-growth-factor)
#  - [The one COVID-19 number to watch](https://www.abc.net.au/news/2020-04-10/coronavirus-data-australia-growth-factor-covid-19/12132478?nw=0)

# In[21]:


fig = px.line(ww_df, x="date", y="growth_factor", 
              title="Worldwide Growth Factor Over Time")
fig.add_trace(go.Scatter(x=[ww_df['date'].min(), ww_df['date'].max()], y=[1., 1.], name='Growth factor=1.', line=dict(dash='dash', color=('rgb(255, 0, 0)'))))
fig.update_yaxes(range=[0., 5.])
fig.show()


# <a id="id_country"></a>
# # Country-wise growth

# In[22]:


country_df = train.groupby(['date', 'country'])[['confirmed', 'fatalities']].sum().reset_index()
country_df.tail()


# What kind of country is in the dataset? How's the distribution of number of confirmed cases by country?

# In[23]:


countries = country_df['country'].unique()
print(f'{len(countries)} countries are in dataset:\n{countries}')


# In[24]:


target_date = country_df['date'].max()

print('Date: ', target_date)
for i in [1, 10, 100, 1000, 10000]:
    n_countries = len(country_df.query('(date == @target_date) & confirmed > @i'))
    print(f'{n_countries} countries have more than {i} confirmed cases')


# In[25]:


ax = sns.distplot(np.log10(country_df.query('date == "2020-03-27"')['confirmed'] + 1))
ax.set_xlim([0, 6])
ax.set_xticks(np.arange(7))
_ = ax.set_xticklabels(['0', '10', '100', '1k', '10k', '100k'])


# It is difficult to see all countries so let's check top countries.

# In[26]:


top_country_df = country_df.query('(date == @target_date) & (confirmed > 1000)').sort_values('confirmed', ascending=False)
top_country_melt_df = pd.melt(top_country_df, id_vars='country', value_vars=['confirmed', 'fatalities'])


# Now **US, Italy and Spain** has more confirmed cases than China, and we can see many Europe countries in the top.
# 
# Korea also appears in relatively top despite of its population, this is because Korea execcutes inspection check aggressively.

# In[27]:


fig = px.bar(top_country_melt_df.iloc[::-1],
             x='value', y='country', color='variable', barmode='group',
             title=f'Confirmed Cases/Deaths on {target_date}', text='value', height=1500, orientation='h')
fig.show()


# Let's check these major country's growth by date.
# 
# As we can see, Coronavirus hit **China** at first but its trend is slowing down in March which is good news.<br/>
# Bad news is 2nd wave comes to **Europe (Italy, Spain, Germany, France, UK)** at March.<br/>
# But more sadly 3rd wave now comes to **US, whose growth rate is much much faster than China, or even Europe**. Its main spread starts from middle of March and its speed is faster than Italy. Now US seems to be in the most serious situation in terms of both total number and spread speed.<br/>
# 
# From June, **Branzil & Russia** getting increased.<br/>
# From July, **Peru, Chile in South America & India in South Asia** getting increased.
# 
# Coronavirus spreads really the "All" over the world, it seems it still need a long time until converge.
# 
# You can click country legend to show/hide each country's line plot.
# If you cannot see the line (due to plotly bug?) in the plot, you can also try clicking country legend. Line will appear.

# In[28]:


top30_countries = top_country_df.sort_values('confirmed', ascending=False).iloc[:30]['country'].unique()
top30_countries_df = country_df[country_df['country'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='date', y='confirmed', color='country',
              title=f'Confirmed Cases for top 30 country as of {target_date}')
fig.show()


# In terms of number of fatalities, Europe & US are serious situation now.<br/>
# Many countries have more fatalities than China now, including US, Italy, Spain, France, UK, Iran Belgium, Germany, Brazil, Netherlands.
# 
# **US's spread speed is the fastest, US's fatality cases become top1 on Apr 10th.**
# 
# In June, the number increases rapidly in South America, **Brazil & Mexico**.

# In[29]:


top30_countries = top_country_df.sort_values('fatalities', ascending=False).iloc[:30]['country'].unique()
top30_countries_df = country_df[country_df['country'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='date', y='fatalities', color='country',
              title=f'Fatalities for top 30 country as of {target_date}')
fig.show()


# Now let's see mortality rate by country

# In[30]:


top_country_df = country_df.query('(date == @target_date) & (confirmed > 100)')
top_country_df['mortality_rate'] = top_country_df['fatalities'] / top_country_df['confirmed']
top_country_df = top_country_df.sort_values('mortality_rate', ascending=False)


# Italy was the most serious situation, whose mortality rate is over 10% as of 2020/3/28.<br/>
# As of June, we can see trend that mortality rate is high in Europe region.<br/>
# We can also find countries from all over the world when we see top mortality rate countries.<br/>
# Iran/Iraq from Middle East, Phillipines & Indonesia from tropical areas.<br/>
# Spain, Netherlands, France, and UK form Europe etc. It shows this coronavirus is really world wide pandemic.
# 
# [UPDATE]: According to the comment by @elettra84, 10% of Italy is due to extreme outlier of Lombardy cluster. Except that mortality rate in Italy is comparable to other country. Refer [Lombardy cluster in wikipedia](https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Italy#Lombardy_cluster).

# In[31]:


fig = px.bar(top_country_df[:30].iloc[::-1],
             x='mortality_rate', y='country',
             title=f'Mortality rate HIGH: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')
fig.show()


# How about the countries whose mortality rate is low?
# Many Asian, Middle-East area is on the figure.
# 
# By investigating the difference between above & below countries, we might be able to figure out what is the cause which leads death.<br/>
# Be careful that there may be a case that these country's mortality rate is low due to these country does not report/measure fatality cases properly.

# In[32]:


fig = px.bar(top_country_df[-30:],
             x='mortality_rate', y='country',
             title=f'Mortality rate LOW: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')
fig.show()


# Let's see number of confirmed cases on map. Again we can see Europe, US, MiddleEast (Turkey, Iran) and Asia (China, Korea) are red.

# In[33]:


all_country_df = country_df.query('date == @target_date')
all_country_df['confirmed_log1p'] = np.log10(all_country_df['confirmed'] + 1)
all_country_df['fatalities_log1p'] = np.log10(all_country_df['fatalities'] + 1)
all_country_df['mortality_rate'] = all_country_df['fatalities'] / all_country_df['confirmed']


# In[34]:


fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="confirmed_log1p", 
                    hover_name="country", hover_data=["confirmed", 'fatalities', 'mortality_rate'],
                    range_color=[all_country_df['confirmed_log1p'].min(), all_country_df['confirmed_log1p'].max()], 
                    color_continuous_scale="peach", 
                    title='Countries with Confirmed Cases')

# I'd like to update colorbar to show raw values, but this does not work somehow...
# Please let me know if you know how to do this!!
trace1 = list(fig.select_traces())[0]
trace1.colorbar = go.choropleth.ColorBar(
    tickvals=[0, 1, 2, 3, 4, 5],
    ticktext=['1', '10', '100', '1000','10000', '10000'])
fig.show()


# When we see mortality rate on map, we see Europe (especaiily Italy) is high. Also we notice MiddleEast (Iran, Iraq) is high.
# 
# When we see tropical area, I wonder why Phillipines and Indonesia are high while other countries (Malaysia, Thai, Vietnam, as well as Australia) are low.
# 
# For Asian region, Korea's mortality rate is lower than China or Japan, I guess this is due to the fact that number of inspection is quite many in Korea.
# Please refer these blogs for detail:
# 
#  - [South Korea launches 'drive-thru' coronavirus testing facilities as demand soars](https://www.japantimes.co.jp/news/2020/03/01/asia-pacific/science-health-asia-pacific/south-korea-drive-thru-coronavirus/#.XoAmw4j7RPY)
#  - [Coronavirus: Why Japan tested so few people](https://asia.nikkei.com/Spotlight/Coronavirus/Coronavirus-Why-Japan-tested-so-few-people)

# In[35]:



fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="fatalities_log1p", 
                    hover_name="country", range_color=[0, 4],
                    hover_data=['confirmed', 'fatalities', 'mortality_rate'],
                    color_continuous_scale="peach", 
                    title='Countries with fatalities')
fig.show()


# Mortality rate map, seems mortality rate is especially high in Europe region, compared to US or Asia.

# In[36]:


fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="mortality_rate", 
                    hover_name="country", range_color=[0, 0.12], 
                    color_continuous_scale="peach", 
                    title='Countries with mortality rate')
fig.show()


# Why mortality rate is different among country? What kind of hint is hidden in this map? Especially mortality rate is high in Europe and US, is there some reasons?
# 
# There is one interesting hypothesis that BCG vaccination<br/>
# The below figure shows BCG vaccination policy by country: Advanced countries like Europe & US, especially Italy and US does not take BCG vaccination. We can notice this map is indeed similar to mortality rate map above. Is it just accidental?
# 
# ![](https://www.researchgate.net/profile/Alice_Zwerling/publication/50892386/figure/fig2/AS:277209752326147@1443103363144/Map-displaying-BCG-vaccination-policy-by-country-A-The-country-currently-has-universal.png)
# 
# Reference: [If I were North American/West European/Australian, I will take BCG vaccination now against the novel coronavirus pandemic.](https://www.jsatonotes.com/2020/03/if-i-were-north-americaneuropeanaustral.html)
# 
#  - [Australia's Trialing a TB Vaccine Against COVID-19, And Health Workers Get It First](https://www.sciencealert.com/australia-is-trialling-a-tb-vaccine-for-coronavirus-and-health-workers-get-it-first)
# 
# Of course this is just one hypothesis but we can notice/find some hints to tackle Coronavirus like this by carefully analyzing/comparing the data.

# The figure showing fatality growth since 10 deaths.
#  - Ref: [COVID-19 Deaths Per Capita](https://covid19dashboards.com/covid-compare-permillion/)

# In[37]:


n_countries = 20
n_start_death = 10
fatality_top_countires = top_country_df.sort_values('fatalities', ascending=False).iloc[:n_countries]['country'].values
country_df['date'] = pd.to_datetime(country_df['date'])


df_list = []
for country in fatality_top_countires:
    this_country_df = country_df.query('country == @country')
    start_date = this_country_df.query('fatalities > @n_start_death')['date'].min()
    this_country_df = this_country_df.query('date >= @start_date')
    this_country_df['date_since'] = this_country_df['date'] - start_date
    this_country_df['fatalities_log1p'] = np.log10(this_country_df['fatalities'] + 1)
    this_country_df['fatalities_log1p'] -= this_country_df['fatalities_log1p'].values[0]
    df_list.append(this_country_df)

tmpdf = pd.concat(df_list)
tmpdf['date_since_days'] = tmpdf['date_since'] / pd.Timedelta('1 days')


# In[38]:


fig = px.line(tmpdf,
              x='date_since_days', y='fatalities_log1p', color='country',
              title=f'Fatalities by country since 10 deaths, as of {target_date}')
fig.add_trace(go.Scatter(x=[0, 28], y=[0, 4], name='Double by 7 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[0, 56], y=[0, 4], name='Double by 14 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[0, 84], y=[0, 4], name='Double by 21 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.show()


# Sudden increase at China on the days 85 may be because updated reporting in Wuhan:
# 
#  - [Coronavirus: China outbreak city Wuhan raises death toll by 50%](https://www.bbc.com/news/world-asia-china-52321529)

# ## Daily NEW confirmed cases trend
# 
# How about **DAILY new cases** trend?<br/>
# We find from below figure:
#  - China has finished its peak at Feb 14, new confirmed cases are surpressed now.
#  - Europe&US spread starts on mid of March, after China slows down.
#  - I feel effect of lock down policy in Europe (Italy, Spain, Germany, France) now comes on the figure,
#    the number of new cases are not so increasing rapidly at the end of March.
#  - Current US new confirmed cases are the worst speed, recording worst speed at more than 30k people/day at peak. <b>Daily new confirmed cases start to decrease from April 4 or April 10, I hope this trend will continue.</b>.
#    - <span style="color:red"><b>After that we can see a weekly trend that the confirmed cases becomes small on Monday. I think this is because people don't (or cannot) get medical care on Sunday so its reporting number is low on Sunday or Monday</b></span>
#  - From August, India is the top. U.S & Brazil's numbers are start decreasing.

# In[39]:


country_df['prev_confirmed'] = country_df.groupby('country')['confirmed'].shift(1)
country_df['new_case'] = country_df['confirmed'] - country_df['prev_confirmed']
country_df['new_case'].fillna(0, inplace=True)
top30_country_df = country_df[country_df['country'].isin(top30_countries)]

fig = px.line(top30_country_df,
              x='date', y='new_case', color='country',
              title=f'DAILY NEW Confirmed cases by country')
fig.show()


# In[40]:


country_df['prev_new_case'] = country_df.groupby('country')['new_case'].shift(1)
country_df['growth_factor'] = country_df['new_case'] / country_df['prev_new_case']
country_df['growth_factor'].fillna(0, inplace=True)
top30_country_df = country_df[country_df['country'].isin(top30_countries)]

fig = px.line(top30_country_df,
              x='date', y='growth_factor', color='country',
              title=f'Growth factor by country')
fig.add_trace(go.Scatter(x=[ww_df['date'].min(), ww_df['date'].max()], y=[1., 1.], name='Growth factor=1.', line=dict(dash='dash', color=('rgb(255, 0, 0)'))))
fig.update_yaxes(range=[0., 5.])
fig.show()


# ## Geographical animation: spready by date
# 
# You can see animation how confirmed cases spread over time, you can see trend moving to China -> Europe -> US.

# In[41]:


country_df['date'] = country_df['date'].apply(str)
country_df['confirmed_log1p'] = np.log1p(country_df['confirmed'])
country_df['fatalities_log1p'] = np.log1p(country_df['fatalities'])

fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="confirmed", size='confirmed', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['confirmed'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Confirmed cases spread Over Time', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# You can see animation how confirmed cases spread over time, you can see trend moving to China -> Europe -> US. But Europe is worse than US for number of fatalities now.

# In[42]:


fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="fatalities", size='fatalities', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['fatalities'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Fatalities growth Over Time', color_continuous_scale="portland")
fig.show()


# New cases trend: it looks like China is almost converged now.

# In[43]:


country_df.loc[country_df['new_case'] < 0, 'new_case'] = 0.
fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="new_case", size='new_case', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['new_case'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Daily NEW cases over Time', color_continuous_scale="portland")
fig.show()


# <a id="id_province"></a>
# # Going into province

# How many country has precise province information?<br/>
# It seems it's 8 countries: Australia, Canada, China, Denmark, France, Netherlands, US, and UK.

# In[44]:


for country in countries:
    province = train.query('country == @country')['province'].unique()
    if len(province) > 1:       
        print(f'Country {country} has {len(province)} provinces: {province}')


# <a id="id_us"></a>
# # Zoom up to US: what is happening in US now??
# 
# As we can see, the spread is fastest in US now, at the end of March. Let's see in detail what is going on in US.

# In[45]:


usa_state_code_df = pd.read_csv('/kaggle/input/usa-state-code/usa_states2.csv')


# In[46]:


train_us


# In[47]:


# Prepare data frame only for US. 

#train_us = train.query('country == "US"')
train_us['mortality_rate'] = train_us['fatalities'] / train_us['confirmed']

# Convert province column to its 2-char code name,
state_name_to_code = dict(zip(usa_state_code_df['state_name'], usa_state_code_df['state_code']))
train_us['province_code'] = train_us['province'].map(state_name_to_code)

# Only show latest days.
train_us_latest = train_us.query('date == @target_date')


# **[Situation in April]**<br/>
# When we see inside of the US, **only New York, and its neighbor New Jersey** dominated its spread and are in serious situation around April.<br/>
# New York confirmed cases is over 50k, while other states are less than about 5k confirmed cases around April.
# 
# **[Situation in July]**<br/>
# The high speed coronavirus spread does not decrease in U.S.. Now it affected to many states, especially **California, Texas, Florida**.

# In[48]:


fig = px.choropleth(train_us_latest, locations='province_code', locationmode="USA-states",
                    color='confirmed', scope="usa", hover_data=['province', 'fatalities', 'mortality_rate'],
                    title=f'Confirmed cases in US on {target_date}')
fig.show()


# Mortality rate in New York seems not high, around 2% for now.

# In[49]:


train_us_latest.sort_values('confirmed', ascending=False)


# **[Situation in July]**<br/>
# The mortality rate is high only around New York & New Jersey, this may be because first wave hit these areas and medical care was not enough in March.
# 
# While the mortality rate is relatively low in California, Texas, Florida. These states provides proper medical care?

# In[50]:


fig = px.choropleth(train_us_latest, locations='province_code', locationmode="USA-states",
                    color='mortality_rate', scope="usa", hover_data=['province', 'fatalities', 'mortality_rate'],
                    title=f'Mortality rate in US on {target_date}')
fig.show()


# **Daily growth**: All state is US got affected from middle of March, and now **growing exponentially**.
# In New York, less than 1k people are confirmed on March 16, but more than 50k people are confirmed on March 30. **50 times explosion in 2 weeks!**
# 
# Now in July, we need to be carefully watch the situation in **California, Texas, Florida**, the number is increasing exponentially in these states now.

# In[51]:


train_us_march = train_us.query('date > "2020-03-01"')
fig = px.line(train_us_march,
              x='date', y='confirmed', color='province',
              title=f'Confirmed cases by state in US, as of {target_date}')
fig.show()


# In[52]:


train_us_march['prev_confirmed'] = train_us_march.groupby('province')['confirmed'].shift(1)
train_us_march['new_case'] = train_us_march['confirmed'] - train_us_march['prev_confirmed']
train_us_march['new_case'].fillna(0, inplace=True)

fig = px.line(train_us_march,
              x='date', y='new_case', color='province',
              title=f'DAILY NEW Confirmed cases by states in US')
fig.show()


# In[53]:


train_us_march['prev_new_case'] = train_us_march.groupby('province')['new_case'].shift(1)
train_us_march['growth_factor'] = train_us_march['new_case'] / train_us_march['prev_new_case']
train_us_march['growth_factor'].fillna(0, inplace=True)
fig = px.line(train_us_march,
              x='date', y='growth_factor', color='province',
              title=f'Growth factor by state in US')
fig.add_trace(go.Scatter(x=[train_us_march['date'].min(), train_us_march['date'].max()], y=[1., 1.],
                         name='Growth factor=1.', line=dict(dash='dash', color=('rgb(255, 0, 0)'))))
fig.update_yaxes(range=[0., 5.])
fig.show()


# <a id="id_europe"></a>
# # Europe

# In[54]:


# Ref: https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper
europe_country_list =list([
    'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
    'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',
    'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
    'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])

country_df['date'] = pd.to_datetime(country_df['date'])
train_europe = country_df[country_df['country'].isin(europe_country_list)]
#train_europe['date_str'] = pd.to_datetime(train_europe['date'])
train_europe_latest = train_europe.query('date == @target_date')


# When we look into the Europe, its Northern & Eastern areas are relatively better situation compared to Eastern & Southern areas.

# In[55]:


fig = px.choropleth(train_europe_latest, locations="country", 
                    locationmode='country names', color="confirmed", 
                    hover_name="country", range_color=[1, train_europe_latest['confirmed'].max()], 
                    color_continuous_scale='portland', 
                    title=f'European Countries with Confirmed Cases as of {target_date}', scope='europe', height=800)
fig.show()


# Especially **Italy, Spain, German, France, UK** are in more serious situation.
# 
# Number of confirmed cases rapidly increasing in **Russia now (as of May 1)**, and **Russia** is in much dangerous situation as of July...

# In[56]:


train_europe_march = train_europe.query('date >= "2020-03-01"')
fig = px.line(train_europe_march,
              x='date', y='confirmed', color='country',
              title=f'Confirmed cases by country in Europe, as of {target_date}')
fig.show()


# **UK's fatality number is growing and becomes top-1 on May 5th, followed by Italy, France and Spain.**

# In[57]:


fig = px.line(train_europe_march,
              x='date', y='fatalities', color='country',
              title=f'Fatalities by country in Europe, as of {target_date}')
fig.show()


# When we check daily new cases in Europe, we notice:
# 
#  - **UK and Russia** daily growth are more than Italy now, These countries are potentially more dangerous now.
#  - Italy new cases are not increasing since March 21, I guess due to lock-down policy is started working. That is not a bad news.
#  - We can see **big second wave coming to Spain & France from August**.

# In[58]:


train_europe_march['prev_confirmed'] = train_europe_march.groupby('country')['confirmed'].shift(1)
train_europe_march['new_case'] = train_europe_march['confirmed'] - train_europe_march['prev_confirmed']
fig = px.line(train_europe_march,
              x='date', y='new_case', color='country',
              title=f'DAILY NEW Confirmed cases by country in Europe')
fig.show()


# In[59]:


train_europe_march['prev_new_case'] = train_europe_march.groupby('country')['new_case'].shift(1)
train_europe_march['growth_factor'] = train_europe_march['new_case'] / train_europe_march['prev_new_case']
train_europe_march['growth_factor'].fillna(0, inplace=True)
fig = px.line(train_europe_march,
              x='date', y='growth_factor', color='country',
              title=f'Growth factor by country in Europe')
fig.add_trace(go.Scatter(x=[train_europe_march['date'].min(), train_europe_march['date'].max()], y=[1., 1.],
                         name='Growth factor=1.', line=dict(dash='dash', color=('rgb(255, 0, 0)'))))
fig.update_yaxes(range=[0., 5.])
fig.show()


# <a id="id_asia"></a>
# # Asia

# **[April]**<br/>
# In Asia, China & Iran have many confirmed cases, followed by South Korea & Turkey. 
# 
# **[July]**<br/>
# 
# When I notice, coronavirus original place China has relatively few confirmed cases compared to the other seriously affected countries (it might be because China does not report all the confirmed cases).<br/>
# **South Asia, including India, Bangladesh, Pakistan, Iran, Saudi Arabia, Turkey** was more affected now.

# In[60]:


country_latest = country_df.query('date == @target_date')

fig = px.choropleth(country_latest, locations="country", 
                    locationmode='country names', color="confirmed", 
                    hover_name="country", range_color=[1, 300000], 
                    color_continuous_scale='portland', 
                    title=f'Asian Countries with Confirmed Cases as of {target_date}', scope='asia', height=800)
fig.show()


# The coronavirus hit Asia in early phase, how is the situation now?<br/>
# China & Korea is already in decreasing phase.<br/>
# 
# Unlike China or Korea, daily new confirmed cases were kept increasing on March or April, especially in Iran or Japan. But the number is started to decrease now on these country as well now.
# 
# The number increases exponentially in **India** as of July, need to be careful.

# In[61]:


top_asian_country_df = country_df[country_df['country'].isin([
    'China', 'Indonesia', 'Iran', 'Japan', 'Korea, South', 'Malaysia', 'Philippines',
    'India', 'Bangladesh', 'Pakistan', 'Saudi Arabia', 'Turkey'
])]

fig = px.line(top_asian_country_df,
              x='date', y='new_case', color='country',
              title=f'DAILY NEW Confirmed cases Asia')
fig.show()


# In[62]:


top_asian_country_df['prev_new_case'] = top_asian_country_df.groupby('country')['new_case'].shift(1)
top_asian_country_df['growth_factor'] = top_asian_country_df['new_case'] / top_asian_country_df['prev_new_case']
top_asian_country_df['growth_factor'].fillna(0, inplace=True)
fig = px.line(top_asian_country_df,
              x='date', y='growth_factor', color='country',
              title=f'Growth factor by country in Asia')
fig.add_trace(go.Scatter(x=[top_asian_country_df['date'].min(), top_asian_country_df['date'].max()], y=[1., 1.],
                         name='Growth factor=1.', line=dict(dash='dash', color=('rgb(255, 0, 0)'))))
fig.update_yaxes(range=[0., 5.])
fig.show()


# <a id="id_recover"></a>
# # Which country is recovering now?

# We saw that Coronavirus now hits Europe & US, in serious situation. How does it converge?
# 
# We can refer other country where confirmed cases is already decreasing.<br/>
# Here I defined `new_case_peak_to_now_ratio`, as a ratio of current new case and the max new case for each country.<br/>
# If new confirmed case is biggest now, its ratio is 1.
# Its ratio is expected to be low value for the countries where the peak has already finished.  

# In[63]:


max_confirmed = country_df.groupby('country')['new_case'].max().reset_index()
country_latest = pd.merge(country_latest, max_confirmed.rename({'new_case': 'max_new_case'}, axis=1))
country_latest['new_case_peak_to_now_ratio'] = country_latest['new_case'] / country_latest['max_new_case']


# In[64]:


recovering_country = country_latest.query('new_case_peak_to_now_ratio < 0.5')
major_recovering_country = recovering_country.query('confirmed > 100')


# The ratio is 0 for the country with very few confirmed cases are reported.<br/>
# I choosed the countries with its confirmed cases more than 100, to see only major countries with the ratio is low.
# 
# We can see:
#  - Middle East coutnries.
#  - South Africa countries.
#  - China & Korea from Asia.

# In[65]:


fig = px.bar(major_recovering_country.sort_values('new_case_peak_to_now_ratio', ascending=False),
             x='new_case_peak_to_now_ratio', y='country',
             title=f'Mortality rate LOW: top 30 countries on {target_date}', text='new_case_peak_to_now_ratio', height=1000, orientation='h')
fig.show()


# Let's see by map. Yellow countries have high ratio, currently increasing countries. **Blue & purple countries** have low ratio, already decreasing countries from its peak.
# 
# [At July 2]: We can see that the number is still increasing in **US, South America, South Asia & Africa**, while we can see converging trend in **Europe & East Asia (around China)**.

# In[66]:


fig = px.choropleth(country_latest, locations="country", 
                    locationmode='country names', color="new_case_peak_to_now_ratio", 
                    hover_name="country", range_color=[0, 1], 
                    # color_continuous_scale="peach", 
                    hover_data=['confirmed', 'fatalities', 'new_case', 'max_new_case'],
                    title='Countries with new_case_peak_to_now_ratio')
fig.show()


# Let's see some recovering countries.
# 
# ## China
# 
# When we check each state stats, we can see Hubei, the starting place, is extremely large number of confirmed cases.<br/>
# Other states records actually few confirmed cases compared to Hubei.

# In[67]:


china_df = train.query('country == "China"')
china_df['prev_confirmed'] = china_df.groupby('province')['confirmed'].shift(1)
china_df['new_case'] = china_df['confirmed'] - china_df['prev_confirmed']
china_df.loc[china_df['new_case'] < 0, 'new_case'] = 0.


# In[68]:


fig = px.line(china_df,
              x='date', y='new_case', color='province',
              title=f'DAILY NEW Confirmed cases in China by province')
fig.show()


# ## The situation of Hubei now?
# 
# Hubei record its new case peak on Feb 14. And finally, new case was not found on March 19.
# 
# To become no new case found, it took **about 2month after confirmed cases occured**, and **1 month after the peak has reached.** <br/>
# This term will be the reference for other country to how long we must lock-down the city.
# 
# The anomally number of 325 on Apr 17 is maybe due to the news that China admits the reporting number was low, so its number is not fatalities on this date, but past fatalities with not reported until now.

# In[69]:


china_df.query('(province == "Hubei") & (date > "2020-03-10")')[['country_province', 'date', 'confirmed', 'fatalities', 'recovered', 'new_case']]


# <a id="id_converge"></a>
# # When will it converge? - Estimation by sigmoid fitting
# 
# I guess everyone is wondering when the coronavirus converges. Let's estimate it roughly using sigmoid fitting.<br/>
# I referenced below kernels for original ideas.
# 
#  - [Sigmoid per country](https://www.kaggle.com/group16/sigmoid-per-country-no-leakage) by @group16
#  - [COVID-19 growth rates per country](https://www.kaggle.com/mikestubna/covid-19-growth-rates-per-country) by @mikestubna
#  
#  
# [At July]: Now I am feeling coronavirus does not converge in this year... I expand prediction term until the end of this year.

# In[70]:


def sigmoid(t, M, beta, alpha, offset=0):
    alpha += offset
    return M / (1 + np.exp(-beta * (t - alpha)))

def error(x, y, params):
    M, beta, alpha = params
    y_pred = sigmoid(x, M, beta, alpha)

    # apply weight, latest number is more important than past.
    weight = np.arange(len(y_pred)) ** 2
    loss_mse = np.mean((y_pred - y) ** 2 * weight)
    return loss_mse

def gen_random_color(min_value=0, max_value=256) -> str:
    """Generate random color for plotly"""
    r, g, b = np.random.randint(min_value, max_value, 3)
    return f'rgb({r},{g},{b})'


# In[71]:


def fit_sigmoid(exclude_days=0):
    target_country_df_list = []
    pred_df_list = []
    for target_country in top30_countries:
        print('target_country', target_country)
        # --- Train ---
        target_country_df = country_df.query('country == @target_country')

        #train_start_date = target_country_df['date'].min()
        train_start_date = target_country_df.query('confirmed > 1000')['date'].min()
        train_end_date = pd.to_datetime(target_date) - pd.Timedelta(f'{exclude_days} days')
        target_date_df = target_country_df.query('(date >= @train_start_date) & (date <= @train_end_date)')
        if len(target_date_df) <= 7:
            print('WARNING: the data is not enough, use 7 more days...')
            train_start_date -= pd.Timedelta('7 days')
            target_date_df = target_country_df.query('(date >= @train_start_date) & (date <= @train_end_date)')

        confirmed = target_date_df['confirmed'].values
        x = np.arange(len(confirmed))

        lossfun = lambda params: error(x, confirmed, params)
        res = sp.optimize.minimize(lossfun, x0=[np.max(confirmed) * 5, 0.04, 2 * len(confirmed) / 3.], method='nelder-mead')
        M, beta, alpha = res.x
        # sigmoid_models[key] = (M, beta, alpha)
        # np.clip(sigmoid(list(range(len(data), len(data) + steps)), M, beta, alpha), 0, None).astype(int)

        # --- Pred ---
        pred_start_date = target_country_df['date'].min()
        pred_end_date = pd.to_datetime('2021-01-01')
        days = int((pred_end_date - pred_start_date) / pd.Timedelta('1 days'))
        # print('pred start', pred_start_date, 'end', pred_end_date, 'days', days)

        x = np.arange(days)
        offset = (train_start_date - pred_start_date) / pd.Timedelta('1 days')
        print('train_start_date', train_start_date, 'offset', offset, 'params', M, beta, alpha)
        y_pred = sigmoid(x, M, beta, alpha, offset=offset)
        # target_country_df['confirmed_pred'] = y_pred

        all_dates = [pred_start_date + np.timedelta64(x, 'D') for x in range(days)]
        pred_df = pd.DataFrame({
            'date': all_dates,
            'country': target_country,
            'confirmed_pred': y_pred,
        })

        target_country_df_list.append(target_country_df)
        pred_df_list.append(pred_df)
    return target_country_df_list, pred_df_list


# In[72]:


def plot_sigmoid_fitting(target_country_df_list, pred_df_list, title=''):
    n_countries = len(top30_countries)

    # --- Plot ---
    fig = go.Figure()

    for i in range(n_countries):
        target_country = top30_countries[i]
        target_country_df = target_country_df_list[i]
        pred_df = pred_df_list[i]
        color = gen_random_color(min_value=20)
        # Prediction
        fig.add_trace(go.Scatter(
            x=pred_df['date'], y=pred_df['confirmed_pred'],
            name=f'{target_country}_pred',
            line=dict(color=color, dash='dash')
        ))

        # Ground truth
        fig.add_trace(go.Scatter(
            x=target_country_df['date'], y=target_country_df['confirmed'],
            mode='markers', name=f'{target_country}_actual',
            line=dict(color=color),
        ))
    fig.update_layout(
        title=title, xaxis_title='Date', yaxis_title='Confirmed cases')
    fig.show()


# In[73]:


target_country_df_list, pred_df_list = fit_sigmoid(exclude_days=0)


# In[74]:


plot_sigmoid_fitting(target_country_df_list, pred_df_list, title='Sigmoid fitting with all latest data')


# If we believe above curve, confirmed cases is slowing down now and it is almost converging in most of the country.<br/>
# But it might take until beginning on June in US, and August in Brazil. Thus it still need long-term to converge in world wide. We have a risk of 2nd wave too.
# 
# I'm not confident how this sigmoid fitting is accurate, it's just an estimation by some modeling.<br/>
# Let's try validation by excluding last 7 days data.

# In[75]:


target_country_df_list, pred_df_list = fit_sigmoid(exclude_days=7)


# In[76]:


plot_sigmoid_fitting(target_country_df_list, pred_df_list, title='Sigmoid fitting without last 7days data')


# Now I noticed that sigmoid fitting tend to **underestimate** the curve, and its actual value tend to be more than sigmoid curve estimation.<br/>
# Therefore, we need to be careful to see sigmoid curve fitting data, actual situation is likely to be worse than the previous figure trained with all data.

# <a id="id_ref"></a>
# # Further reading
# 
# That's all! Thank you for reading long kernel. I hope the world get back peace & usual daily life as soon as possible.
# 
# Here are the other information for further reading.
# 
# My other kernels:
#  - [COVID-19: Effect of temperature/humidity](https://www.kaggle.com/corochann/covid-19-effect-of-temperature-humidity)
#  - [COVID-19: Spread situation by prefecture in Japan](https://www.kaggle.com/corochann/covid-19-spread-situation-by-prefecture-in-japan)
