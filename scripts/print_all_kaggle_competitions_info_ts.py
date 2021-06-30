import time

from kaggle import KaggleApi
from multiprocessing import Pool
import datetime
import itertools
import csv


def new_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api


def print_competition_keys():
    api = new_kaggle_api()
    competitions = api.competitions_list()
    for key in dir(competitions[0]):
        print('{}: {}'.format(key, getattr(competitions[0], key)))


def print_competitions_info(page=1):
    api = new_kaggle_api()

    competitions = api.competitions_list(sort_by='latestDeadline', page=page)

    time.sleep(1)
    print('Competitions size ' + str(len(competitions)))
    # competitions from the list of visual inspection of Kaggle competition website
    # ts_competitions =
    # competitions from the list of Kaggle API printing and data set is cross checked to have date time feature
    ts_completed_competitions = ['birdclef-2021', 'ncaam-march-mania-2021',
                                 'ncaam-march-mania-2021-spread', 'rfcx-species-audio-detection', 'acea-water-prediction',
                                 'nfl-big-data-bowl-2021', 'predict-volcanic-eruptions-ingv-oe', 'osic-pulmonary-fibrosis-progression',
                                 'birdsong-recognition', 'm5-forecasting-accuracy', 'm5-forecasting-uncertainty', 'herbarium-2020-fgvc7',
                                 'liverpool-ion-switching', 'covid19-global-forecasting-week-5', 'march-madness-analytics-2020',
                                 'covid19-global-forecasting-week-4', 'covid19-global-forecasting-week-3', 'covid19-global-forecasting-week-2', 'covid19-global-forecasting-week-1',
                                 'data-science-bowl-2019', 'nfl-big-data-bowl-2020', 'nfl-playing-surface-analytics',
                                 'ashrae-energy-prediction', 'bigquery-geotab-intersection-congestion', 'youtube8m-2019', 'ieee-fraud-detection',
                                 'two-sigma-financial-news', 'freesound-audio-tagging-2019',
                                 'inaturalist-2019-fgvc6', 'iwildcam-2019-fgvc6', 'LANL-Earthquake-Prediction', 'mens-machine-learning-competition-2019',
                                 'womens-machine-learning-competition-2019', 'vsb-power-line-fault-detection', 'elo-merchant-category-recommendation',
                                 'ga - customer - revenue - prediction', 'reducing-commercial-aviation-fatalities', 'NFL-Punt-Analytics-Competition',
                                  'PLAsTiCC-2018', 'quickdraw-doodle-recognition', 'new-york-city-taxi-fare-prediction'
                                 'predicting-red-hat-business-value', 'flavours-of-physics-kernels-only',
                                 'home-credit-default-risk', 'freesound-audio-tagging', 'avito-demand-prediction', 'inaturalist-2018',
                                 'talkingdata-adtracking-fraud-detection', 'donorschoose-application-screening',
                                 'mens-machine-learning-competition-2018', 'womens-machine-learning-competition-2018',
                                 'recruit-restaurant-visitor-forecasting', 'favorita-grocery-sales-forecasting',
                                 'zillow-prize-1', 'kkbox-music-recommendation-challenge', 'web-traffic-time-series-forecasting',
                                 ]
    # cdp-unlocking-climate-solutions - huge dataset with multiple datasets -  don't know whether time element is available
    # Not full list of time series competitions are added to downloading list of notebooks
    # Excluded forest-cover-type-kernels-only because it had time related values for example value at noon, value at 3 pm etc.
    # Stopped at LANL-Earthquake-Prediction and
    # added predicting-red-hat-business-value from down list

    for competition in competitions:
        print('Competition Name : ' + competition.ref)
        if competition.ref in ts_completed_competitions:
            print('Competition Name : ' + competition.ref)
            print_all_kernels_info(competition=competition.ref, api=api)

    if competitions:
        print_competitions_info(page=page + 1)


def print_script_kernels_info(competition, api):
    with open('/Users/shangeetha/Desktop/Kaggle_Competition_Code/scripts_votes_path_3_ts.csv', 'a') as file:
        writer = csv.writer(file)
        for pagenumber in itertools.count(1):
            kernels_scripts = api.kernels_list(competition=competition,
                                               language='python',
                                               kernel_type='script',
                                               sort_by='voteCount',
                                               page=pagenumber)

            # kernelRef = ""
            for kernel in kernels_scripts:
                path = kernel.ref.split("/")
                file_name = path[1] + ".py"
                writer.writerow([competition, kernel.ref, kernel.totalVotes, file_name])
                kernelRef = kernel.ref
                # api.kernels_pull(kernel.ref, "/Users/shangeetha/Desktop/Kaggle_Competition_Scripts_2_TS")

            if not kernels_scripts: break


def print_notebook_kernels_info(competition, api):
    with open('/Users/shangeetha/Desktop/Kaggle_Competition_Code/notebooks_votes_path_ts.csv', 'a') as file:
        writer = csv.writer(file)
        for pagenumber in itertools.count(1):
            kernels_scripts = api.kernels_list(competition=competition,
                                               language='python',
                                               kernel_type='notebook',
                                               sort_by='voteCount',
                                               page=pagenumber)

            # kernelRef = ""
            #for kernel in kernels_scripts:
                #if kernel.totalVotes >= 150:
                    #print('Kernel Name : ' + kernel.)
                    #api.kernels_pull(kernel.ref, "/Users/shangeetha/Desktop/Kaggle_Competition_TS_150")

            #if not kernels_scripts: break


def print_all_kernels_info(competition, api):
    for pageNum in itertools.count(1):
        kernels_scripts = api.kernels_list(competition=competition,
                                           language='python',
                                           kernel_type='all',
                                           sort_by='voteCount',
                                           page=pageNum)

        # kernelRef = ""
        for kernel in kernels_scripts:
            if kernel.totalVotes >= 150:
                folderName = "/Users/shangeetha/Desktop/Kaggle_Competition_TS_150/" + competition
                api.kernels_pull(kernel.ref, folderName)

        if not kernels_scripts: break


if __name__ == "__main__":
    # print_competition_keys()
    p = Pool(4)
    p.map(print_competitions_info())
