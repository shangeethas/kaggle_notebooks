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

    completed_competitions = ['deepfake-detection-challenge', 'tensorflow2-question-answering',
                              'data-science-bowl-2019', 'google-quest-challenge',
                              'nfl-big-data-bowl-2020', 'nfl-playing-surface-analytics',
                              'pku-autonomous-driving', 'santa-workshop-tour-2019',
                              'bengaliai-cv19', 'digit-recognizer', 'titanic',
                              'house-prices-advanced-regression-techniques', 'imagenet-object-localization-challenge',
                              'competitive-data-science-predict-future-sales', 'nlp-getting-started',
                              'santa-2019-revenge-of-the-accountants']
    nlp_competitions = ['gendered-pronoun-resolution', 'petfinder-adoption-prediction',
                        'quora-insincere-questions-classification', 'avito-demand-prediction',
                        'jigsaw-toxic-comment-classification-challenge', 'text-normalization-challenge-russian-language',
                        'text-normalization-challenge-english-language', 'msk-redefining-cancer-treatment',
                        'quora-question-pairs', 'two-sigma-connect-rental-listing-inquiries',
                        'jigsaw-unintended-bias-in-toxicity-classification',
                        'data-science-for-good-city-of-los-angeles', '20-newsgroups-ciphertext-challenge',
                        'movie-review-sentiment-analysis-kernels-only', 'transfer-learning-on-stack-exchange-tags',
                        'whats-cooking-kernels-only']

    for competition in competitions:
        # title = getattr(competition, 'title')
        # print('{}'.format(title))
        # print_notebook_kernels_info(competition=competition.ref, api=api, page=1)
        # now = datetime.datetime.utcnow()
        # ended_at = competition.deadline

        # if ended_at < now:
        # print_notebook_kernels_info(competition, api)
        # print_script_kernels_info(competition, api)
        # print_script_kernels_info_above_100_votes(competition=competition.ref, api=api)
        if competition.ref in nlp_competitions:
            print('Competition Name : ' + competition.ref)
            print_all_kernels_info(competition=competition.ref, api=api)

    if competitions != []:
        print_competitions_info(page=page + 1)

def print_script_kernels_info(competition, api):
    with open('/Users/shangeetha/Desktop/Kaggle_Competition_Code/scripts_votes_path_3.csv', 'a') as file:
        writer = csv.writer(file)
        for pagenum in itertools.count(1):
            kernels_scripts = api.kernels_list(competition=competition,
                                       language='python',
                                       kernel_type='script',
                                       sort_by='voteCount',
                                       page=pagenum)

            # kernelRef = ""
            for kernel in kernels_scripts:
                #if kernel.totalVotes >= 100:
               # if kernel.ref == kernelRef:
               #     continue
                path = kernel.ref.split("/")
                file_name = path[1] + ".py"
                writer.writerow([competition, kernel.ref, kernel.totalVotes, file_name])
                kernelRef = kernel.ref
                api.kernels_pull(kernel.ref, "/Users/shangeetha/Desktop/Kaggle_Competition_Scripts_2")

            if not kernels_scripts: break

def print_notebook_kernels_info(competition, api):
    with open('/Users/shangeetha/Desktop/Kaggle_Competition_Code/notebooks_votes_path.csv', 'a') as file:
        writer = csv.writer(file)
        for pagenum in itertools.count(1):
            kernels_scripts = api.kernels_list(competition=competition,
                                               language='python',
                                               kernel_type='notebook',
                                               sort_by='voteCount',
                                               page=pagenum)

            # kernelRef = ""
            for kernel in kernels_scripts:
                if kernel.totalVotes >= 150:
                # if kernel.ref == kernelRef:
                #     continue
                # path = kernel.ref.split("/")
                # file_name = path[1] + ".py"
                # writer.writerow([competition, kernel.ref, kernel.totalVotes, file_name])
                # kernelRef = kernel.ref
                    api.kernels_pull(kernel.ref, "/Users/shangeetha/Desktop/Kaggle_Competition_NLP_150")

            if not kernels_scripts: break

def print_all_kernels_info(competition, api):
    # with open('/Users/shangeetha/Desktop/Kaggle_Competition_Kernels_All/kernels_votes_path_all.csv', 'a') as file:
      #   writer = csv.writer(file)
        for pagenum in itertools.count(1):
            kernels_scripts = api.kernels_list(competition=competition,
                                               language='python',
                                               kernel_type='all',
                                               sort_by='voteCount',
                                               page=pagenum)

            # kernelRef = ""
            for kernel in kernels_scripts:
                if kernel.totalVotes >= 150:
                # if kernel.ref == kernelRef:
                #     continue
                # path = kernel.ref.split("/")
                # file_name = path[1] + ".py"
                # writer.writerow([competition, kernel.ref, kernel.totalVotes, file_name])
                    folderName = "/Users/shangeetha/Desktop/Kaggle_Competition_NLP_150/" + competition
                    api.kernels_pull(kernel.ref, folderName)

            if not kernels_scripts: break



if __name__ == "__main__":
    # print_competition_keys()
    p = Pool(4)
    p.map(print_competitions_info())
