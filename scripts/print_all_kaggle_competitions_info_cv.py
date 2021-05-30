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

    cv_completed_competitions = ['Digit Recognizer', 'State Farm Distracted Driver Detection', 'Kannada MNIST',
                                 'Peking University/Baidu - Autonomous Driving', 'TReNDS Neuroimaging',
                                 'Google Landmark Recognition 2020', 'Open Images 2019 - Object Detection',
                                 'Google Landmark Retrieval 2020', 'NFL 1st and Future - Impact Detection',
                                 'Open Images 2019 - Visual Relationship', 'Machine Learning@NTUT - Computer Vision',
                                 'KUL H02A5a Computer Vision: Group assignment 0',
                                 'NYU Computer Vision - CSCI-GA.2271 2019',
                                 'NYU Computer Vision - CSCI-GA.2271 2020', 'SFU CMPT Computer Vision Course CNN Lab',
                                 'Computer Vision Competition[SC-2020]', 'Computer Vision Competition[SC-2020]',
                                 'ML Guild Computer Vision Practicum', 'Computer Vision Training Camp 2020 - Starter',
                                 'DUTh DEECE - Computer Vision 2019-20 - Homework 4',
                                 'KUL HO2A5a Computer Vision: Group assignment 1',
                                 'Inter IT -- Computer Vision', 'Computer Vision Training Camp 2020 - Advanced',
                                 'Applications of Deep Learning(WUSTL, Spring 2020B)',
                                 'Applications of Deep Learning(WUSTL, Spring 2020)',
                                 'Computer Vision Training Camp 2020',
                                 'Rice University: Data Mining & Statistical Learning', 'AU-ECE-CVML2021',
                                 'Thousand Facial Landmarks', 'Car plates OCR', 'Computer vision CS543/ECE549',
                                 'Bird Identification', 'intro-dl', 'AU-ENG-CVML2018',
                                 'AlBiz 2020 Spring Task 3: Fashion MNIST', 'Tiny Image Net Challenge - TCD',
                                 'AU-ENG-CVML2020', 'AU-ENG-CVML2019', 'JAMP Hackathon Drive 1',
                                 'ECE281B2017', 'Jio Hackthon 2', 'Gradient\'s Computer Vision Challenge',
                                 'TJ Computer Vision Test', 'Bird Identification',
                                 'Parrot 2nd computer vision competition', 'VietAI Advanced Class 00 Final Project',
                                 'Tiny Imagenet Challenge', 'MNIST classification', '2020 Fall:Instance Segmentation',
                                 '2021 Sptring Instance Segmentation', 'Machine Learning@NTUT 2018',
                                 'CS543/ECE549 Assignment 4: Deep Convoultional Neural Networks',
                                 'Machine Learning@NTUT 2019', 'ML Guild CV Practicum',
                                 'CTE-DL4CV Assignment 1'
                                 ]
    cv_same_competition_names = ['Digit Recognizer']

    for competition in competitions:
        if competition.ref in cv_completed_competitions:
            print('Competition Name : ' + competition.ref)
            print_all_kernels_info(competition=competition.ref, api=api)

    if competitions:
        print_competitions_info(page=page + 1)


def print_script_kernels_info(competition, api):
    with open('/Users/shangeetha/Desktop/Kaggle_Competition_Code/scripts_votes_path_3_cv.csv', 'a') as file:
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
                api.kernels_pull(kernel.ref, "/Users/shangeetha/Desktop/Kaggle_Competition_Scripts_2_CV")

            if not kernels_scripts: break


def print_notebook_kernels_info(competition, api):
    with open('/Users/shangeetha/Desktop/Kaggle_Competition_Code/notebooks_votes_path_cv.csv', 'a') as file:
        writer = csv.writer(file)
        for pagenumber in itertools.count(1):
            kernels_scripts = api.kernels_list(competition=competition,
                                               language='python',
                                               kernel_type='notebook',
                                               sort_by='voteCount',
                                               page=pagenumber)

            # kernelRef = ""
            for kernel in kernels_scripts:
                if kernel.totalVotes >= 150:
                    api.kernels_pull(kernel.ref, "/Users/shangeetha/Desktop/Kaggle_Competition_NLP_150")

            if not kernels_scripts: break


def print_all_kernels_info(competition, api):
    for pageNum in itertools.count(1):
        kernels_scripts = api.kernels_list(competition=competition,
                                           language='python',
                                           kernel_type='all',
                                           sort_by='voteCount',
                                           page=pageNum)

        # kernelRef = ""
        for kernel in kernels_scripts:
            if kernel.totalVotes >= 0:
                folderName = "/Users/shangeetha/Desktop/Kaggle_Competition_CV_0/" + competition
                api.kernels_pull(kernel.ref, folderName)

        if not kernels_scripts: break


if __name__ == "__main__":
    # print_competition_keys()
    p = Pool(4)
    p.map(print_competitions_info())
