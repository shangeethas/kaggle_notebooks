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
    #competitions from the list of visual inspection of Kaggle competition website
    #cv_competitions = ['digit-recognizer', 'state-farm-distracted-driver-detection', 'kannada-mnist',
    #                             'peking-university/baidu - autonomous-driving', 'trends-neuroimaging',
    #                             'google-landmark-recognition-2020', 'open-images-2019 - object-detection',
    #                             'google-landmark-retrieval-2020', 'nfl-1st-and-future - impact-detection',
    #                             'Open Images 2019 - Visual Relationship', 'Machine Learning@NTUT - Computer Vision',
    #                             'KUL H02A5a Computer Vision: Group assignment 0',
    #                             'NYU Computer Vision - CSCI-GA.2271 2019',
    #                             'NYU Computer Vision - CSCI-GA.2271 2020', 'sfu-cmpt-computer-vision-course-cnn-lab',
    #                             'Computer Vision Competition[SC-2020]', 'computer-vision-competition[sc-2020]',
    #                             'ml-guild-computer-vision-practicum', 'Computer Vision Training Camp 2020 - Starter',
    #                             'DUTh DEECE - Computer Vision 2019-20 - Homework 4',
    #                            'kul-HO2a5a-computer-vision: group-assignment-1',
    #                             'Inter IT -- Computer Vision', 'Computer Vision Training Camp 2020 - Advanced',
    #                             'applications-of-deep-learning(WUSTL, Spring 2020B)',
    #                            'applications-of-deep-learning(WUSTL, Spring 2020)',
    #                             'computer-vision-training-camp-2020',
    #                             'Rice University: Data Mining & Statistical Learning', 'au-ece-cvml2021',
    #                             'thousand-facial-landmarks', 'car-plates-ocr', 'computer-vision-cs543/ece549',
    #                             'bird-identification', 'intro-dl', 'au-eng-cvml2018',
    #                            'AlBiz 2020 Spring Task 3: Fashion MNIST', 'Tiny Image Net Challenge - TCD',
    #                             'au-eng-cvml2020', 'au-eng-cvml2019', 'jamp-hackathon-drive-1',
    #                             'ece281b2017', 'jio-hackthon-2', 'Gradient\'s Computer Vision Challenge',
    #                             'tj-computer-vision-test', 'bird-identification',
    #                             'parrot-2nd-computer-vision-competition', 'vietai-advanced-class-00-final-project',
    #                             'tiny-imagenet-challenge', 'mnist-classification', '2020-fall:instance-segmentation',
    #                             '2021-sptring-instance-segmentation', 'machine-learning@ntut-2018',
    #                             'cs543/ece549-assignment-4: deep-convoultional-neural-networks',
    #                             'Machine Learning@NTUT 2019', 'ml-guild-cv-practicum',
    #                             'cte-dl4cv-assignment-1'
    #                             ]
    # competitions from the list of Kaggle API printing and data set is cross checked to have images
    cv_downloaded_competitions = ['bms-molecular-translation', 'iwildcam2021-fgvc8', 'herbarium-2021-fgvc8',
                                 'plant-pathology-2021-fgvc8', 'hotel-id-2021-fgvc8', 'indoor-location-navigation',
                                 'hpa-single-cell-image-classification', 'hubmap-kidney-segmentation', 'shopee-product-matching',
                                 'vinbigdata-chest-xray-abnormalities-detection', 'ranzcr-clip-catheter-line-classification',
                                 'cassava-leaf-disease-classification', 'nfl-impact-detection', 'lyft-motion-prediction-autonomous-vehicles',
                                 'rsna-str-pulmonary-embolism-detection', 'osic-pulmonary-fibrosis-progression', 'landmark-recognition-2020',
                                 'global-wheat-detection', 'landmark-retrieval-2020', 'siim-isic-melanoma-classification',
                                 'open-images-object-detection-rvc-2020', 'open-images-instance-segmentation-rvc-2020',
                                 'prostate-cancer-grade-assessment', 'alaska2-image-steganalysis', 'trends-assessment-prediction',
                                 'imet-2020-fgvc7', 'iwildcam-2020-fgvc7', 'herbarium-2020-fgvc7', 'plant-pathology-2020-fgvc7',
                                 'imaterialist-fashion-2020-fgvc7', 'flower-classification-with-tpus', 'imagenet-object-localization-challenge',
                                 'ds4g-environmental-insights-explorer', 'pku-autonomous-driving', 'Kannada-MNIST', 'understanding_cloud_organization',
                                 'rsna-intracranial-hemorrhage-detection', '3d-object-detection-for-autonomous-vehicles',
                                 'severstal-steel-defect-detection', 'kuzushiji-recognition', 'open-images-2019-object-detection',
                                 'open-images-2019-visual-relationship', 'open-images-2019-instance-segmentation',
                                 'recursion-cellular-image-classification', 'aptos2019-blindness-detection', 'siim-acr-pneumothorax-segmentation',
                                 'generative-dog-images', 'aerial-cactus-identification', 'imaterialist-fashion-2019-FGVC6', 'inaturalist-2019-fgvc6',
                                 'imet-2019-fgvc6', 'iwildcam-2019-fgvc6']
    cv_completed_competitions = ['petfinder-adoption-prediction', 'histopathologic-cancer-detection', 'humpback-whale-identification',
                                 'human-protein-atlas-image-classification', 'airbus-ship-detection', 'inclusive-images-challenge',
                                 'rsna-pneumonia-detection-challenge', 'tgs-salt-identification-challenge', 'google-ai-open-images-object-detection-track',
                                 'google-ai-open-images-visual-relationship-track', 'whale-categorization-playground', 'avito-demand-prediction',
                                 'imaterialist-challenge-fashion-2018', 'imaterialist-challenge-furniture-2018']

    for competition in competitions:
        print('Competition Name : ' + competition.ref)
        if competition.ref in cv_completed_competitions:
            print('Competition Name : ' + competition.ref)
            #print_all_kernels_info(competition=competition.ref, api=api)

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
                    api.kernels_pull(kernel.ref, "/Users/shangeetha/Desktop/Kaggle_Competition_CV_150")

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
            if kernel.totalVotes >= 150:
                folderName = "/Users/shangeetha/Desktop/Kaggle_Competition_CV_150/" + competition
                api.kernels_pull(kernel.ref, folderName)

        if not kernels_scripts: break


if __name__ == "__main__":
    # print_competition_keys()
    p = Pool(4)
    p.map(print_competitions_info())
