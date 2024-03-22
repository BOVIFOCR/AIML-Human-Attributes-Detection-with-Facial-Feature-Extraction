''' Facial attribute extraction using mxnet and facenet '''
#--------------------------------
# Date : 10-07-2020
# Project : Facial Attribute Extraction
# Category : DeepLearning
# Company : weblineindia
# Department : AI/ML
#--------------------------------
import os
import cv2
import sys
import glob
import logging
import argparse
import numpy as np
import mxnet as mx
import pandas as pd
from pathlib import Path
import pickle
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt

import model.emotion.detectemotion as ime
from mxnet_moon.lightened_moon import lightened_moon_feature

from Face_Pose.pose_detection_retinaface import *

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# import pdb

# Load path from .env
faceProto = os.getenv("FACEDETECTOR")
faceModel = os.getenv("FACEMODEL")
ageProto = os.getenv("AGEDETECTOR")
ageModel = os.getenv("AGEMODEL")
genderProto = os.getenv("GENDERDETECTOR")
genderModel = os.getenv("GENDERMODEL")
pathImg = os.getenv("IMGPATH")
APPROOT = os.getenv("APPROOT")

#Load face detection model
faceNet=cv2.dnn.readNet(faceModel,faceProto)
#Load age detection model
ageNet=cv2.dnn.readNet(ageModel,ageProto)
#Load gender detection model
genderNet=cv2.dnn.readNet(genderModel,genderProto)
#create instance for emotion detection
ed = ime.Emotional()


""" Detects face and extracts the coordinates"""
def getFaceBox(net, image, conf_threshold=0.7):
    image=image.copy()
    imageHeight=image.shape[0]
    imageWidth=image.shape[1]
    blob=cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*imageWidth)
            y1=int(detections[0,0,i,4]*imageHeight)
            x2=int(detections[0,0,i,5]*imageWidth)
            y2=int(detections[0,0,i,6]*imageHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), int(round(imageHeight/150)), 8)
    return image,faceBoxes


""" Detects age and gender """
def genderAge(image, faceBox=None):

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']
    
    if not faceBox is None:
        padding=20
        face=image[max(0,faceBox[1]-padding):
            min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding)
            :min(faceBox[2]+padding, image.shape[1]-1)]
    else:
        face = image
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict the gender
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    # Predict the age
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    # Return
    return gender,age


def count_verification_protocol_stats(file_path):
    all_pairs = []
    true_labels = []
    race_count = {}
    subject_count = {}
    sample_count = {}

    print(f'Loading pairs from file \'{file_path}\'')
    with open(file_path, 'r') as file:
        for l, line in enumerate(file):
            line = line.strip()
            print(f'line {l} - \'{line}\'', end='\r')
            paths_pair = line.split(';')
            pair_dict = {}
            pair_dict['race'] = paths_pair[0].split('/')[0]

            subj0 = paths_pair[0].split('/')[-2]
            subj1 = paths_pair[1].split('/')[-2]
            pair_dict['label'] = 1 if subj0 == subj1 else 0
            true_labels.append(pair_dict['label'])

            sample_dict0 = {}
            sample_dict0['path'] = paths_pair[0]
            pair_dict['sample0'] = sample_dict0
            sample_dict1 = {}
            sample_dict1['path'] = paths_pair[1]
            pair_dict['sample1'] = sample_dict1
            
            all_pairs.append(pair_dict)
            for face in paths_pair:
                race, subject, sample = face.split('/')

                if race in race_count:
                    race_count[race] += 1
                else:
                    race_count[race] = 1
                
                if subject in subject_count:
                    subject_count[subject] += 1
                else:
                    subject_count[subject] = 1

                if sample in sample_count:
                    sample_count[sample] += 1
                else:
                    sample_count[sample] = 1
        print('')

    return all_pairs, true_labels, race_count, subject_count, sample_count


def adjust_paths(all_pairs, img_path, ext='.png'):
    print('Adjusting pairs paths')
    for i, pair in enumerate(all_pairs):
        print(f'pair {i}/{len(all_pairs)}', end='\r')
        # print('pair:', pair)
        # race = pair['race']
        sample0, sample1 = pair['sample0']['path'], pair['sample1']['path']
        if ext != '':
            sample0 = sample0.replace('.jpg', ext)
            sample1 = sample1.replace('.jpg', ext)
        path_sample0 = os.path.join(img_path, sample0)
        path_sample1 = os.path.join(img_path, sample1)
        assert os.path.isfile(path_sample0), f'Error, file doesn\'t exists: \'{path_sample0}\''
        assert os.path.isfile(path_sample1), f'Error, file doesn\'t exists: \'{path_sample1}\''
        all_pairs[i]['sample0']['path'], all_pairs[i]['sample1']['path'] = path_sample0, path_sample1
    print('')
    return all_pairs


def save_attributes(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_attributes(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_predict_scores_labels(file_path):
    scores = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            score, label = line.strip().split(',')
            scores.append(float(score))
            labels.append(int(label))
    return scores, labels


def count_facial_attributes_by_race(facial_attributes_list):
    race_attributes_count = {'African': {}, 'Asian': {}, 'Caucasian': {}, 'Indian': {}}

    for item in facial_attributes_list:
        race = item['race']
        for idx_sample in range(2):
            key_sample = f'sample{idx_sample}'
            sample = item[key_sample]
            # print('sample.keys():', sample.keys())
            # sys.exit(0)
            attributes = sample['attributes']

            if race not in race_attributes_count:
                race_attributes_count[race] = {attr: 0 for attr in attributes}
            else:
                for attr in attributes:
                    if attr in race_attributes_count[race]:
                        race_attributes_count[race][attr] += 1
                    else:
                        race_attributes_count[race][attr] = 1

    return race_attributes_count


def count_face_pose_by_race(data):
    merged_data = {'African': {}, 'Asian': {}, 'Caucasian': {}, 'Indian': {}}

    print('Counting face poses')
    for e, entry in enumerate(data):
        print(f'Pair {e}/{len(data)}', end='\r')
        # print(f'Pair {e}/{len(data)}')
        race = entry['race']
        if race in merged_data:
            if not merged_data[race]:
                merged_data[race] = {
                    'roll': [entry['sample0']['roll']],
                    'yaw': [entry['sample0']['yaw']],
                    'pitch': [entry['sample0']['pitch']],
                    'angle': [entry['sample0']['angle']],
                    'Xfrontal': [entry['sample0']['Xfrontal']],
                    'Yfrontal': [entry['sample0']['Yfrontal']]
                }
            else:
                if 'roll' in entry['sample0']:
                    merged_data[race]['roll'].append(entry['sample0']['roll'])
                    merged_data[race]['yaw'].append(entry['sample0']['yaw'])
                    merged_data[race]['pitch'].append(entry['sample0']['pitch'])
                    merged_data[race]['angle'].append(entry['sample0']['angle'])
                    merged_data[race]['Xfrontal'].append(entry['sample0']['Xfrontal'])
                    merged_data[race]['Yfrontal'].append(entry['sample0']['Yfrontal'])

                if 'roll' in entry['sample1']:
                    merged_data[race]['roll'].append(entry['sample1']['roll'])
                    merged_data[race]['yaw'].append(entry['sample1']['yaw'])
                    merged_data[race]['pitch'].append(entry['sample1']['pitch'])
                    merged_data[race]['angle'].append(entry['sample1']['angle'])
                    merged_data[race]['Xfrontal'].append(entry['sample1']['Xfrontal'])
                    merged_data[race]['Yfrontal'].append(entry['sample1']['Yfrontal'])
    print('')
    return merged_data


def count_gender_age_by_race(face_verification_list):
    gender_age_by_race = {
        'African': {'Male': [], 'Female': []},
        'Asian': {'Male': [], 'Female': []},
        'Caucasian': {'Male': [], 'Female': []},
        'Indian': {'Male': [], 'Female': []}
    }

    for verification_pair in face_verification_list:
        race = verification_pair['race']
        sample0_gender = verification_pair['sample0']['gender']
        sample0_age = verification_pair['sample0']['age']
        sample1_gender = verification_pair['sample1']['gender']
        sample1_age = verification_pair['sample1']['age']

        gender_age_by_race[race][sample0_gender].append(sample0_age)
        gender_age_by_race[race][sample1_gender].append(sample1_age)

    return gender_age_by_race


def plot_gender_age_histograms(gender_age_by_race, output_file):
    races = list(gender_age_by_race.keys())
    genders = ['Male', 'Female']
    ages = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    fig, axs = plt.subplots(len(races), 2, figsize=(12, 9), sharey='row')
    fig.suptitle('Gender and Age Histograms by Race')

    for i, race in enumerate(races):
        for j, gender in enumerate(genders):
            num_samples = len(gender_age_by_race[race][gender])
            age_counts = [gender_age_by_race[race][gender].count(age) for age in ages]
            x = np.arange(len(ages))
            axs[i,j].bar(x, age_counts, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])
            axs[i,j].set_ylim(bottom=0, top=1500)
            axs[i,j].set_title(f'{race} - {gender} - #Samples={num_samples}')
            axs[i,j].set_xticks(x)
            axs[i,j].set_xticklabels(ages, rotation=45)
            axs[i,j].set_xlabel('Age')
            axs[i,j].set_ylabel('Count')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()


def save_bar_subplots(race_attributes_count, output_file):
    races = list(race_attributes_count.keys())
    # attributes = list(race_attributes_count[races[0]].keys())

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 20))

    for i, race in enumerate(races):
        # print(f'save_bar_subplots: {i}')
        # print(f'attributes: {attributes}')
        # print(f'race_attributes_count[{race}]: {race_attributes_count[race]}')
        attributes = list(race_attributes_count[race].keys())
        counts = [race_attributes_count[race][attr] for attr in attributes]
        ax = axes[i]
        ax.bar(np.arange(len(attributes)), counts)
        ax.set_ylim(bottom=0, top=4000)
        ax.set_xticks(np.arange(len(attributes)))
        ax.set_xticklabels(attributes, rotation=90)
        ax.set_title(race)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_face_pose_histograms(data, bins, ylim, filename):
    races = list(data.keys())
    # print('data.keys():', data.keys())
    # print('data[African].keys():', data['African'].keys())
    
    # attributes = ['roll', 'yaw', 'pitch', 'angle', 'Xfrontal', 'Yfrontal']
    attributes = ['roll', 'yaw', 'pitch']

    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig, axs = plt.subplots(len(races), len(attributes), figsize=(12, 10))
    # axs = axs.flatten()

    for i, race in enumerate(races):
        race_data = data[race]
        for j, attr in enumerate(attributes):
            ax = axs[i, j]
            ax.set_title(race + ' ' + attr.capitalize() + ' Histogram')
            ax.hist(race_data[attr], bins=bins, density=True, alpha=0.5)

            if attr in ['roll', 'yaw']:
                ax.set_xlim(left=-40, right=40)
                if attr in ['roll']:
                    ax.set_ylim(bottom=0, top=0.5)
                if attr in ['yaw']:
                    ax.set_ylim(bottom=0, top=0.1)
            elif attr in ['pitch']:
                ax.set_xlim(left=0, right=10)
                ax.set_ylim(bottom=0, top=1)

        '''
        for i, attr in enumerate(attributes):
            ax = axs[i]
            ax.set_title(attr.capitalize() + ' Histogram')
            for race, race_data in data.items():
                ax.hist(race_data[attr], bins=bins, density=True, alpha=0.5, label=race)
                ax.set_ylim(bottom=ylim[0], top=ylim[1])

                if attr in ['roll', 'yaw']:
                    ax.set_xlim(left=-40, right=40)
                elif attr in ['pitch']:
                    ax.set_xlim(left=0, right=10)

            ax.legend()
        '''

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_samples_one_pair(pair, path_save_sample):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    # title = ''
    # plt.suptitle(title)
    
    for i in range(2):
        sample_key = f'sample{i}'
        sample = pair[sample_key]
        ax = axes[i,0]
        image = sample['image_rgb_lmk'] if 'image_rgb_lmk' in sample.keys() else sample['image']
        # ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        ax.imshow(image)
        sub_path = '/'.join(sample['path'].split('/')[-3:])
        ax.set_title(sub_path)

    plt.tight_layout()
    plt.savefig(path_save_sample)
    plt.close()


def find_correct_wrong_predict_labels(true_labels, pred_labels):
    assert len(true_labels) == len(pred_labels), "Lengths of true_labels and pred_labels must be the same"
    TP_indices = []
    TN_indices = []
    FP_indices = []
    FN_indices = []
    
    for i in range(len(true_labels)):
        if true_labels[i] == 1 and pred_labels[i] == 1:
            TP_indices.append(i)
        elif true_labels[i] == 0 and pred_labels[i] == 0:
            TN_indices.append(i)
        elif true_labels[i] == 0 and pred_labels[i] == 1:
            FP_indices.append(i)
        elif true_labels[i] == 1 and pred_labels[i] == 0:
            FN_indices.append(i)

    return TP_indices, TN_indices, FP_indices, FN_indices


def filter_list_by_indices(pairs_list, keep_idxs):
    keep_list = [None] * len(keep_idxs)
    for i, keep_idx in enumerate(keep_idxs):
        keep_list[i] = pairs_list[keep_idx]
    return keep_list


def count_attributes_save_charts(attrib_pairs, output_path):
    os.makedirs(output_path, exist_ok=True)

    #############################
    print('Counting facial attributes')
    race_facial_attributes_count = count_facial_attributes_by_race(attrib_pairs)
    # print('race_facial_attributes_count:', race_facial_attributes_count)

    path_facial_attributes_chart_file = os.path.join(output_path, 'race_face_attributes_count.png')
    print(f'Saving chart \'{path_facial_attributes_chart_file}\'')
    save_bar_subplots(race_facial_attributes_count, path_facial_attributes_chart_file)


    #############################
    gender_age_count = count_gender_age_by_race(attrib_pairs)
    # print('gender_age_count:', gender_age_count)
    path_face_gender_age_chart_file = os.path.join(output_path, f'gender_age_count.png')
    print(f'Saving chart \'{path_face_gender_age_chart_file}\'')
    plot_gender_age_histograms(gender_age_count, path_face_gender_age_chart_file)


    #############################
    print('Counting face poses')
    race_face_pose_count = count_face_pose_by_race(attrib_pairs)
    # print('race_face_pose_count:', race_face_pose_count)
    # print('race_face_pose_count.keys():', race_face_pose_count.keys())
    # print('race_face_pose_count[\'African\'].keys():', race_face_pose_count['African'].keys())
    # sys.exit(0)

    bins, ylim = 50, (0,1)
    path_face_pose_chart_file = os.path.join(output_path, f'race_face_pose_count_bins={bins}_ylim='+str(ylim).replace(' ','')+'.png')
    print(f'Saving chart \'{path_face_pose_chart_file}\'')
    plot_face_pose_histograms(race_face_pose_count, bins, ylim, path_face_pose_chart_file)
    


""" Function for gender detection,age detection and """            
def main(args):
    output_path = os.path.join(os.path.dirname(__file__), 'results_' + args.load_scores_labels.split('/')[-2]+ '_on_dataset_bupt')
    # os.makedirs(output_path, exist_ok=True)

    print(f'\nLoading protocol pairs: \'{args.protocol}\'')
    protocol_pairs, true_labels, race_count, subject_count, sample_count = count_verification_protocol_stats(args.protocol)
    print('len(protocol_pairs):', len(protocol_pairs))
    # sys.exit(0)

    # all_pairs = adjust_paths(all_pairs, args.img_path, '.png')

    # args.start_idx_pair = int(max(args.start_idx_pair, 0))
    # args.start_idx_pair = int(min(args.start_idx_pair, len(all_pairs)-1))
    # all_pairs = all_pairs[args.start_idx_pair:]

    print(f'\nLoading computed attributes: \'{args.load_attributes}\'')
    attrib_pairs = load_attributes(args.load_attributes)
    print('len(attrib_pairs):', len(attrib_pairs))
    # print('attrib_pairs[-1]:', attrib_pairs[-1])
    # sys.exit(0)

    print(f'\nLoading pred scores and labels: \'{args.load_attributes}\'')
    pred_scores, pred_labels = load_predict_scores_labels(args.load_scores_labels)
    print(f'len(pred_scores): {len(pred_scores)}    len(pred_labels): {len(pred_labels)}')

    TP_indices, TN_indices, FP_indices, FN_indices = find_correct_wrong_predict_labels(true_labels, pred_labels)
    right_idx = TP_indices + TN_indices
    mistake_idx = FP_indices + FN_indices

    rights_attrib_pairs = filter_list_by_indices(attrib_pairs, right_idx)
    mistakes_attrib_pairs = filter_list_by_indices(attrib_pairs, mistake_idx)
    assert len(rights_attrib_pairs)+len(mistakes_attrib_pairs) == len(attrib_pairs)
    # attrib_pairs = mistake_attrib_pairs

    count_attributes_save_charts(rights_attrib_pairs, output_path+'/rights')
    count_attributes_save_charts(mistakes_attrib_pairs, output_path+'/mistakes')

    print('Finished!\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict the face attribution of one input image")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--pad', type=float, nargs='+',
                                 help="pad (left,top,right,bottom) for face detection region")
    parser.add_argument('--model-load-prefix', dest = 'model_load_prefix', type=str, default='../model/lightened_moon/lightened_moon_fuse',
                        help='the prefix of the model to load')

    parser.add_argument('--protocol', type=str, default='/datasets2/2nd_frcsyn_cvpr2024/comparison_files/comparison_files_2/sub-tasks_1.1_1.2_1.3/bupt_comparison.txt')
    parser.add_argument('--img-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112')
    # parser.add_argument('--detect-face', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112')
    parser.add_argument('--start-idx-pair', type=int, default=0, help='Min=0, Max=7999')
    parser.add_argument('--load-attributes', type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_AIML-Human-Attributes-Detection-with-Facial-Feature-Extraction/results/analysis_dataset_bupt/attributes_dataset_bupt.pkl')
    parser.add_argument('--load-scores-labels', type=str, default='/home/bjgbiesseck/GitHub/BOVIFOCR_insightface_2nd_FRCSyn_CVPR2024/work_dirs/idiffface-uniform_sdfr2024_r50/model_target=bupt_frcsyn_scores_labels_thresh=0.22.txt')
    parser.add_argument('--save-pair-imgs', action='store_true')

    args = parser.parse_args()

    logging.info(args)
    main(args)
