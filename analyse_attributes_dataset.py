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
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt

import model.emotion.detectemotion as ime
from mxnet_moon.lightened_moon import lightened_moon_feature

from Face_Pose.pose_detection_retinaface import *
from deepface import DeepFace

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


def find_files(folder_path, extensions=['.jpg', '.png']):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for ext in extensions:
            pattern = os.path.join(root, '*' + ext)
            matching_files = glob.glob(pattern)
            image_paths.extend(matching_files)
    return sorted(image_paths)


def find_sample_by_substring(paths_list, substring):
    for i, path in enumerate(paths_list):
        if substring in path:
            return i
    return -1


def save_attributes_bin(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def save_attributes_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_attributes_bin(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def group_attributes_by_race(facial_data):
    grouped_data = {}
    for i, data in enumerate(facial_data):
        print(f'{i}/{len(facial_data)}', end='\r')
        race = data['race']['dominant_race']
        if race not in grouped_data:
            grouped_data[race] = {
                'gender': [],
                'age': [],
                'emotion': [],
                'roll': [],
                'yaw': [],
                'pitch': [],
                'angle': [],
                'Xfrontal': [],
                'Yfrontal': []
            }
        
        try:
            grouped_data[race]['gender'].append(data['gender'])
            grouped_data[race]['age'].append(data['age'])
            grouped_data[race]['emotion'].append(data['emotion'])
            grouped_data[race]['roll'].append(data['roll'])
            grouped_data[race]['yaw'].append(data['yaw'])
            grouped_data[race]['pitch'].append(data['pitch'])
            grouped_data[race]['angle'].append(data['angle'])
            grouped_data[race]['Xfrontal'].append(data['Xfrontal'])
            grouped_data[race]['Yfrontal'].append(data['Yfrontal'])
        except KeyError:
            continue

    print('')

    return grouped_data


'''
def save_attribute_histograms(grouped_data):
    # attributes = ['gender', 'age', 'emotion', 'roll', 'yaw', 'pitch', 'angle', 'Xfrontal', 'Yfrontal']
    attributes = ['gender', 'age', 'roll', 'yaw', 'pitch']
    races = list(grouped_data.keys())
    num_races = len(races)
    num_attributes = len(attributes)

    # size=(15, 10)
    size=(25, 16)
    fig, axs = plt.subplots(num_races, num_attributes, figsize=size)
    
    for i, race in enumerate(races):
        for j, attribute in enumerate(attributes):
            ax = axs[i, j] if num_races > 1 and num_attributes > 1 else axs[j] if num_races == 1 else axs[i]
            ax.hist(grouped_data[race][attribute], bins=20, alpha=0.7)
            ax.set_title(f'{attribute} - {race}')
            ax.set_xlabel(attribute)
            ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('attribute_histograms.png')
    plt.close()
'''


def save_attribute_histograms(grouped_data, path_figure):
    # races = list(grouped_data.keys())
    races = ['black', 'asian', 'white', 'indian', 'latino hispanic', 'middle eastern']

    # attributes = ['gender', 'age', 'emotion', 'roll', 'yaw', 'pitch', 'angle', 'Xfrontal', 'Yfrontal']
    attributes = ['gender', 'age', 'roll', 'yaw', 'pitch']

    possible_values = {}
    possible_values['gender'] = ['Male', 'Female']
    possible_values['age'] = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    possible_values['emotion'] = ['happy', 'neutral', 'surprise', 'angry', 'fear', 'sad', 'disgust']

    num_races = len(races)
    num_attributes = len(attributes)

    fig, axs = plt.subplots(num_races, num_attributes, figsize=(22, 19))

    for i, race in enumerate(races):
        for j, attribute in enumerate(attributes):
            print(f'Mounting figure with all histograms - race={race}({i}/{len(races)}) - attrib={attribute}({j}/{len(attributes)})', end='\r')
            ax = axs[i, j] if num_races > 1 and num_attributes > 1 else axs[j] if num_races == 1 else axs[i]
            data = grouped_data[race][attribute]
            counts = data
            # if attribute in ['gender', 'age', 'emotion']:
            if attribute in list(possible_values.keys()):
                counts = {poss_value:0 for poss_value in possible_values[attribute]}
                for value in data:
                    counts[value] += 1
                counts = [counts[key] for key in counts.keys()]
                x = np.arange(len(possible_values[attribute]))
                ax.bar(x, counts)
                ax.set_xticks(x)
                ax.set_xticklabels(possible_values[attribute], rotation=45, ha='right')
            else:
                ax.hist(counts, bins=50, density=True, alpha=0.7)
                if attribute in ['roll', 'yaw']:
                    ax.set_xlim(left=-40, right=40)
                    if attribute in ['roll']:
                        ax.set_ylim(bottom=0, top=0.5)
                    if attribute in ['yaw']:
                        ax.set_ylim(bottom=0, top=0.1)
                elif attribute in ['pitch']:
                    ax.set_xlim(left=0, right=10)
                    ax.set_ylim(bottom=0, top=1)
            ax.set_title(f'{attribute} - {race}')
            ax.set_xlabel(attribute)
            ax.set_ylabel('Proportion')
    print('')

    plt.tight_layout()
    # path_figure = 'attribute_histograms.png'
    print(f'Saving figure \'{path_figure}\'')
    plt.savefig(path_figure)
    plt.close()


def extract_facial_attributes(args):
    symbol = lightened_moon_feature(num_classes=40, use_fuse=True)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    _, arg_params, aux_params = mx.model.load_checkpoint('model/lightened_moon/lightened_moon_fuse', 82)

    args.input = args.input.rstrip('/').rstrip(' ')
    if args.output == '':
        args.output = args.input + '_FACE_ATTRIB'

    print(f'\nSearching images in \'{args.input}\'')
    types = ('.jpg', '.png')
    image_path_list = find_files(args.input, types)
    if len(image_path_list) == 0:
        raise Exception('No input images found in \''+ args.input +'\'')
    print(f'Found {len(image_path_list)} images')
    
    start_idx = 0
    end_idx = len(image_path_list)
    if args.start_string != '':
        print('\nSearching string \''+args.start_string+'\'')
        found_string_idx = find_sample_by_substring(image_path_list, args.start_string)
        if found_string_idx > -1:
            print('String \''+args.start_string+'\' found at index '+str(found_string_idx)+'\n')
            start_idx = found_string_idx
        else:
            print('Could not find string \''+args.start_string+'\'!\n')

    image_path_list = image_path_list[start_idx:end_idx]
    total_num = len(image_path_list)

    print('')
    for i, path in enumerate(image_path_list):
        start_time = time.time()
        
        print('i:', str(i)+'/'+str(total_num), '    path:', path)
        face_attrib = {}
        
        image = cv2.imread(path)
        # img = cv2.imread(path, -1)
        # face_attrib['image'] = image


        # print("#====Detected Age and Gender====#")
        # gender, age = genderAge(image, faceBox)
        gender, age = genderAge(image)
        face_attrib['gender'] = gender
        face_attrib['age'] = age
        print('    gender: ', gender, '    age:', age)

        # Predict emotions in the image
        # print("#====Detected Emotion===========#")
        # emlist = ed.emotionalDet(path, faceBox)
        emlist = ed.emotionalDet(path)
        emlist = emlist[0]
        print('    emotion: ', emlist)
        face_attrib['emotion'] = emlist

        # Detect the facial attributes using mxnet
        # crop face area
        # left = faceBox[0]
        # width = faceBox[2] - faceBox[0]
        # top = faceBox[1]
        # height =  faceBox[3] - faceBox[1]
        # right = faceBox[2]
        # bottom = faceBox[3]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # pad = [0.25, 0.25, 0.25, 0.25] if args.pad is None else args.pad
        # left = int(max(0, left - width*float(pad[0])))
        # top = int(max(0, top - height*float(pad[1])))
        # right = int(min(gray.shape[1], right + width*float(pad[2])))
        # bottom = int(min(gray.shape[0], bottom + height*float(pad[3])))
        # gray = gray[left:right, top:bottom]
        # resizing image and increasing the image size
        gray = cv2.resize(gray, (args.size, args.size))/255.0
        img = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
        # get image parameter from mxnet
        arg_params['data'] = mx.nd.array(img, devs)
        exector = symbol.bind(devs, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exector.forward(is_train=False)
        exector.outputs[0].wait_to_read()
        output = exector.outputs[0].asnumpy()
        # print('output:', output)
        # sys.exit(0)
        # 40 facial attributes
        text = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald", "Bangs","Big_Lips","Big_Nose",
                "Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee",
                "Gray_Hair", "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
                "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
                "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
        
        #Predict the results
        # pred = np.ones(40)
        # create a list based on the attributes generated.
        attrDict = {}
        detectedAttributeList = []
        for j in range(40):
            # attr = text[i].rjust(20)
            attr = text[j]
            if output[0][j] < 0:
                attrDict[attr] = 'No'
            else:
                attrDict[attr] = 'Yes'
                detectedAttributeList.append(text[j])

        # print("#====Detected Attributes========#")
        face_attrib['attrDict'] = attrDict
        face_attrib['attributes'] = detectedAttributeList
        print('    detectedAttributeList:', detectedAttributeList)


        # Detect landmarks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# convert to rgb
        landmarks, bboxes, scores = detect_faces(image_rgb,image_shape_max=640)
        # print('landmarks:', landmarks)
        # print('bboxes:', bboxes)
        # print('scores:', scores)
        if len(landmarks) > 0 and len(bboxes) > 0:
            lmarks = np.transpose(landmarks)
            bbs = bboxes.copy()
            bb, lmarks_5 = one_face(image_rgb, bbs, lmarks)
            image_rgb_lmk = draw_landmarks(image_rgb, bb, lmarks_5)
            # face_attrib['image_rgb_lmk'] = image_rgb_lmk

            roll = find_roll(lmarks_5)
            yaw = find_yaw(lmarks_5)
            pitch = find_pitch(lmarks_5)
            face_attrib['roll'] = roll
            face_attrib['yaw'] = yaw
            face_attrib['pitch'] = pitch
            print(f'    roll: {roll}    yaw: {yaw}    pitch: {pitch}')

            angle, Xfrontal, Yfrontal = find_pose(lmarks_5)
            face_attrib['angle'] = angle
            face_attrib['Xfrontal'] = Xfrontal
            face_attrib['Yfrontal'] = Yfrontal
            print(f'    angle: {angle}    Xfrontal: {Xfrontal}    Yfrontal: {Yfrontal}')


        # ESTIMATE RACE, POSSIBLE VALUES=('asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic')
        # objs = DeepFace.analyze(img_path=image_bgr, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        race = DeepFace.analyze(img_path=image, actions=['race'], enforce_detection=False)
        race = race[0]
        face_attrib['race'] = race
        print('    race:', race)
        
        if not args.dont_save_bin:
            attrib_output_file_bin = os.path.basename(path).split('/')[-1].split('.')[0] + '.pkl'
            dir_output_file = os.path.dirname(path).replace(args.input, args.output)
            path_attrib_output_file_bin = os.path.join(dir_output_file, attrib_output_file_bin)
            os.makedirs(dir_output_file, exist_ok=True)
            print('    bin_output_file:', path_attrib_output_file_bin)
            save_attributes_bin(face_attrib, path_attrib_output_file_bin)
        
        if args.save_text:
            attrib_output_file_text = os.path.basename(path).split('/')[-1].split('.')[0] + '.json'
            dir_output_file = os.path.dirname(path).replace(args.input, args.output)
            path_attrib_output_file_text = os.path.join(dir_output_file, attrib_output_file_text)
            os.makedirs(dir_output_file, exist_ok=True)
            print('    text_output_file:', path_attrib_output_file_text)
            save_attributes_json(face_attrib, path_attrib_output_file_text)

        spent_time = time.time() - start_time
        print('    Spent time: %.2fsec' % (spent_time))
        remaining = total_num-i
        print('    Remaining:', remaining)
        est_time_sec = spent_time*remaining
        est_time_min = est_time_sec / 60
        est_time_hour = est_time_min / 60
        est_time_days = est_time_hour / 24
        print('    Estimated time: %.2fsec, %.2fmin, %.2fhour, %.2fdays' % (est_time_sec, est_time_min, est_time_hour, est_time_days))
        print('----------------')
        # sys.exit(0)


def summarize_results(args):
    args.input = args.input.rstrip('/').rstrip(' ')
    if args.output == '':
        args.output = args.input + '_FACE_ATTRIB_SUMMARY'

    print(f'\nSearching attributes files in \'{args.input}\'')
    types = ('.pkl')
    attrib_path_list = find_files(args.input, types)
    if len(attrib_path_list) == 0:
        raise Exception('No input attributes files found in \''+ args.input +'\'')
    print(f'Found {len(attrib_path_list)} attributes files')

    all_attribs_list = [None] * len(attrib_path_list)
    for i, attrib_path in enumerate(attrib_path_list):
        print(f'Loading attributes files - {i}/{len(attrib_path_list)} - \'{attrib_path}\'', end='\r')
        all_attribs_list[i] = load_attributes_bin(attrib_path)
        # print(all_attribs_list[i]); sys.exit(0)
    print('')

    print(f'Counting facial attributes')
    grouped_data = group_attributes_by_race(all_attribs_list)
    # for k, key in enumerate(grouped_data.keys()):
    #     print(f'grouped_data[\'{key}\'].keys():', grouped_data[key].keys())
    # sys.exit(0)

    dataset_name = '_'.join(args.output.split('/')[-2:])
    path_figure = 'attribute_histograms_'+dataset_name+'.png'
    save_attribute_histograms(grouped_data, path_figure)




def main(args):

    # Extract facial attributes
    if not args.summarize_results:
        extract_facial_attributes(args)
    else:
        summarize_results(args)
    
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
    
    parser.add_argument('--input', type=str, default='/datasets2/2nd_frcsyn_cvpr2024/datasets/synthetic/IDiff-Face/ca-cpd25-synthetic-uniform-10050')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--save-text', action='store_true')
    parser.add_argument('--dont-save-bin', action='store_true')
    parser.add_argument('--start-string', default='', type=str, help='String to find out in image paths and start processing')

    parser.add_argument('--summarize-results', action='store_true')
        
    args = parser.parse_args()
       
    logging.info(args)
    main(args)
