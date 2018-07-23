# -*- coding: UTF-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import cv2
from sklearn.model_selection import train_test_split
import matplotlib


from keras.utils import np_utils
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

from fashionAI.config import config
from fashionAI.Utils.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from fashionAI.Utils.preprocessing.simplepreprocessor import SimplePreprocessor
from fashionAI.Utils.preprocessing.meanpreprocessor import MeanPreprocessor
from fashionAI.Utils.preprocessing.patchpreprocessor import PatchPreprocessor
from fashionAI.Utils.preprocessing.croppreprocessor import CropPreprocessor
from fashionAI.callbacks.trainingmonitor import TrainingMonitor
from fashionAI.Utils.io.datagenerator import DataGenerator
from fashionAI.nn.inceptionresnet_v2 import InceptionResnetV2



def predict1(testb_data_root_path = '/data/Attributes/Round2b/', output_csv_root_path = '../'):

    df_test = pd.read_csv(testb_data_root_path+'Tests/question.csv', header=None)
    df_test.columns = ['image_id', 'class', 'x']
    del df_test['x']


    ##########attributes setting##
    classes = ['collar_design_labels', 'lapel_design_labels', 'neck_design_labels',  'neckline_design_labels',
    'coat_length_labels', 'pant_length_labels', 'skirt_length_labels','sleeve_length_labels']
    design_classes = ['collar_design_labels', 'lapel_design_labels', 'neck_design_labels',  'neckline_design_labels']
    design_label_count = {'collar': 5,'lapel': 5,'neck': 5,'neckline': 10}
    length_classes = ['coat_length_labels', 'pant_length_labels', 'skirt_length_labels','sleeve_length_labels']
    length_label_count = {'coat': 8,'pant': 6,'skirt': 6,'sleeve': 9}

    ##########model############
    incepres1 = InceptionResnetV2(500, 500, design_label_count, weight_decay=0.001)
    design_model = incepres1.build_net()

    incepres2 = InceptionResnetV2(500, 500, length_label_count, weight_decay=0.001)
    length_model = incepres2.build_net()


    design_model.load_weights('../models/model1/multitask_design_final.h5')
    length_model.load_weights('../models/model1/multitask_length_final.h5')

    ##########functions#########

    pre_resize = SimplePreprocessor(530, 530) #use opecv to resize in the width of 530*530
    cp = CropPreprocessor(500, 500) #when 10crops, 530*530 -> 500*500
    iap = ImageToArrayPreprocessor()  # transform data format

    design_means = json.loads(open('./model1_mean/multitask_mean_design.json').read())
    length_means = json.loads(open('./model1_mean/multitask_mean_length.json').read())

    design_mp = MeanPreprocessor(design_means['R'], design_means['G'], design_means['B'])
    length_mp = MeanPreprocessor(length_means['R'], length_means['G'], length_means['B'])

    val_aug = ImageDataGenerator(rescale=1./255)
    ##########predict############
    tmp_df = {}

    print('model1\'s design_model:predict design')
    cnt = 0
    for idx in range(4):
        print()
        cur_class = design_classes[idx]
        df_load = df_test[(df_test['class'] == cur_class)].copy()
        df_load.reset_index(inplace=True)
        del df_load['index']

        X_test = [testb_data_root_path + test_img for test_img in df_load['image_id']]
        print('design samples num-{0}:'.format(cur_class),len(X_test))

        print('[INFO] predicting on test data (with crops)...')
        print()
        testGen = DataGenerator((X_test, None), 32, aug=val_aug, preprocessors=[pre_resize, design_mp])

        predictions = []

        for (i, images) in enumerate(testGen.generator(training=False, passes=1)):
            if i % 10 == 0:
                print('{}_test_batch_num/epochs:{}/{}'.format(cur_class, i, int(len(X_test) / 32)))
            for image in images:
                crops = cp.preprocess(image)
                crops = np.array([iap.preprocess(c) for c in crops], dtype='float32')
                pred = design_model.predict(crops)
                predictions.append(pred[idx].mean(axis=0))

        result = []

        for i in range(len(X_test)):
            tmp_list = predictions[i]
            tmp_result = ''
            for tmp_ret in tmp_list:
                tmp_result += '{:.4f};'.format(tmp_ret)
            print(X_test[i].split('/')[-1],' predicted: ', tmp_result[:-1])
            result.append(tmp_result[:-1])
            cnt = cnt +1
        df_load['result'] = result
        print(len(df_load))
        tmp_df[cur_class] = df_load.copy()

    print('Model1-design_model completes the prediction of design:on the {0}of samples'.format(cnt))

    print('############################################')
    print()
    print('next is to predict length using Model1-lenght_model')

    print('model1\'s length_model:predict length')
    cnt = 0
    for idx in range(4):
        print()
        cur_class = length_classes[idx]
        df_load = df_test[(df_test['class'] == cur_class)].copy()
        df_load.reset_index(inplace=True)
        del df_load['index']

        X_test = [testb_data_root_path + test_img for test_img in df_load['image_id']]
        print('length samples num-{0}:'.format(cur_class), len(X_test))

        print('[INFO] predicting on test data (with crops)...')
        print()
        testGen = DataGenerator((X_test, None), 32, aug=val_aug, preprocessors=[pre_resize, length_mp])

        predictions = []

        for (i, images) in enumerate(testGen.generator(training=False, passes=1)):
            if i%10 ==0:
                print('{}_test_batch_num/epochs:{}/{}'.format(cur_class, i, int(len(X_test) / 32)))
            for image in images:
                crops = cp.preprocess(image)
                crops = np.array([iap.preprocess(c) for c in crops], dtype='float32')
                pred = length_model.predict(crops)
                predictions.append(pred[idx].mean(axis=0))

        result = []

        for i in range(len(X_test)):
            tmp_list = predictions[i]
            tmp_result = ''
            for tmp_ret in tmp_list:
                tmp_result += '{:.4f};'.format(tmp_ret)
            print(X_test[i].split('/')[-1], ' predicted: ', tmp_result[:-1])
            result.append(tmp_result[:-1])
            cnt = cnt + 1
        df_load['result'] = result
        print(len(df_load))
        tmp_df[cur_class] = df_load.copy()

    print('Model1-length_model completes the prediction of length:on the {0}of samples'.format(cnt))
    print()
    print('Complete!')

    ###########output csv######
    df_result = []

    for cur in classes:
        tmp = tmp_df[cur]
        tmp.reset_index(inplace=True)
        del tmp['index']
        df_result.append(tmp)
    for i in df_result:
        i.columns = ['image_id', 'class', 'label']
    result = pd.concat(df_result)

    result.to_csv('../output/model1_result.csv', index=None, header=None)
    print('model1 predicts the {} samples'.format(len(result)))

    ###result1###
    return result
