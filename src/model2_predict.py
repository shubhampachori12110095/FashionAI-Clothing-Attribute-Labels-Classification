# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.utils import shuffle
import keras
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.regularizers import *
from keras import initializers
from keras.preprocessing.image import *
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import multiprocessing
from collections import Counter #
import datetime



def predict2(testb_data_root_path = '/data/Attributes/Round2b/',output_csv_root_path = '../'):


    df_test = pd.read_csv(testb_data_root_path+'Tests/question.csv', header=None)
    df_test.columns = ['image_id', 'class', 'x']
    del df_test['x']


    ##########attributes setting##
    classes = ['collar_design_labels', 'lapel_design_labels', 'neck_design_labels',  'neckline_design_labels',
    'coat_length_labels', 'pant_length_labels', 'skirt_length_labels','sleeve_length_labels']
    design_classes = ['collar_design_labels', 'lapel_design_labels', 'neck_design_labels',  'neckline_design_labels']
    design_label_count = {'collar_design_labels': 5,'lapel_design_labels': 5,'neck_design_labels': 5,'neckline_design_labels': 10}

    length_classes = ['coat_length_labels', 'pant_length_labels', 'skirt_length_labels','sleeve_length_labels']
    length_label_count = {'coat_length_labels': 8,'pant_length_labels': 6,'skirt_length_labels': 6,'sleeve_length_labels': 9}
    length_label_count['coat_length_labels_pingpu'] = length_label_count['coat_length_labels']
    length_label_count['pant_length_labels_pingpu'] = length_label_count['pant_length_labels']
    length_label_count['skirt_length_labels_pingpu'] = length_label_count['skirt_length_labels']
    length_label_count['sleeve_length_labels_pingpu'] = length_label_count['sleeve_length_labels']
    length_classes.append('coat_length_labels_pingpu')
    length_classes.append('pant_length_labels_pingpu')
    length_classes.append('skirt_length_labels_pingpu')
    length_classes.append('sleeve_length_labels_pingpu')



    width = 499
    ##########model############
    base_model1 = InceptionResNetV2(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')
    input_tensor = Input((width, width, 3))
    x = input_tensor
    x = Lambda(tf.keras.applications.inception_v3.preprocess_input)(x)
    x = base_model1(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in design_label_count.items()]
    design_model = Model(input_tensor, x)

    design_model.load_weights('../models/model2/B_ception_design__training_width499__by_finetuning_based_on_fai_a_20180529_043659.h5',by_name=True)
    print('design_model装载完成')
    #####length preprcessing model and preidict model########
    pre_width = 224
    length_cls_label_count = {'coat_length_labels': 2, 'pant_length_labels': 2, 'skirt_length_labels': 2, 'sleeve_length_labels': 2}

    base_model2 = ResNet50(weights='imagenet', input_shape=(pre_width, pre_width, 3), include_top=False, pooling='avg')
    #InceptionResNetV2,Xception
    input_tensor = Input((pre_width, pre_width, 3))
    x = input_tensor
    x = Lambda(keras.applications.resnet50.preprocess_input)(x)
    x = base_model2(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in length_cls_label_count.items()]
    preprocess_model = Model(input_tensor, x)
    preprocess_model.load_weights('../models/model2/preprocess/B_ResNet50_cls_for_person_and_ping_pu_hard_sample_20180522_104910')
    print('preprocess_model装载完成')
    length_model = load_model('../models/model2/B_ception_length_20180531_174426',{'imagenet_utils':imagenet_utils})
    print('length_model装载完成')
    print()
    print('开始预测:')
    ##########functions#########
    def sqr_padding(im):
        desired_size=0
        old_size = im.shape[:2]  # old_size is in (height, width) format
        if old_size[0] > old_size[1]:
            desired_size=old_size[0]
        else:
            desired_size = old_size[1]
        #ratio = float(desired_size) / max(old_size)
        #new_size = tuple([int(x * ratio) for x in old_size])
        # new_size should be in (width, height) format
        #im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        return new_im

    #test:do crops for better result
    #do crops for better result
    def slb_rotate(image, angle, center=None, scale=1.0): #
        (h, w) = image.shape[:2]
        if center is None: #3
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(image, M, (w, h),borderMode=cv2.BORDER_CONSTANT,borderValue=[255,255,255])
        return rotated

    def design_crops_rotate_preprocessor(image,width, height, horiz=True, inter=cv2.INTER_AREA): #crps6
        crops = []
        (h, w) = image.shape[:2] #530
        # 四角裁剪
        coords = [[0, 0, width, height],
                [w - width, 0, w, height]]
        dW = int(0.5 * (w - width))
        dH = int(0.5 * (h - height))
        # 中心裁剪
        coords.append([dW, 0, w - dW, height])
        # 处理四角裁剪和中心裁剪
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (width, height), interpolation=inter)
            crops.append(crop)
        # 镜像裁剪
        if horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
        rotated_im = []
        for img in crops:
            angle = int((-1 + 2*np.random.random())*25) #-25~25
            rotated_tmp = slb_rotate(image=img,angle=angle)
            rotated_im.append(rotated_tmp)
        crops.extend(rotated_im)
        return np.array(crops)

    def crops_rotate_preprocessor(image,width, height, horiz=True, inter=cv2.INTER_AREA): #crops10
        crops = []
        (h, w) = image.shape[:2] #530
        # 四角裁剪
        coords = [[0, 0, width, height],
                [w - width, 0, w, height],
                [w - width, h - height, w, h],
                [0, h - height, width, h]]
        dW = int(0.5 * (w - width))
        dH = int(0.5 * (h - height))
        # 中心裁剪
        coords.append([dW, dH, w - dW, h - dH])
        # 处理四角裁剪和中心裁剪
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (width, height), interpolation=inter)
            crops.append(crop)
        # 镜像裁剪
        if horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
        rotated_im = []
        for img in crops:
            angle = int((-1 + 2*np.random.random())*25) #-25~25
            rotated_tmp = slb_rotate(image=img,angle=angle)
            rotated_im.append(rotated_tmp)
        crops.extend(rotated_im)
        return np.array(crops)

    def crops10_preprocessor(image,width, height, horiz=True, inter=cv2.INTER_AREA): #crops10
        crops = []
        (h, w) = image.shape[:2] #530
        # 四角裁剪
        coords = [[0, 0, width, height],
                [w - width, 0, w, height],
                [w - width, h - height, w, h],
                [0, h - height, width, h]]
        dW = int(0.5 * (w - width))
        dH = int(0.5 * (h - height))
        # 中心裁剪
        coords.append([dW, dH, w - dW, h - dH])
        # 处理四角裁剪和中心裁剪
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (width, height), interpolation=inter)
            crops.append(crop)
        # 镜像裁剪
        if horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)


    ##########predict############
    tmp_df = {}

    print('model2\'s design_model:predict design')
    cnt = 0
    #y_crops_pre = [np.zeros((n_test, design_label_count[x])) for x in label_count.keys()]
    for idx in range(4):
        print()
        cur_class = design_classes[idx]
        df_load = df_test[(df_test['class'] == cur_class)].copy()
        df_load.reset_index(inplace=True)
        del df_load['index']
        image_id = df_load.image_id

        predictions = []

        for im_path in image_id:

            img = sqr_padding(cv2.imread(testb_data_root_path + im_path))
            img_tmp = cv2.resize(img, (520, 520), interpolation=cv2.INTER_CUBIC)
            crops_tmp = crops_rotate_preprocessor(image=img_tmp, width=width, height=width)
            crops10 = crops_tmp[:, :, :, ::-1]  # BGR->RGB
            # crops10 = np.asarray(crops10, dtype=np.float32)
            pred = design_model.predict(crops10)
            probs = pred[idx].mean(axis=0)  # will auto clear betch dimesion
            str_probs = ';'.join(np.char.mod('%.4f', probs))  # probs is []type!!!!
            predictions.append(str_probs)
            # print(str_probs)
            cnt = cnt + 1
            if cnt % 100 == 0:
                print('###########')
                print('Processed Samples Nums: {}'.format(cnt))
                print('###########')
            print('y_crops_pre:', ';'.join(np.char.mod('%.4f', probs)), ' ', im_path.split('/')[-1])
        df_load['result'] = predictions
        print(cur_class, ':', len(df_load))
        tmp_df[cur_class] = df_load.copy()

    print('Model2-design_model completes the prediction of design:on the {0}of samples'.format(cnt))









    print('############################################')
    print()
    print('next is to predict length using Model2-length_model')

    print('model2\'s length_model:predict length')
    cnt = 0
    for idx in range(4):
        print()
        cur_class = length_classes[idx]
        df_load = df_test[(df_test['class'] == cur_class)].copy()
        df_load.reset_index(inplace=True)
        del df_load['index']
        image_id = df_load.image_id

        predictions = []
        #ii = 0
        flg = ''
        for im_path in image_id:

            img = sqr_padding(cv2.imread(testb_data_root_path + im_path))
            # preprocess
            pre_process_img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_CUBIC)
            pre_image = pre_process_img.copy()
            image1 = pre_image[:, :, ::-1]
            image2 = np.expand_dims(image1, axis=0)
            label_index = length_classes.index(cur_class)
            pre_y_pred = preprocess_model.predict(image2, batch_size=1)
            pre_probs = pre_y_pred[label_index]
            pre_idx = np.array(pre_probs).argmax(axis=-1)
            if pre_idx == 1: #person
                flg_int = 0
            else: #pingpu
                flg_int = 1

            # predict
            if flg_int > 0.5:  # ping_pu for crops
                flg = 'ping_pu'
                # crops:
                img_tmp = cv2.resize(img, (524, 524), interpolation=cv2.INTER_CUBIC)
                crops_tmp = crops_rotate_preprocessor(image=img_tmp, width=width, height=width)
                crops10 = crops_tmp[:, :, :, ::-1]  # BGR->RGB
                # crops10 = np.asarray(crops10, dtype=np.float32)
                pred = length_model.predict(crops10)
                probs = pred[idx + 4].mean(axis=0)
                str_probs = ';'.join(np.char.mod('%.5f', probs))
            else:  # non_ping_pu for crops use pre-width = 512

                flg = 'person'

                # crops:
                img_tmp = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                crops_tmp = crops10_preprocessor(image=img_tmp, width=width, height=width)
                crops10 = crops_tmp[:, :, :, ::-1]  # BGR->RGB
                # crops10 = np.asarray(crops10, dtype=np.float32)
                pred = length_model.predict(crops10)
                probs = pred[idx].mean(axis=0)
                str_probs = ';'.join(np.char.mod('%.5f', probs))

            predictions.append(str_probs)
            print(flg, ': ', str_probs, "  ", im_path.split('/')[-1])
            #ii = ii + 1
            cnt = cnt + 1
            if cnt % 100 == 0:
                print('###########')
                print('Processed Samples Nums: {}'.format(cnt))
                print('###########')
        #del df_load['added_label']
        df_load.reset_index(inplace=True)
        del df_load['index']
        df_load['result'] = predictions
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

    result.to_csv('../output/model2_result.csv', index=None, header=None)
    print('model2 predicts the {} samples'.format(len(result)))

    ###result1###
    return result
