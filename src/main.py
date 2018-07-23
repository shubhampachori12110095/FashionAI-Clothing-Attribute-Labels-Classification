# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import datetime
import argparse
import os

import model1_predict
import model2_predict

# set parameters
parser = argparse.ArgumentParser(description="use model1 to predict design and length attributes",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('testb_data_root_path', type=str, default='/home/slb/Desktop/data/Attributes/Round2b/',
                    help='the root path of the testb data')
parser.add_argument('output_csv_root_path', type=str, default='../',
                    help='the root path of the CSV output')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
args = parser.parse_args()


#########setting############
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
testb_data_root_path = args.testb_data_root_path
output_csv_root_path = args.output_csv_root_path

print('testb_data_root_path:' ,testb_data_root_path)
print('output_csv_root_path:' ,output_csv_root_path)
print('GPU ID:' ,args.gpus)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# testb_data_root_path = '/home/slb/Desktop/data/Attributes/Round2b/'
# output_csv_root_path = '../'

#########main function#######
def main():

    csv_result1 = model1_predict.predict1(testb_data_root_path=testb_data_root_path, output_csv_root_path=output_csv_root_path)
    df_fusion = csv_result1.copy()
    df1 = csv_result1.copy()

    csv_result2 = model2_predict.predict2(testb_data_root_path=testb_data_root_path, output_csv_root_path=output_csv_root_path)
    df2 = csv_result2.copy()

    # csv_result1 = pd.read_csv('../output/model1_result.csv', header=None)
    # csv_result2 = pd.read_csv('../output/model2_result.csv', header=None)
    # 输出融合后的csv文件,替换label列的方法
    # tmp = pd.read_csv('../output/model2_result.csv', header=None)

    print('fusion_reference 样本数:', len(df_fusion))
    df_fusion.columns = ['image_id', 'class', 'label']
    df_fusion.reset_index(inplace=True)
    del df_fusion['index']
    # del df_test['label']
    nn = len(df_fusion)

    # 需要融合的csv文件:
    # df1 = pd.read_csv('../output/model1_result.csv', header=None)
    # df2 = pd.read_csv('../output/model2_result.csv', header=None)


    df1.columns = ['image_id', 'class', 'label']
    df2.columns = ['image_id', 'class', 'label']
    df1.reset_index(inplace=True)
    del df1['index']
    df2.reset_index(inplace=True)
    del df2['index']

    print("核验要融合的结果文件的样本数:")
    print(len(df1))
    print(len(df2))

    # 下面是计算均值法的融合
    print("开始融合...")
    dict_df1 = {}
    dict_df2 = {}

    for i in range(len(df1)):
        list_value = [float(j) for j in df1['label'][i].split(';')]  # [:-1])
        np_value = np.array(list_value)
        dict_df1[df1['image_id'][i]] = np_value
    for i in range(len(df2)):
        list_value = [float(j) for j in df2['label'][i].split(';')]  # [:-1]
        np_value = np.array(list_value)
        dict_df2[df2['image_id'][i]] = np_value

    for i in range(nn):
        prob1 = dict_df1[df_fusion['image_id'][i]]
        prob2 = dict_df2[df_fusion['image_id'][i]]
        if np.argmax(prob1) == np.argmax(prob2):

            if max(prob1) > max(prob2):

                np_mean = prob1 * 0.8 + prob2 * 0.2
            else:

                np_mean = prob1 * 0.2 + prob2 * 0.8
        else:

            np_mean = (prob1 + prob2) / 2.0
        tmp_result = ''
        for tmp_ret in np_mean:
            tmp_result += '{:.4f};'.format(tmp_ret)
        df_fusion['label'][i] = tmp_result[:-1]  ##当时在这有错!!!没有加[:-1]

    # 输出融合后的文件:
    if not os.path.exists(output_csv_root_path):
        os.mkdir(output_csv_root_path)
    out_csv = output_csv_root_path+'result' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    print('输出结果文件:',out_csv)
    df_fusion.to_csv(out_csv, index=None, header=None)
    print("完成!!")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print('main.py start_time:', start_time)

    main()
    print('Complete!Result outputs to AbsolutePath:{}'.format(output_csv_root_path))

    end_time = datetime.datetime.now()
    print('main.py end_time:', end_time)
    print('Running Tatal time:', (end_time - start_time).seconds / 3600, 'h')