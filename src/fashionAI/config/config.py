IMAGES_PATH = '/home/bo/dataset/fashionAI/base'

DATA_TYPE = ['collar', 'neckline', 'skirt', 'sleeve',
             'neck', 'coat', 'lapel', 'pant']
NUM_CLASSES = [5, 10, 6, 9, 5, 8, 5, 6]
DATASET_MEAN = ['./output/'+ data +'_mean.json' for data in DATA_TYPE]
VAL_PARTTITIONAL = 0.1

TRAIN_HDF5 = ['../data/' + data + '_train.hdf5' for data in DATA_TYPE]
VAL_HDF5 = ['../data/' + data + '_val.hdf5' for data in DATA_TYPE]


MODEL_PATH = 'output/'


OUTPUT_PATH = 'output'

