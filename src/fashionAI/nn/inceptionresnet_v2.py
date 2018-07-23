from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.regularizers import *
from keras import initializers


class InceptionResnetV2:

    def __init__(self, width, height, classes_dict, dropout_prob=0.5, weight_decay=0.001):

        self.width = width
        self.height = height
        self.label_count = classes_dict
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay


    def build_net(self):

        cnn_model = InceptionResNetV2(include_top=False, input_shape=(self.height, self.width, 3), weights=None)
        input_tensor = Input((self.height, self.width, 3))
        x = input_tensor
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout_prob)(x)
        x = [Dense(count, activation='softmax', kernel_regularizer=regularizers.l2(self.weight_decay),
                   name=name)(x) for name, count in self.label_count.items()]
        model = Model(input_tensor, x)
        print('[INFO] Load Model Done!')
        return model