# Generator
# 接受numpy数组形式的数据，返回batchsize大小的处理过的数据


import numpy as np
import cv2


class DataGenerator:

    def __init__(self, data, batchSize=32, preprocessors=None, aug=None):

        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        # self.binarize = bianrize
        # self.classes = classes
        self.data = data
        self.numImages = len(data[0])

    def generator(self, training=True, passes=np.inf):

        epochs = 0
        while epochs < passes:

            for i in np.arange(0, self.numImages, self.batchSize):

                images = self.data[0][i: i + self.batchSize]

                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        image = cv2.imread(image)
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)

                    images = np.array(procImages)

                if self.aug is not None:
                    images = next(self.aug.flow(images, batch_size=self.batchSize, shuffle=False))

                if training:
                    labels = [label[i: i + self.batchSize] for label in self.data[1]]
                    yield (images, labels)

                else:
                    yield images

            epochs += 1