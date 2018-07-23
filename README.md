# FashionAI-Clothing-Attribute-Labels-Classification
This is a open project of the Team of ‘010101’ about the clothing attribute labels classification on ‘Fashion AI Global Challenge’. 


了解赛题请戳：[天池赛题](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100068.5678.1.27cd17f0PeeY2M&raceId=231649)

【若要运行main.py测试，需要models目录下模型文件，请给我邮箱发邮件索取。】

### Project描述：
├── models:   model1 model2

├── output

├── README.md

├── run.sh

├── src:    fashionAI main.py model1_mean model1_predict.py model2_predict.py

本团队提交的Project结构如下：

- **models**目录包含针对测试集b的两个单模型，分别为：**model1和model2**。其中，
  - model1是由对design相关的四个属性进行四任务联合训练得到的design-model和对length相关的四个属性进行四任务联合训练得到的length-model组成。
  - model2是由对design相关的四个属性进行四任务联合训练得到的design-model和对length相关的四个属性进行八任务（四任务关于模特的属性分类+四任务关于平铺的属性分类）联合训练得到的length-model组成。另外，model2包含了重要的预处理模型，主要用于对length相关的四个属性进行粗粒度的模特和平铺二分类，用于对model2的length相关属性进行八任务训练。
  - **注**：由于线下预测时会生成中间过程文件，根据官方的说明，其可能将模型用于test-c数据集或其他数据集上的预测，因此，我们把该预处理模型整合到了预测代码中。
- **output**目录在预测结束时会生成model1_result.csv和model2_result.csv，它们分别是由单模型model1和单模型model2预测产出的中间文件，这两个输出文件会进行符合官方限制和要求的简单融合，**最终**产出的预测csv文件默认参数下会保存在Project根目录下，格式如：result20180602_010816.csv
- **src**目录包含预测阶段的代码，其中main.py文件会调用model1_predict模块的predict1函数和model2_predict模块的predict2函数，分别对round2的测试集b进行预测，产出的csv文件将在main.py文件中进一步融合，生成最终的csv的文件到指定目录(默认保存在工程的根目录下)。src目录下的fashionAI和model1_mean目录主要包含了model1在预测阶段所需要的模型结构定义、一些图像处理的模块和所计算的均值文件。
- 其他说明：本Project严格按照官方要求提交相应文件及整合的预测代码，但为保持工程的简洁性，未附加模型的训练代码及线下生成的中间文件，如需提供，请与本团队成员进一步及时沟通和索取。



### 预测说明

#### 1.运行及产出CSV文件方式

/root/Project/run.sh  【**input data directory**】 【**output csv directory**】

其中，[input data directory]为输入评测图片集的根目录，[output csv directory]为产出评测csv文件的目录。example：

sh /Project/run.sh  /data/Attributes/Round2b/  /Project/
 
注：默认参数下，输出csv文件在根目录下,输出的csv文件样式为:'result20180602_010816.csv'

#### 2.预测步骤描述

##### 使用单模型model1进行预测：model1包含由针对design相关属性进行四任务训练的design-model和针对length相关属性进行四任务训练的length-model。

- 使用model1的design-model对测试集design相关的四个属性进行预测

  design相关的四个属性包括：

  - collar_design_labels 
  - lapel_design_labels
  - neck_design_labels
  - neckline_design_labels

  design-model描述：

  - 模型定义文件在【Project/src/fashionAI/nn】和【Project/src/model1_predict.py】，其结构设计：

    | Layer(type)                 | Output Shape         | Param#   | Connected to               |
    | :-------------------------- | -------------------- | -------- | -------------------------- |
    | input_2 (InputLayer)        | (None, 500, 500, 3)  | 0        |                            |
    | inception_resnet_v2 (Model) | (None, 14, 14, 1536) | 54336736 | input_2\[0][0]             |
    | global_average_pooling2d_1  | (None, 1536)         | 0        | inception_resnet_v2\[1][0] |
    | dropout_1 (Dropout)         | (None, 1536)         | 0        | global_average_pooling2d_1 |
    | collar (Dense)              | (None, 5)            | 7685     | dropout_1\[0][0]           |
    | lapel (Dense)               | (None, 5)            | 7685     | dropout_1\[0][0]           |
    | neck (Dense)                | (None, 5)            | 7685     | dropout_1\[0][0]           |
    | neckline (Dense)            | (None, 10)           | 15370    | dropout_1\[0][0]           |

    - Total params: 54,375,161
      Trainable params: 54,314,617
      Non-trainable params: 60,544
    - Layer num:8

  - 模型参数保存在【Project/models/model1】，其大小为：218.3M

  输出结果说明：输出结果包含了测试集design相关的四个属性的预测，运行时，暂存于内存中，最终会与length相关属性的预测结果整合并输出一个csv文件，保存在output目录下，文件名为：model1_result.csv

- 使用model1的length-model对测试集length相关的四个属性进行预测

  length相关的四个属性包括：

  - coat_length_labels
  - pant_length_labels
  - skirt_length_labels
  - sleeve_length_labels

  length-model描述：

  - 模型定义文件在【Project/src/fashionAI/nn】和【Project/src/model1_predict.py】，其结构设计：

    | Layer (type)                | Output Shape         | Param #  | Connected to               |
    | --------------------------- | -------------------- | -------- | -------------------------- |
    | input_2 (InputLayer)        | (None, 500, 500, 3)  | 0        |                            |
    | inception_resnet_v2 (Model) | (None, 14, 14, 1536) | 54336736 | input_2\[0][0]             |
    | global_average_pooling2d_1  | (None, 1536)         | 0        | inception_resnet_v2\[1][0] |
    | dropout_1 (Dropout)         | (None, 1536)         | 0        | global_average_pooling2d_1 |
    | coat (Dense)                | (None, 8)            | 12296    | dropout_1\[0][0]           |
    | pant (Dense)                | (None, 6)            | 9222     | dropout_1\[0][0]           |
    | skirt (Dense)               | (None, 6)            | 9222     | dropout_1\[0][0]           |
    | sleeve (Dense)              | (None, 9)            | 13833    | dropout_1\[0][0]           |

    - Total params: 54,381,309
      Trainable params: 54,320,765
      Non-trainable params: 60,544
    - Layer num:8

  - 模型参数保存在【Project/models/model1】，其大小为：218.3M

  输出结果说明：输出结果包含了测试集length相关的四个属性的预测，运行时，暂存于内存中，最终会与design相关属性的预测结果整合并输出一个csv文件，保存在output目录下，文件名为：model1_result.csv

- 其他说明：我们采用了创新性的训练策略来训练模型（具体操作见我们之后提交的技术报告），model1的design-model和length-model采用InceptionResNetV2作为基网络和同一网络改型，该模型由本团队的**陈博同学**进行实际操作和训练，送入网络的图片尺度为width = 500。预测时对输入模型的每张图片会做减均值和10crops裁剪处理，具体的裁剪方式是由四角裁剪和中心裁剪经过镜像操作得到每张图片对应的10个crops，分别预测，并最终取均值为该图片最终的预测概率（裁剪时图像的预放大阈值设置为pre-width = 300，该阈值由大量的对比实验得到）。

  ​



##### 使用单模型model2进行预测：model2包含由针对design相关属性进行四任务训练的design-model和针对length相关属性进行八任务训练的length-model。

- 使用model2的design-model对测试集design相关的四个属性进行预测 

  design相关的四个属性包括：

  - collar_design_labels 
  - lapel_design_labels
  - neck_design_labels
  - neckline_design_labels

  design-model描述：

  - 模型定义文件在Keras源代码库和预测代码【Project/src/model2_predict.py】中，其结构设计：

    | Layer (type)                   | Output Shape        | Param #  | Connected to               |
    | ------------------------------ | ------------------- | -------- | -------------------------- |
    | input_2 (InputLayer)           | (None, 499, 499, 3) | 0        |                            |
    | lambda_1 (Lambda)              | (None, 499, 499, 3) | 0        | input_2\[0][0]             |
    | inception_resnet_v2 (Model)    | (None, 1536)        | 54336736 | lambda_1\[0][0]            |
    | dropout_1 (Dropout)            | (None, 1536)        | 0        | inception_resnet_v2\[1][0] |
    | collar_design_labels (Dense)   | (None, 5)           | 7685     | dropout_1\[0][0]           |
    | lapel_design_labels (Dense)    | (None, 5)           | 7685     | dropout_1\[0][0]           |
    | neck_design_labels (Dense)     | (None, 5)           | 7685     | dropout_1\[0][0]           |
    | neckline_design_labels (Dense) | (None, 10)          | 15370    | dropout_1\[0][0]           |

    - Total params: 54,375,161
      Trainable params: 54,314,617
      Non-trainable params: 60,544
    - Layer num:8

  - 模型参数保存在【Project/models/model2】，其大小为：218M

  输出结果说明：输出结果包含了测试集design相关的四个属性的预测，运行时，暂存于内存中，最终会与length相关属性的预测结果整合并输出一个csv文件，保存在output目录下，文件名为：model2_result.csv

- 使用model2的length-model对测试集length相关的四个属性进行预测 

  length相关的四个属性包括：

  - coat_length_labels
  - pant_length_labels
  - skirt_length_labels
  - sleeve_length_labels

  design-model描述：

  - 模型定义文件在Keras源代码库和预测代码【Project/src/model2_predict.py】中，其结构设计：

    | Layer (type)                    | Output Shape        | Param #  | Connected to         |
    | ------------------------------- | ------------------- | -------- | -------------------- |
    | input_5 (InputLayer)            | (None, 499, 499, 3) | 0        |                      |
    | model1 (Model)                  | (None, 1536)        | 54336736 | input_5\[0][0]       |
    | dropout_2 (Dropout)             | (None, 1536)        | 0        | model1\[1][0]        |
    | model2 (Model)                  | (None, 128)         | 54533472 | input_5\[0][0]       |
    | concatenate_1 (Concatenate)     | (None, 1664)        | 0        | dropout_2和model2     |
    | coat_length_labelsmerged        | (None, 8)           | 13320    | concatenate_1\[0][0] |
    | pant_length_labelsmerged        | (None, 6)           | 9990     | concatenate_1\[0][0] |
    | skirt_length_labelsmerged       | (None, 6)           | 9990     | concatenate_1\[0][0] |
    | sleeve_length_labelsmerged      | (None, 9)           | 14985    | concatenate_1\[0][0] |
    | coat_length_labels_pingpumerged | (None, 8)           | 13320    | concatenate_1\[0][0] |
    | pant_length_labels_pingpumerged | (None, 6)           | 9990     | concatenate_1\[0][0] |
    | skirt_length_labels_pingpumerge | (None, 6)           | 9990     | concatenate_1\[0][0] |
    | sleeve_length_labels_pingpumerg | (None, 9)           | 14985    | concatenate_1\[0][0] |

    - Total params: 108,966,778
      Trainable params: 96,570
      Non-trainable params: 108,870,208
    - Layer num:13

  - 模型参数保存在【Project/models/model2】，其大小为：439M（包含模型结构及定义）

  输出结果说明：输出结果包含了测试集design相关的四个属性的预测，运行时，暂存于内存中，最终会与design相关属性的预测结果整合并输出一个csv文件，保存在output目录下，文件名为：model2_result.csv

- 其他说明：同样地，我们采用了创新性地训练策略来训练模型（具体操作见我们之后提交的技术报告），在model2的design-model和length-model采用InceptionResNetV2作为基网络和同一网络改型，该模型由本团队的**孙立波同学**进行实际操作和训练，送入网络的图片尺度为width =499。其中，model2的design-model与陈博同学提供的model1的design-model和length-model所使用的网络架构和训练策略基本一致。

  需要注意的是，考虑到length相关属性中存在“模特”域和“平铺”域的不一致性和复杂性，以及大量的m标记问题，**我们model2的length-model基于InceptionResNetV2网络架构，设计了双通道的length属性特征提取网络**。我们的预处理模型可以很好的识别预测图片是模特还是平铺图片，基于这样的粗粒度识别，我们训练了基于多任务属性分类的模特识别模型和基于多任务属性分类的平铺图片识别模型，最后以两个模型的权重赋值给基于八任务属性分类的length-model进行更有效的特征提取，从而取得了更佳的识别效果。

  更多地，在我们使用单模型model2预测时，对输入模型的每张图片会做10crops裁剪处理和图片微角度的随机旋转处理，具体的裁剪方式是由四角裁剪和中心裁剪经过镜像操作得到每张图片对应的10个crops，然后对其做随机的旋转处理，分别进行预测，并最终取均值为该图片最终的预测概率（裁剪和旋转时图像的预放大阈值设置：模特pre-width = 512和平铺图片pre-width = 524,阈值由大量的对比实验得到）。



#### 整个预测Pipeline综述

**1.**由单模型model1进行testb数据集上的预测，产出model1_result.csv；(输入图片处理：减均值+10crops裁剪)

**2.**由单模型model2进行testb数据集上的预测，产出model2_result.csv; (图片处理：10crops+random rotation）

**3.**对于单模型model1与单模型model2分别在测试集上预测输出的两个csv文件采取简单的“平均融合的策略”，该融合满足官方的融合策略限制和要求，最终在根目录下或指定的输出目录下输出“result20180602_010816.csv样式”的结果文件。
