from mindspore.train.serialization import load_checkpoint, load_param_into_net
from lenet import LeNet5
from lenet import create_dataset
from mindspore import Model
import os
import numpy as np
from mindspore import Tensor

mnist_path = "../datasets/MNIST_Data"
net = LeNet5();
# 加载已经保存的用于测试的模型
param_dict = load_checkpoint("./checkpoints/checkpoint_lenet-5_1875.ckpt")
# 加载参数到网络中
load_param_into_net(net, param_dict)

net = LeNet5();

# 定义测试数据集，batch_size设置为1，则取出一张图片
ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=1).create_dict_iterator()
data = next(ds_test)

# images为测试图片，labels为测试图片的实际分类
images = data["image"].asnumpy()
labels = data["label"].asnumpy()

# 使用函数model.predict预测image对应分类
model = Model(net)
output = model.predict(Tensor(data['image']))
predicted = np.argmax(output.asnumpy(), axis=1)

# 输出预测分类与实际分类
print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')

