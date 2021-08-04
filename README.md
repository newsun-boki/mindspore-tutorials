# 华为mindspore 学习过程

------------------------------

## 环境安装

建议上官网使用pip自行安装 https://www.mindspore.cn/install/

## 开始实战

#### step1 下载数据集

首先需要下载Mnist数据集并放在datasets下
  
```bash 
mkdir -p ./datasets/MNIST_Data/train ./datasets/MNIST_Data/test
wget -NP ./datasets/MNIST_Data/train https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte --no-check-certificate
wget -NP ./datasets/MNIST_Data/train https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte --no-check-certificate
wget -NP ./datasets/MNIST_Data/test https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte --no-check-certificate
wget -NP ./datasets/MNIST_Data/test https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte --no-check-certificate
tree ./datasets/MNIST_Data
```

```bash
./datasets/MNIST_Data
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte

2 directories, 4 files
```

---

#### step2 训练

找到lenet文件夹下的lenet.py文件进行训练

```bash
cd lenet
python lenet.py -device_target=CPU
```

#### step3 检测
运行lenet_test.py调用训练好的模型进行检测

```bash 
python lenet_test.py
```
 