# 图像质量评估 - IQA
一个基于CNN和Transformer的图像质量评估模型

无关美学特征，依据清晰度等进行评分

主要是在CNN中间层，引入Transformer来学习更多的全局特征
## 背景
在各种图像数据不断生成的生活中，人们需要在大量的图像中挑选出可以使用的、符合自己要求的图像，使用算法/深度学习来解决，可以缓解人力不足的问题。
## 数据集
图像质量评估数据集有着许多公开数据，这里采用公开数据进行实验。

需要下载相应地公开数据才能使用

LIVE数据集

KONIQ数据集

CSIQ数据集

LIVEC数据集

BID数据集

## 模型
回归预测模型，通过对CNN引入Transformer来提升模型性能。

CNN对于局部视野有着良好的性能，但是图像的评分不仅仅依据局部内容，也需要对整体内容进行学习。因此通过Transformer加入更多的全局特征来提升模型性能。

## 优化器
采用的Adam优化器，方便训练

## 模型依赖
pytorch

## 模型运行
```
nohup python train.py --dataset=livec --train_test_num=5 --cudas=0 --use_seed --random_single > /dev/null 2> e.txt &
```
