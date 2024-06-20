# 图像质量评估 - IQA
一个基于CNN和Transformer的图像质量评估模型
无关美学特征，依据清晰度等进行评分

回归预测模型，通过对CNN引入Transformer来提升模型性能

主要是在CNN中间层，引入Transformer来学习更多的全局特征

## 数据集

需要下载相应地公开数据才能使用

LIVE数据集
KONIQ数据集
CSIQ数据集
LIVEC数据集
BID数据集


## 模型依赖
pytorch

## 模型运行
```
nohup python train.py --dataset=livec --train_test_num=5 --cudas=0 --use_seed --random_single > /dev/null 2> e.txt &
```
