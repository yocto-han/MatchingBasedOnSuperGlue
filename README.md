### 中国科学院自动化研究所复杂系统管理与控制国家重点实验室

# LearningBasedSemanticMatching

## Introduction
本工程尝试用一个基于学习策略的框架完成匹配任务。

* Authors: *韩钰, 陆昱辰, 杨旭, 曾少峰,杨菁*


## Dependencies
* Python 3 >= 3.5
* PyTorch >= 1.1
* OpenCV >= 3.4 (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib >= 3.1
* NumPy >= 1.18
* 其他详见requirements.txt

## Contents

1. `model.py` : 执行后运行训练，训练集为data172，在训练结束后，会选择训练集上准确率最高的参数进行测试
2. `dataloader.py`: 生成离散训练数据集，目录为data172
3. `dataloaderfortest.py`: 生成离散测试数据集，目录为test
4. `results`: 训练过程结束后存储匹配结果
5. `parameter`: 模型存储
6. `DescriptorAnalysis`: 模型前端输出的深度描述符的一些分析
7. `dataAugmented`: 数据增广实践
8. `multiGraphMatching`: 多图匹配实践