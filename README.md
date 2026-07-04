# -基于yolov8实现的火焰/烟雾识别（支持图片和视频检测）-
***
## Introduction
🎯配置环境
初次部署

可以先在pycharm里配置好环境pytorch（b站有教程），之后可以先让ai生成一份训练代码，或者直接下载我训练代码里的直接用

模型我都放在release里了，3个模型，表现最好的是模型2，模型3加入的psa注意力纯当练手，表现一般

🔥模型介绍

模型1：数据集容量在6000左右，训练集占90%，验证和测试全部加起来仅10%

数据集标注的也有问题，标注基本上都分布在居中和居中左右位置，边界标注基本没有

各项指标如图，map仅0.6，而且出现了过拟合，能够识别大部分火焰场景，但是置信度偏低


<img src="asset/model1/results.png" width="800" alt="监控曲线">
<img src="asset/model1/confusion_matrix_normalized.png" height="250" alt="归一化混淆"> <img src="asset/model1/labels.jpg" height="250" alt="label">
</div>
