# FaceRecognition_facenet
基于facenet的人脸识别的实现

### 准备工作
1. 环境配置
> 本人实现环境：Windows 10（20H2）、Python 3.8、TensorFlow-GPU 2.3、numpy 1.19、matplotlib 3.3、keras 2.4.3、cudataoolkit 10.1.243、cudnn 7.6.5.
另显卡型号GeForce GTX 1050Ti，驱动版本号为466.11.

2. 文件添加
> 向haarcascades文件夹中添加OpenCV的级联器文件，主要是有关人脸检测相关的几个级联文件，.xml格式


#### 使用
1. 添加人脸，运行face_colletor.py
2. 识别，运行 face_recognize.py
