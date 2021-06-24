# -*- conding:utf-8 -*-


from facenet import Facenet

import os
import pickle
import numpy as np
import cv2

from threading import Thread
from PIL import Image


# 人脸特征值存放的文件
features_path = "E:/Reference/facenet-new/facedate_person/features_dataset/features.dataset"
# 人脸检测 Haar级联文件
classfier_paths = ["E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_alt.xml",
                   "E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_alt2.xml",
                   "E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_alt_tree.xml",
                   "E:/Reference/facenet-new/harrcascades/haarcascade_frontalface_default.xml"]
# 人名
current_name = []

# image 和数据库中的所有向量对比
def _thread_recogizer(model, frame, faceRects):
    global current_name
    with open(features_path, "rb") as file:
        features = pickle.load(file)

    # 添加标记
    flag = False
    # 当前出现的人脸的特征向量
    reference_features = []
    for x, y, w, h in faceRects:
        # opencv 格式转换成 image
        image = Image.fromarray(cv2.cvtColor(frame[y-10:y+h+10, x-10:x+w+10], cv2.COLOR_BGR2RGB))
        reference_features.append(model.get_tensor(image))

    for reference_feature in reference_features:
        flag = False
        for name, feature in features.items():
            l1 = np.sqrt(np.sum(np.square(reference_feature - feature), axis=-1))
            if l1 < 0.9:
                current_name.append(name)
                flag = True
                break
        if not flag:
            current_name.append("Unknow")

    # 输出识别到的人脸
    if len(current_name) > 0:
        for i in range(len(faceRects)):
            x, y, w, h = faceRects[i]
            cv2.imshow(current_name[i] ,frame[y-10:y+h+10, x-10:x+w+10])
        print(current_name)
        current_name = []
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("jiancewancheng")


class myThread(Thread):
    def __init__(self, model, frame, faceRects):
        Thread.__init__(self)
        self.model = model
        self.frame = frame
        self.faceRects = faceRects

    def run(self):
        _thread_recogizer(self.model, self.frame, self.faceRects)

# 人脸识别
def face_recognize(model):
    global current_name
    classfier = cv2.CascadeClassifier(classfier_paths[0])
    color = (255, 0, 0)

    cv2.namedWindow('frame')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        key = cv2.waitKey(1) & 0xff
        if not ret:
            print("Capture is not opened!")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        key = cv2.waitKey(1) & 0xff
        if len(faceRects) > 0:
            if key == ord('b'):
                # 备份一份数据
                frame_new = frame.copy()
                try:
                    # print(len(faceRects))
                    th = myThread(model, frame_new, faceRects)
                    th.start()
                    # th.join()
                except:
                    print('Error: 无法启动线程')


            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x + w - 15, y + h + 15), color, 2)

        cv2.imshow('frame', frame)

        if key == 27:
            break


    cap.release()
    cv2.destroyAllWindows()


def add():
    cap = cv2.VideoCapture(1)
    cap.set()
# ————————————————————————————————#
# 以下为测试代码
# ————————————————————————————————#
def test_recognizer():
    try:
        image = Image.open("E:/Reference/facenet-new/facedate_person/pengyuyan_000.jpg")
    except:
        print("error")
    # print(type(image))
    model = Facenet()
    # recognizer(image, model)
    print(current_name)
    if current_name != None:
        print(current_name)


def test_thread_():
    model = Facenet()
    face_recognize(model)

if __name__ == "__main__":
    test_thread_()